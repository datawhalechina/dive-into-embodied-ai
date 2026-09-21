"""Run with pytest after copying recap/ into OpenPI; no checkpoints or GPU needed."""

import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from examples.recap import annotate
from examples.recap.config import (
    AdvantagePrompt,
    RecapInputs,
    ValueTarget,
    get_configs,
    register,
)
from examples.recap.value_model import ValueConfig, two_hot
from openpi import transforms
from openpi.models import model
from openpi.training import config


def test_inference_worker_transforms_collates_and_pads():
    import torch

    items = [
        (np.array([3, index]), {"state": torch.tensor([index, index + 1]), "actions": torch.arange(7)[None]})
        for index in range(2)
    ]
    keys, batch = annotate.inference_collate(items)
    np.testing.assert_array_equal(keys, [[3, 0], [3, 1]])
    assert isinstance(batch["actions"], np.ndarray)
    assert batch["actions"].shape == (2, 1, 7)
    padded = annotate.pad_batch(batch, 4)
    np.testing.assert_array_equal(padded["state"], [[0, 1], [1, 2], [1, 2], [1, 2]])


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("n_step", [1, 3, 10])
def test_perfect_values_have_zero_advantage_including_terminal(success, n_step):
    targets = annotate.value_targets(5, success)
    np.testing.assert_allclose(annotate.n_step_advantages(targets, targets, n_step), 0, atol=1e-7)


def test_failure_penalty_is_retained_at_terminal():
    targets = annotate.value_targets(3, False, failure_penalty=3, return_scale=10)
    np.testing.assert_allclose(targets, [-0.5, -0.4, -0.3])
    np.testing.assert_allclose(annotate.n_step_advantages(targets, np.zeros(3), 2), [-0.2, -0.4, -0.3])


def test_bootstrap_uses_nth_observation():
    targets = np.array([-0.3, -0.2, -0.1, 0])
    values = np.array([-0.4, -0.25, -0.12, -0.01])
    np.testing.assert_allclose(annotate.n_step_advantages(targets, values, 2), [0.08, 0.04, 0.02, 0.01])


def test_thresholds_are_per_task_and_keep_ties():
    labels, thresholds = annotate.positive_labels(np.array([0, 1, 2, 100, 100, 100]), np.array([0, 0, 0, 1, 1, 1]), 0.3)
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 1, 1])
    assert thresholds[0] < thresholds[1]


def test_no_implicit_success_or_silent_clipping():
    with pytest.raises(ValueError):
        annotate.value_targets(3, None)
    with pytest.raises(ValueError):
        annotate.value_targets(3, "false")
    with pytest.raises(ValueError):
        annotate.value_targets(10, False, return_scale=1)


def test_two_hot_preserves_expectation_and_endpoints():
    values = jnp.array([-1.0, -0.752, -0.005, 0.0])
    probs = np.asarray(two_hot(values))
    np.testing.assert_allclose(probs.sum(-1), 1, atol=1e-6)
    np.testing.assert_allclose(probs @ np.linspace(-1, 0, 201), values, atol=1e-6)
    assert (probs >= 0).all()


def test_value_forward_and_loss_shapes_with_upstream_backbone():
    cfg = ValueConfig(pi05=True, action_horizon=10, discrete_state_input=False)

    def forward():
        value = cfg.create(jax.random.key(0))
        observation = cfg.fake_obs(batch_size=2)
        actions = -jnp.ones((2, 10, 32)) * 0.5
        return value.compute_loss(jax.random.key(1), observation, actions), value.predict_value(observation)

    loss, prediction = jax.eval_shape(forward)
    assert loss.shape == (2, 1)
    assert prediction.shape == (2,)


def test_prompt_conditions_training_and_inference():
    assert AdvantagePrompt(0)({"prompt": "task", "is_positive": 0})["prompt"].endswith("negative")
    assert AdvantagePrompt(1)({"prompt": "task", "is_positive": 1})["prompt"] == "task"
    # Even with dropout=1, serving has a deterministic positive condition.
    for _ in range(5):
        assert AdvantagePrompt(1)({"prompt": "task"})["prompt"].endswith("positive")
    with pytest.raises(ValueError):
        AdvantagePrompt()({"prompt": "task", "is_positive": -1})


def test_label_survives_libero_and_is_not_action_normalized():
    sample = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(8),
        "actions": np.ones((10, 7)),
        "prompt": "task",
        "value_target": -0.75,
    }
    data = RecapInputs(model.ModelType.PI05)(sample)
    stats = transforms.NormStats(mean=np.full(7, 10), std=np.full(7, 2))
    data = transforms.Normalize({"actions": stats})(data)
    data = transforms.PadStatesAndActions(32)(data)
    data = ValueTarget()(data)
    assert data["actions"].shape == (10, 32)
    np.testing.assert_allclose(data["actions"], -0.75)
    assert "value_target" not in data


def test_register_and_value_freeze_filter():
    original = config.get_config("pi05_libero")
    register()
    assert config.get_config("pi05_libero") is original
    entries = get_configs()
    value = next(c for c in entries if c.name.endswith("value"))
    for path, frozen in [
        ("PaliGemma/img/kernel", False),
        ("PaliGemma/llm/layers/attn/kernel", False),
        ("PaliGemma/llm/layers/attn/kernel_1", True),
        ("value_fc1/kernel", False),
        ("action_in_proj/kernel", True),
    ]:
        assert value.freeze_filter(tuple(path.split("/")), None) is frozen


@pytest.mark.parametrize("stage", ["sft", "value", "acp"])
def test_upstream_training_initialization_and_backward_trace(stage):
    from openpi.training import sharding
    from scripts.train import init_train_state, train_step

    cfg = next(c for c in get_configs() if c.name == f"pi05_recap_{stage}")
    assert cfg.model.discrete_state_input is True
    # Exercise the real trainer, including frozen dtype conversion, gradients,
    # optimizer and EMA. Shape tracing does not allocate the full model on CPU.
    state, _ = init_train_state(cfg, jax.random.key(0), sharding.make_mesh(1), resume=True)
    updated, metrics = jax.eval_shape(
        lambda state, batch: train_step(cfg, jax.random.key(1), state, batch),
        state,
        cfg.model.inputs_spec(batch_size=1),
    )
    assert updated.step.shape == () and metrics["loss"].shape == ()
    assert metrics["grad_norm"].dtype == jnp.float32


def make_episode(tmp_path, success=True):
    (tmp_path / "meta").mkdir()
    (tmp_path / "data").mkdir()
    path = tmp_path / "data/episode_000000.parquet"
    pq.write_table(
        pa.table({"episode_index": [0] * 3, "frame_index": [0, 1, 2], "task_index": [0] * 3}),
        path,
    )
    (tmp_path / "meta/info.json").write_text(json.dumps({"features": {}}))
    (tmp_path / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 3, "success": success}) + "\n"
    )
    return path


def test_targets_cli_updates_parquet_and_metadata(tmp_path, monkeypatch):
    path = make_episode(tmp_path, success=False)
    monkeypatch.setattr(sys, "argv", ["annotate", "targets", "--dataset-root", str(tmp_path)])
    annotate.main()
    table = pq.read_table(path)
    np.testing.assert_allclose(np.asarray(table["value_target"]), [-302 / 820, -301 / 820, -300 / 820])
    assert json.loads((tmp_path / "meta/info.json").read_text())["features"]["value_target"]["shape"] == [1]


def test_missing_success_fails_before_writing(tmp_path, monkeypatch):
    path = make_episode(tmp_path, success=None)
    monkeypatch.setattr(sys, "argv", ["annotate", "targets", "--dataset-root", str(tmp_path)])
    with pytest.raises(ValueError):
        annotate.main()
    assert "value_target" not in pq.read_schema(path).names


def test_advantages_cli_preserves_frame_alignment(tmp_path, monkeypatch):
    path = make_episode(tmp_path)
    annotate.write_columns(path, {"value_target": np.array([-0.2, -0.1, 0], dtype=np.float32)})
    # Predictions arrive in deliberately shuffled key order.
    monkeypatch.setattr(
        annotate,
        "predict_values",
        lambda *args: {(0, 2): -0.01, (0, 0): -0.4, (0, 1): -0.2},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotate",
            "advantages",
            "--dataset-root",
            str(tmp_path),
            "--checkpoint",
            "unused",
            "--n-step",
            "1",
        ],
    )
    annotate.main()
    table = pq.read_table(path)
    np.testing.assert_allclose(np.asarray(table["predicted_value"]), [-0.4, -0.2, -0.01])
    np.testing.assert_allclose(np.asarray(table["advantage"]), [0.1, 0.09, 0.01], atol=1e-7)
    np.testing.assert_array_equal(np.asarray(table["is_positive"]), [1, 0, 0])


def test_latest_checkpoint_uses_numeric_order_and_ignores_partial_steps(tmp_path):
    from examples.recap.run import latest_checkpoint

    for step in ("9", "20", "100.orbax-checkpoint-tmp"):
        (tmp_path / "sft/exp" / step / "params").mkdir(parents=True)
    (tmp_path / "sft/exp/200").mkdir()
    assert latest_checkpoint("sft", "exp", tmp_path).name == "20"
    with pytest.raises(ValueError):
        latest_checkpoint("missing", "exp", tmp_path)


def test_collection_and_evaluation_states_are_disjoint():
    from examples.recap.rollout import initial_state_ids

    assert set(initial_state_ids("collect", 30)).isdisjoint(initial_state_ids("eval", 20))
    with pytest.raises(ValueError):
        initial_state_ids("eval", 21)


def make_published_split(raw_dir, split, success):
    import av
    from examples.recap import data

    source = raw_dir / f"libero10_task0_{split}"
    (source / "meta").mkdir(parents=True)
    (source / "data/chunk-000").mkdir(parents=True)
    task = "turn on the stove and put the moka pot on it" if split == "sft" else data.TASK
    features = {
        key: {"dtype": dtype, "shape": shape, "names": None}
        for key, dtype, shape in (
            ("state", "float32", [8]),
            ("actions", "float32", [7]),
            ("timestamp", "float32", [1]),
            ("frame_index", "int64", [1]),
            ("episode_index", "int64", [1]),
            ("index", "int64", [1]),
            ("task_index", "int64", [1]),
        )
    }
    for key in ("image", "wrist_image"):
        features[key] = {
            "dtype": "video",
            "shape": [32, 32, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": 10, "video.codec": "mpeg4"},
        }
        video = source / f"videos/chunk-000/{key}/episode_000000.mp4"
        video.parent.mkdir(parents=True)
        with av.open(str(video), "w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for _ in range(2):
                frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24")
                container.mux(stream.encode(frame))
            container.mux(stream.encode())
    info = {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "fps": 10,
        "chunks_size": 1000,
        "total_episodes": 1,
        "total_frames": 2,
        "total_tasks": 1,
        "total_chunks": 9,
        "total_videos": 99,
        "splits": {"train": "0:128"},
        "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    data.write_jsonl(source / "meta/tasks.jsonl", [{"task_index": 0, "task": task}])
    data.write_jsonl(
        source / "meta/episodes.jsonl",
        [
            {"episode_index": 0, "length": 2, "tasks": [task], "is_success": success},
        ],
    )
    table = pa.table(
        {
            "state": [np.zeros(8, dtype=np.float32)] * 2,
            "actions": [np.ones(7, dtype=np.float32) * 0.25] * 2,
            "timestamp": np.array([0, 0.1], dtype=np.float32),
            "frame_index": [0, 1],
            "episode_index": [0, 0],
            "index": [123, 124],
            "task_index": [0, 0],
            "done": [False, True],
            "is_success": [False, success],
            "return": [-99.0, -98.0],
        }
    ).replace_schema_metadata({b"huggingface": b'{"info":{"features":{"missing":{"dtype":"float32"}}}}'})
    pq.write_table(table, source / "data/chunk-000/episode_000000.parquet")
    return source


@pytest.mark.parametrize("split,success", [("sft", True), ("train", False), ("eval", True)])
def test_prepare_published_splits_use_real_lerobot_and_preserve_raw(tmp_path, monkeypatch, split, success):
    from examples.recap import data
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    raw = tmp_path / "raw"
    source = make_published_split(raw, split, success)
    source_table = source / "data/chunk-000/episode_000000.parquet"
    original = source_table.read_bytes()
    monkeypatch.setattr(data, "dataset_root", lambda repo: tmp_path / repo)
    data.prepare(raw, split, f"local/{split}")
    root = tmp_path / f"local/{split}"
    dataset = LeRobotDataset(f"local/{split}", root=root, video_backend="pyav")
    assert len(dataset) == 2 and dataset.num_episodes == 1
    np.testing.assert_allclose(dataset[0]["actions"], 0.25)
    assert dataset[0]["image"].shape == (3, 32, 32)
    stream = annotate.InferenceDataset(dataset, lambda sample: sample, np.arange(2))
    streamed = list(stream)
    assert [key.tolist() for key, _ in streamed] == [[0, 0], [0, 1]]
    assert all(sample["image"].shape == (3, 32, 32) for _, sample in streamed)
    assert dataset[0]["index"] == 0
    assert "return" not in dataset[0]
    assert root.joinpath("videos").is_symlink()
    assert data.read_jsonl(root / "meta/episodes.jsonl")[0]["success"] is success
    info = json.loads((root / "meta/info.json").read_text())
    assert info["splits"] == {"train": "0:1"} and info["total_videos"] == 2 and info["total_chunks"] == 1
    assert source_table.read_bytes() == original
    with pytest.raises(FileExistsError):
        data.prepare(raw, split, f"local/{split}")


def test_download_validation_rejects_offline_fallback_directory(tmp_path, monkeypatch):
    import huggingface_hub
    from examples.recap import data

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(tmp_path))
    with pytest.raises(RuntimeError, match="incomplete"):
        data.download(tmp_path)
    # Keep the marker and partial files so the next invocation can resume.
    assert (tmp_path / "INCOMPLETE").is_file()


def test_download_validation_accepts_three_complete_splits(tmp_path, monkeypatch):
    from examples.recap import data

    raw = tmp_path / "raw"
    for split, success in (("sft", True), ("train", False), ("eval", True)):
        make_published_split(raw, split, success)
    monkeypatch.setattr(data, "EXPECTED_EPISODES", dict.fromkeys(data.REPOS, 1))
    data.validate_download(raw)


def test_eval_cannot_be_used_for_training_or_advantage_thresholds(tmp_path, monkeypatch):
    import dataclasses

    import lerobot.common.constants as constants

    make_episode(tmp_path)
    (tmp_path / "recap_source.json").write_text(json.dumps({"split": "eval"}))
    monkeypatch.setattr(constants, "HF_LEROBOT_HOME", tmp_path.parent)
    for cfg in get_configs():
        factory = dataclasses.replace(cfg.data, repo_id=tmp_path.name)
        with pytest.raises(ValueError, match="held-out"):
            factory.create(cfg.assets_dirs, cfg.model)
    monkeypatch.setattr(sys, "argv", ["annotate", "advantages", "--dataset-root", str(tmp_path)])
    with pytest.raises(SystemExit):
        annotate.main()


def test_value_evaluation_uses_aligned_held_out_labels_without_mutating_them(tmp_path, monkeypatch):
    path = make_episode(tmp_path)
    (tmp_path / "recap_source.json").write_text(json.dumps({"split": "eval"}))
    annotate.write_columns(path, {"value_target": np.array([-0.2, -0.1, 0], dtype=np.float32)})
    original = path.read_bytes()
    monkeypatch.setattr(annotate, "predict_values", lambda *args: {(0, 2): -0.1, (0, 0): -0.4})
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotate",
            "evaluate",
            "--dataset-root",
            str(tmp_path),
            "--checkpoint",
            "unused",
            "--output",
            str(output),
        ],
    )
    annotate.main()
    metrics = json.loads(output.read_text())
    assert metrics["frames"] == 2
    assert metrics["mse"] == pytest.approx(0.025)
    assert metrics["mae"] == pytest.approx(0.15)
    assert path.read_bytes() == original


def test_rollout_records_pre_action_observations_and_only_executed_actions(monkeypatch):
    from types import SimpleNamespace

    from examples.recap import rollout

    # Only the upstream helper constant is needed; avoid importing the simulator.
    monkeypatch.setitem(sys.modules, "examples.libero.main", SimpleNamespace(LIBERO_DUMMY_ACTION=[0] * 7))
    monkeypatch.setattr(
        rollout,
        "observation_fields",
        lambda obs: {
            "image": np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "state": np.full(8, obs),
        },
    )

    class Environment:
        steps = 0

        def reset(self):
            self.steps = 0

        def set_init_state(self, state):
            return 0

        def step(self, action):
            self.steps += 1
            return self.steps, 0, self.steps == 12, {}

        def check_success(self):
            return self.steps == 12

    class Client:
        def reset(self):
            pass

        def infer(self, obs):
            return {"actions": np.arange(70).reshape(10, 7)}

    frames, success = rollout.run_episode(Environment(), Client(), None, "task")
    assert success and len(frames) == 2
    np.testing.assert_array_equal(frames[0]["state"], np.full(8, 10))
    np.testing.assert_array_equal(frames[1]["actions"], np.arange(7, 14))

    class ValueClient:
        def reset(self):
            pass

        def infer(self, obs):
            assert obs["prompt"] == "task"
            return {"value": -0.25}

    import io

    samples, log = [], io.StringIO()
    rollout.run_episode(Environment(), Client(), None, "task", value_client=ValueClient(), value_samples=samples, value_log=log)
    assert samples == [{"step": 0, "value": -0.25}]
    assert json.loads(log.getvalue()) == samples[0]
