"""Value targets and n-step advantage labels for a COPY of a LeRobot v2 dataset."""

import argparse
import dataclasses
import json
import os
import pathlib
import time

# This module is imported before JAX for annotation commands. Growing the GPU
# pool on demand leaves room for CUDA video/worker libraries and avoids JAX's
# default large reservation on every visible inference device.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


class InferenceDataset(torch.utils.data.IterableDataset):
    """Decode each episode once, then transform and yield its selected frames."""

    def __init__(self, dataset, transform, indices):
        self.dataset = dataset
        self.transform = transform
        self.selected = np.zeros(len(dataset), dtype=np.bool_)
        self.selected[np.asarray(indices, dtype=np.int64)] = True

    def __len__(self):
        return int(self.selected.sum())

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id, worker_count = (0, 1) if worker is None else (worker.id, worker.num_workers)
        for episode in range(worker_id, self.dataset.num_episodes, worker_count):
            start = int(self.dataset.episode_data_index["from"][episode])
            end = int(self.dataset.episode_data_index["to"][episode])
            absolute_indices = np.flatnonzero(self.selected[start:end]) + start
            if not len(absolute_indices):
                continue

            samples = [self.dataset.hf_dataset[int(index)] for index in absolute_indices]
            timestamps = [float(sample["timestamp"]) for sample in samples]
            query_timestamps = {key: timestamps for key in self.dataset.meta.video_keys}
            videos = self.dataset._query_videos(query_timestamps, episode)
            # LeRobot squeezes the batch axis when only one timestamp is queried.
            videos = {
                key: frames[None] if frames.ndim == 3 else frames
                for key, frames in videos.items()
            }

            for offset, sample in enumerate(samples):
                sample = {**{key: frames[offset] for key, frames in videos.items()}, **sample}
                sample["task"] = self.dataset.meta.tasks[int(sample["task_index"])]
                key = np.asarray([sample["episode_index"], sample["frame_index"]], dtype=np.int64)
                # Value inference does not consume actions. A one-step dummy
                # keeps the normal OpenPI input transforms reusable.
                sample["actions"] = np.asarray(sample["actions"])[None, :]
                yield key, self.transform(sample)


def inference_collate(items):
    """Stack transformed samples as NumPy so workers do not share Torch FDs."""

    keys, samples = zip(*items, strict=True)

    def stack(values):
        first = values[0]
        if isinstance(first, dict):
            return {key: stack([value[key] for value in values]) for key in first}
        if isinstance(first, tuple):
            return tuple(stack([value[i] for value in values]) for i in range(len(first)))
        if isinstance(first, list):
            return [stack([value[i] for value in values]) for i in range(len(first))]
        return np.stack([np.asarray(value) for value in values])

    return np.stack(keys), stack(list(samples))


def init_inference_worker(_worker_id):
    """Keep decode/transform workers off CUDA; only the parent runs the model."""

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def pad_batch(batch, size):
    """Repeat the final item so the last batch keeps one compiled shape."""

    def first_leaf(value):
        if isinstance(value, dict):
            return first_leaf(next(iter(value.values())))
        if isinstance(value, (tuple, list)):
            return first_leaf(value[0])
        return value

    length = len(first_leaf(batch))
    if length == size:
        return batch
    if not 0 < length < size:
        raise ValueError(f"Cannot pad batch of length {length} to {size}")

    def pad(value):
        if isinstance(value, dict):
            return {key: pad(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(pad(item) for item in value)
        if isinstance(value, list):
            return [pad(item) for item in value]
        value = np.asarray(value)
        return np.concatenate([value, np.repeat(value[-1:], size - length, axis=0)])

    return pad(batch)


def episode_columns(path):
    table = pq.read_table(path, columns=["episode_index", "frame_index", "task_index"])
    episode, frame, task = (np.asarray(table[key]) for key in table.column_names)
    if len(frame) == 0 or len(np.unique(episode)) != 1 or len(np.unique(task)) != 1:
        raise ValueError(f"Expected one nonempty, single-task episode per parquet: {path}")
    if not np.array_equal(frame, np.arange(len(frame))):
        raise ValueError(f"Frames must be ordered and contiguous from zero: {path}")
    return int(episode[0]), int(task[0]), frame


def value_targets(length, success, failure_penalty=300.0, return_scale=820.0):
    if length < 1 or not isinstance(success, bool):
        raise ValueError("Expected a positive episode length and an explicit boolean success label")
    if not np.isfinite([failure_penalty, return_scale]).all() or failure_penalty < 0 or return_scale <= 0:
        raise ValueError("failure_penalty must be nonnegative; return_scale must be positive")
    returns = -np.arange(length - 1, -1, -1, dtype=np.float32)
    if not success:
        returns -= failure_penalty
    if np.min(returns) < -return_scale:
        raise ValueError("return_scale is too small; use the same larger scale for all datasets")
    return returns / return_scale


def n_step_advantages(targets, values, n_step=10):
    """Gamma=1, one complete episode. Terminal bootstrap is exactly zero.

    G_t is the normalized Monte Carlo return. r_t = G_t - G_(t+1),
    and the final reward is G_(T-1), retaining the failure penalty.
    """
    targets, values = np.asarray(targets), np.asarray(values)
    if n_step < 1 or targets.ndim != 1 or targets.shape != values.shape or not len(targets):
        raise ValueError("Expected matching nonempty 1-D arrays and n_step >= 1")
    if not np.isfinite(targets).all() or not np.isfinite(values).all():
        raise ValueError("Targets and predictions must be finite")
    end = np.minimum(np.arange(len(targets)) + n_step, len(targets))
    # Telescoping sum of rewards; no bootstrap crosses an episode boundary.
    return targets - np.append(targets, 0)[end] + np.append(values, 0)[end] - values


def positive_labels(advantages, tasks, positive_ratio=0.3):
    if not 0 < positive_ratio < 1:
        raise ValueError("positive_ratio must be strictly between 0 and 1")
    indicators = np.zeros(len(advantages), dtype=np.int64)
    thresholds = {}
    for task in np.unique(tasks):
        mask = tasks == task
        threshold = float(np.quantile(advantages[mask], 1 - positive_ratio))
        indicators[mask] = advantages[mask] >= threshold
        thresholds[int(task)] = threshold
    return indicators, thresholds


def write_columns(path, columns):
    table = pq.read_table(path)
    for name, values in columns.items():
        array = pa.array(values)
        if name in table.column_names:
            table = table.set_column(table.column_names.index(name), name, array)
        else:
            table = table.append_column(name, array)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)


def update_features(root, columns):
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    for name in columns:
        info["features"][name] = {
            "dtype": "int64" if name == "is_positive" else "float32",
            "shape": [1],
            "names": None,
        }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(info, indent=2) + "\n")
    temporary.replace(path)


def predict_values(root, checkpoint, batch_size, max_frames=None, num_workers=4):
    import jax
    import jax.numpy as jnp
    from examples.recap.config import get_configs
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils
    from openpi.training import checkpoints

    cfg = next(c for c in get_configs() if c.name == "pi05_recap_value")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    # Checkpoint assets are authoritative, even if current local stats changed.
    data_cfg = dataclasses.replace(
        data_cfg,
        norm_stats=checkpoints.load_norm_stats(checkpoint / "assets", data_cfg.asset_id),
    )
    devices = jax.devices()
    if batch_size % len(devices):
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by the number of visible JAX devices ({len(devices)})"
        )
    mesh = jax.sharding.Mesh(np.asarray(devices), ("batch",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_parallel = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    model = cfg.model.load(
        model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16, sharding=replicated)
    )
    model.eval()
    predict = nnx_utils.module_jit(
        model.predict_value,
        in_shardings=(replicated, data_parallel),
        out_shardings=replicated,
    )
    # No action chunk is needed for value inference; use a single dummy action
    # sequence, still keeping the normal observation preprocessing.
    dataset = LeRobotDataset(cfg.data.repo_id, root=root)
    prompt = transforms.PromptFromLeRobotTask(dataset.meta.tasks)
    transform = transforms.compose(
        [
            prompt,
            *data_cfg.repack_transforms.inputs,
            *data_cfg.data_transforms.inputs,
            transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            *data_cfg.model_transforms.inputs,
        ]
    )
    predictions = {}
    indices = np.arange(len(dataset))
    if max_frames is not None and max_frames < len(indices):
        # Same evenly spaced frames across checkpoints, covering the entire split.
        indices = np.linspace(0, len(dataset) - 1, max_frames, dtype=int)

    # Video decoding is substantially slower than reading the small parquet
    # tables. Decode and prefetch samples in worker processes while the main
    # process runs transforms and value inference on the GPU.
    selected_dataset = InferenceDataset(dataset, transform, indices)
    loader_kwargs = {
        "dataset": selected_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "collate_fn": inference_collate,
        "worker_init_fn": init_inference_worker,
    }
    if num_workers > 0:
        loader_kwargs.update(multiprocessing_context="spawn", prefetch_factor=2)
    loader = torch.utils.data.DataLoader(**loader_kwargs)

    warmup_started = time.monotonic()
    steady_started = None
    steady_frames = 0
    processed_count = 0
    print(
        f"Value inference uses {len(devices)} device(s), global batch {batch_size}, "
        f"{num_workers} transform/decode worker(s)",
        flush=True,
    )
    for batch_index, (key_array, batch) in enumerate(loader):
        count = len(key_array)
        keys = [(int(key[0]), int(key[1])) for key in key_array]
        batch = pad_batch(batch, batch_size)
        values = np.asarray(predict(model_lib.Observation.from_dict(batch)))[:count]
        for key, value in zip(keys, values, strict=True):
            if key in predictions:
                raise ValueError(f"Duplicate frame in dataset: {key}")
            predictions[key] = float(value)
        processed_count += count
        if batch_index == 0:
            warmup_seconds = time.monotonic() - warmup_started
            steady_started = time.monotonic()
            print(
                f"Value inference warmup: {processed_count}/{len(indices)} "
                f"(workers + decoding + XLA compile: {warmup_seconds:.1f}s)",
                flush=True,
            )
            continue

        steady_frames += count
        if batch_index % 100 == 0 or processed_count == len(indices):
            elapsed = max(time.monotonic() - steady_started, 1e-6)
            rate = steady_frames / elapsed
            eta_seconds = (len(indices) - processed_count) / max(rate, 1e-6)
            print(
                f"Value inference: {processed_count}/{len(indices)} "
                f"({rate:.1f} frames/s, ETA {eta_seconds / 3600:.2f}h)",
                flush=True,
            )
    return predictions


def evaluate_values(paths, predictions):
    targets = {}
    for path in paths:
        ep, _, frames = episode_columns(path)
        values = np.asarray(pq.read_table(path, columns=["value_target"])["value_target"]).reshape(-1)
        targets.update({(ep, int(frame)): float(value) for frame, value in zip(frames, values, strict=True)})
    if not predictions or not predictions.keys() <= targets.keys():
        raise ValueError("Evaluation predictions must match frames in the held-out dataset")
    errors = np.asarray([value - targets[key] for key, value in predictions.items()])
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite evaluation targets or predictions")
    return {"frames": len(errors), "mse": float(np.mean(errors**2)), "mae": float(np.mean(np.abs(errors)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["targets", "advantages", "evaluate"])
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers used to decode and prefetch video")
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--positive-ratio", type=float, default=0.3)
    parser.add_argument("--failure-penalty", type=float, default=300.0)
    parser.add_argument("--return-scale", type=float, default=820.0)
    parser.add_argument("--max-frames", type=int, default=10000, help="Fixed validation subset; evaluate only")
    parser.add_argument("--output", type=pathlib.Path, help="Value evaluation JSON; evaluate only")
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    if (root / "INCOMPLETE").exists():
        parser.error("Dataset preparation is incomplete")
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths or not (root / "meta/info.json").is_file():
        parser.error("Expected a LeRobot v2 dataset with data/**/*.parquet and meta/info.json")
    if args.batch_size < 1 or args.num_workers < 0 or args.n_step < 1 or not 0 < args.positive_ratio < 1:
        parser.error("batch-size and n-step must be positive; num-workers must be nonnegative; positive-ratio must be in (0, 1)")
    episodes = [episode_columns(path) for path in paths]
    if len({ep for ep, _, _ in episodes}) != len(episodes):
        parser.error("An episode must occupy exactly one parquet file (LeRobot v2 layout)")
    source_path = root / "recap_source.json"
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    if args.stage == "advantages" and source.get("split") == "eval":
        parser.error("The eval split must not contribute to advantage thresholds or ACP training")
    if args.stage == "evaluate":
        if source.get("split") != "eval" or args.checkpoint is None or args.output is None or args.max_frames < 1:
            parser.error("evaluate requires a prepared eval split, --checkpoint, --output, and positive --max-frames")
        predictions = predict_values(
            root,
            args.checkpoint.expanduser().resolve(),
            args.batch_size,
            args.max_frames,
            args.num_workers,
        )
        result = evaluate_values(paths, predictions)
        result.update(checkpoint=str(args.checkpoint.resolve()), dataset_root=str(root))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    pending = []
    if args.stage == "targets":
        metadata = {
            row["episode_index"]: row
            for line in (root / "meta/episodes.jsonl").read_text().splitlines()
            if line.strip()
            for row in [json.loads(line)]
        }
        for ep, _, frames in episodes:
            meta = metadata[ep]
            if meta["length"] != len(frames):
                raise ValueError(f"Episode {ep}: metadata length disagrees with parquet")
            pending.append(
                {
                    "value_target": value_targets(
                        len(frames),
                        meta.get("success"),
                        args.failure_penalty,
                        args.return_scale,
                    )
                }
            )
    else:
        if args.checkpoint is None:
            parser.error("advantages requires --checkpoint pointing to a value checkpoint step directory")
        # Validate every label before loading the large model or writing any file.
        targets = [np.asarray(pq.read_table(p, columns=["value_target"])["value_target"]).reshape(-1) for p in paths]
        for target, (_, _, frames) in zip(targets, episodes, strict=True):
            if target.shape != frames.shape or not np.isfinite(target).all() or np.any((target < -1) | (target > 0)):
                raise ValueError("Run targets first, or supply finite value_target scalars in [-1, 0]")
        predictions = predict_values(
            root,
            args.checkpoint.expanduser().resolve(),
            args.batch_size,
            num_workers=args.num_workers,
        )
        if len(predictions) != sum(len(frames) for _, _, frames in episodes):
            raise ValueError("LeRobot loader and parquet files have different frame counts")
        task_indices = []
        for (ep, task, frames), target in zip(episodes, targets, strict=True):
            values = np.asarray([predictions[(ep, int(frame))] for frame in frames], dtype=np.float32)
            pending.append(
                {
                    "predicted_value": values,
                    "advantage": n_step_advantages(target, values, args.n_step).astype(np.float32),
                }
            )
            task_indices.extend([task] * len(frames))
        labels, thresholds = positive_labels(
            np.concatenate([p["advantage"] for p in pending]),
            np.asarray(task_indices),
            args.positive_ratio,
        )
        start = 0
        for item in pending:
            end = start + len(item["advantage"])
            item["is_positive"] = labels[start:end]
            start = end
        print(f"Per-task thresholds: {thresholds}; actual positive fraction: {labels.mean():.3f}")
    # This command intentionally updates the dataset copy supplied by the user.
    for path, columns in zip(paths, pending, strict=True):
        write_columns(path, columns)
    update_features(root, pending[0])
    print(f"Annotated {len(paths)} episodes in {root}")


if __name__ == "__main__":
    main()
