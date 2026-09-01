from pathlib import Path
from copy import deepcopy

import pytest
from stable_baselines3 import PPO

from train import DEFAULT_CONFIG_PATH, load_config, train


def test_load_default_config():
    config = load_config()
    assert Path(DEFAULT_CONFIG_PATH).exists()
    assert config["run"]["device"] == "cpu"
    assert config["ppo"]["net_arch"] == [256, 256]


def _test_config(tmp_path):
    config = deepcopy(load_config())
    config["run"].update({
        "total_timesteps": 32,
        "n_envs": 1,
        "output_dir": str(tmp_path / "outputs"),
        "tensorboard": False,
        "resume_from": None,
        "transfer_from": None,
    })
    config["ppo"].update({
        "n_steps": 32,
        "batch_size": 32,
        "n_epochs": 1,
        "net_arch": [32, 32],
        "verbose": 0,
    })
    return config


def test_train_missing_checkpoint_raises(tmp_path):
    missing = tmp_path / "does_not_exist.zip"
    config = _test_config(tmp_path)
    config["run"]["resume_from"] = str(missing)
    with pytest.raises(FileNotFoundError):
        train(config)


def test_train_missing_transfer_checkpoint_raises(tmp_path):
    missing = tmp_path / "does_not_exist.zip"
    config = _test_config(tmp_path)
    config["run"]["transfer_from"] = str(missing)
    with pytest.raises(FileNotFoundError):
        train(config)


def test_short_training_produces_checkpoint(tmp_path):
    final_path = train(_test_config(tmp_path))
    assert final_path == Path(tmp_path / "outputs" / "pupper_ppo.zip")
    assert final_path.exists()


def test_tensorboard_logging(tmp_path):
    config = _test_config(tmp_path)
    config["run"]["tensorboard"] = True
    train(config)
    assert any((tmp_path / "outputs" / "tb").rglob("events.out.tfevents.*"))


def test_transfer_45_dim_policy_to_gait_policy(tmp_path):
    source_config = _test_config(tmp_path / "source")
    source_path = train(source_config)

    target_config = _test_config(tmp_path / "target")
    target_config["gait"]["enabled"] = True
    target_config["run"]["transfer_from"] = str(source_path)
    target_path = train(target_config)

    model = PPO.load(target_path)
    assert model.observation_space.shape == (50,)
