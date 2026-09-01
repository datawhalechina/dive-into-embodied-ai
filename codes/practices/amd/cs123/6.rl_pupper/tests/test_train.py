from pathlib import Path

import pytest

from train import train


def test_train_missing_checkpoint_raises(tmp_path):
    missing = tmp_path / "does_not_exist.zip"
    with pytest.raises(FileNotFoundError):
        train(
            timesteps=1,
            n_envs=1,
            seed=0,
            out=str(tmp_path / "outputs"),
            tensorboard=False,
            checkpoint=str(missing),
        )


def test_short_training_produces_checkpoint(tmp_path):
    final_path = train(
        timesteps=32,
        n_envs=1,
        seed=0,
        out=str(tmp_path / "outputs"),
        tensorboard=False,
        checkpoint=None,
    )
    assert final_path == Path(tmp_path / "outputs" / "pupper_ppo.zip")
    assert final_path.exists()
