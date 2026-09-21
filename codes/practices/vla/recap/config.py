"""Three configs registered at runtime; no edits to openpi/training/config.py."""

import dataclasses
import os
import random
from typing import Literal

import flax.nnx as nnx
import numpy as np
from examples.recap.value_model import ValueConfig, ValueWeightLoader
from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import libero_policy
from openpi.shared import nnx_utils
from openpi.training import config, optimizer, weight_loaders

# Set these before EVERY command (including serving), or edit the defaults here.
SFT_REPO_ID = os.environ.get("RECAP_SFT_REPO_ID", "local/libero10_fewshot_sft")
REPO_ID = os.environ.get("RECAP_REPO_ID", "local/libero10_task0_train")
INIT_PARAMS = os.environ.get("RECAP_INIT_PARAMS", "gs://openpi-assets/checkpoints/pi05_base/params")
NORM_DIR = "./assets/pi05_recap_sft"


@dataclasses.dataclass(frozen=True)
class AdvantagePrompt:
    dropout: float = 0.3

    def __call__(self, data):
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be in [0, 1]")
        indicator = data.pop("is_positive", None)
        if indicator is not None:
            indicator = np.asarray(indicator)
            if indicator.size != 1 or indicator.item() not in (0, 1):
                raise ValueError("is_positive must be a scalar 0 or 1")
            if random.random() < self.dropout:
                return data
        # No training label at inference: always request positive advantage.
        positive = indicator is None or bool(indicator.item())
        prompt = str(np.asarray(data["prompt"]).item())
        data["prompt"] = f"{prompt}\nAdvantage: {'positive' if positive else 'negative'}"
        return data


@dataclasses.dataclass(frozen=True)
class RecapInputs:
    model_type: object

    def __call__(self, data):
        inputs = libero_policy.LiberoInputs(self.model_type)(data)
        # Upstream LiberoInputs constructs a new dict and would drop these labels.
        for key in ("value_target", "is_positive"):
            if key in data:
                inputs[key] = data[key]
        return inputs


@dataclasses.dataclass(frozen=True)
class ValueTarget:
    def __call__(self, data):
        target = np.asarray(data.pop("value_target"))
        if target.size != 1 or not np.isfinite(target).all() or not -1 <= target.item() <= 0:
            raise ValueError("value_target must be a finite scalar in [-1, 0]")
        # Runs AFTER action normalization/padding, so value labels keep their scale.
        data["actions"] = np.full_like(data["actions"], target.item(), dtype=np.float32)
        return data


@dataclasses.dataclass(frozen=True)
class RecapData(config.LeRobotLiberoDataConfig):
    stage: Literal["sft", "value", "acp"] = "sft"
    acp_dropout: float = 0.3

    def create(self, assets_dirs, model_config):
        from lerobot.common.constants import HF_LEROBOT_HOME

        if (HF_LEROBOT_HOME / self.repo_id / "INCOMPLETE").exists():
            raise ValueError("Dataset preparation is incomplete; finish it before computing stats or training")
        source = HF_LEROBOT_HOME / self.repo_id / "recap_source.json"
        if source.exists():
            import json

            if json.loads(source.read_text())["split"] == "eval":
                raise ValueError("The held-out eval split is only for value evaluation, never stats or training")
        base = super().create(assets_dirs, model_config)
        mapping = dict(base.repack_transforms.inputs[0].structure)
        if self.stage == "value":
            mapping["value_target"] = "value_target"
        elif self.stage == "acp":
            mapping["is_positive"] = "is_positive"
        model_inputs = list(base.model_transforms.inputs)
        if self.stage == "acp":
            model_inputs.insert(0, AdvantagePrompt(self.acp_dropout))
        elif self.stage == "value":
            model_inputs.append(ValueTarget())
        return dataclasses.replace(
            base,
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform(mapping)]),
            data_transforms=transforms.Group(
                inputs=[RecapInputs(model_config.model_type)],
                outputs=base.data_transforms.outputs,
            ),
            model_transforms=dataclasses.replace(base.model_transforms, inputs=model_inputs),
        )


def get_configs():
    configs = []
    for stage in ("sft", "value", "acp"):
        is_value = stage == "value"
        model = (ValueConfig if is_value else pi0_config.Pi0Config)(pi05=True, action_horizon=10)
        trainable_value = nnx.Any(
            nnx_utils.PathRegex("PaliGemma/img/.*"),
            nnx.All(
                nnx_utils.PathRegex(".*llm.*"),
                nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
            ),
            nnx_utils.PathRegex("(final_norm|value_fc1|value_fc2)/.*"),
        )
        configs.append(
            config.TrainConfig(
                name=f"pi05_recap_{stage}",
                model=model,
                data=RecapData(
                    repo_id=SFT_REPO_ID if stage == "sft" else REPO_ID,
                    stage=stage,
                    assets=config.AssetsConfig(assets_dir=NORM_DIR, asset_id=SFT_REPO_ID),
                    base_config=config.DataConfig(prompt_from_task=True),
                    extra_delta_transform=False,
                ),
                weight_loader=(ValueWeightLoader if is_value else weight_loaders.CheckpointWeightLoader)(INIT_PARAMS),
                freeze_filter=nnx.Not(trainable_value) if is_value else nnx.Nothing,
                batch_size=8,
                fsdp_devices=1,
                num_workers=2,
                num_train_steps=30_000 if is_value else 45_000,
                lr_schedule=optimizer.CosineDecaySchedule(
                    warmup_steps=1_000 if is_value else 1_500,
                    peak_lr=1e-5 if is_value else 5e-5,
                    decay_steps=30_000 if is_value else 45_000,
                    decay_lr=1e-6 if is_value else 5e-6,
                ),
                optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
                ema_decay=0.999,
                save_interval=1_000,
                keep_period=10_000,
            )
        )
    return configs


def register():
    for entry in get_configs():
        config._CONFIGS_DICT[entry.name] = entry
