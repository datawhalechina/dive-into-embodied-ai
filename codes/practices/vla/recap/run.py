"""Copy this directory to openpi/examples/recap, then run from the OpenPI root."""

import importlib
import logging
import pathlib
import subprocess
import sys

OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
COMMANDS = {
    "stats": "scripts.compute_norm_stats",
    "train": "scripts.train",
    "serve": "scripts.serve_policy",
    "annotate": "examples.recap.annotate",
    "data": "examples.recap.data",
    "rollout": "examples.recap.rollout",
    "serve-value": "examples.recap.serve_value",
}


def latest_checkpoint(config_name, exp_name, base=pathlib.Path("checkpoints")):
    directory = base / config_name / exp_name
    steps = [p for p in directory.glob("*") if p.name.isdigit() and (p / "params").is_dir()]
    if not steps:
        raise ValueError(f"No saved checkpoint with params found in {directory}")
    return max(steps, key=lambda p: int(p.name)).resolve()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in (*COMMANDS, "checkpoint"):
        print("Usage: python examples/recap/run.py {data|stats|train|serve|serve-value|rollout|annotate|checkpoint} [arguments]")
        raise SystemExit(0 if "--help" in sys.argv else 2)
    root = pathlib.Path(__file__).resolve().parents[2]
    if pathlib.Path.cwd().resolve() != root or not (root / "src/openpi").is_dir():
        raise SystemExit("Copy recap/ to openpi/examples/recap/ and run from the OpenPI root.")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != OPENPI_REVISION:
        print(
            f"Warning: this example was tested with OpenPI {OPENPI_REVISION}; found {revision}. "
            "Upstream API changes may require adaptation.",
            file=sys.stderr,
        )
    sys.path.insert(0, str(root))
    command = sys.argv.pop(1)
    if command == "checkpoint":
        if len(sys.argv) != 3:
            raise SystemExit("Usage: run.py checkpoint CONFIG_NAME EXP_NAME")
        print(latest_checkpoint(sys.argv[1], sys.argv[2]))
        return
    if command in ("stats", "train", "serve"):
        from examples.recap.config import register

        register()
    # Import by module name so spawned data workers can pickle upstream transforms.
    # Running compute_norm_stats as __main__ breaks its RemoveStrings transform.
    module = importlib.import_module(COMMANDS[command])
    if command == "train":
        from openpi.training.config import cli

        module.main(cli())
    elif command in ("stats", "serve"):
        import tyro

        if command == "stats":
            tyro.cli(module.main)
        else:
            logging.basicConfig(level=logging.INFO, force=True)
            module.main(tyro.cli(module.Args))
    else:
        module.main()


if __name__ == "__main__":
    main()
