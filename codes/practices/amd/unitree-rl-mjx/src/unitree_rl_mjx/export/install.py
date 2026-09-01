"""Install an exported policy into an upstream deployment checkout.

Exports the policy to a temporary folder first - export verification gates
the copy, so a policy that fails the JAX-ONNX contract never reaches the
deploy tree - then places the versioned folder at the robot's policy
directory, touching nothing else in the checkout.

    python -m unitree_rl_mjx.export.install --robot go2 \\
        --params runs/r0/params.bin --dest ~/unitree_rl_mjlab --version v1
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from unitree_rl_mjx.export.policy_onnx import export_policy

_POLICY_DIRS = {
  "go2": Path("deploy/robots/go2/config/policy/velocity"),
}


def install(robot: str, params: str, dest: str, version: str) -> Path:
  policy_root = Path(dest).expanduser() / _POLICY_DIRS[robot]
  if not policy_root.parent.is_dir():
    raise FileNotFoundError(
      f"{policy_root.parent} does not exist - is {dest!r} a deployment checkout?"
    )
  target = policy_root / version
  if target.exists():
    raise FileExistsError(f"{target} already exists; pick another --version")

  with tempfile.TemporaryDirectory() as tmp:
    exported = export_policy(params, Path(tmp) / version)
    shutil.copytree(exported, target)
  print(f"installed {target}")
  return target


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--robot", choices=sorted(_POLICY_DIRS), required=True)
  parser.add_argument("--params", required=True, help="params.bin of a trained run")
  parser.add_argument("--dest", required=True, help="upstream checkout root")
  parser.add_argument("--version", default="v1", help="policy folder name")
  args = parser.parse_args()
  install(args.robot, args.params, args.dest, args.version)


if __name__ == "__main__":
  main()
