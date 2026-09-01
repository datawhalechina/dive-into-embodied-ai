"""AMD-local `train` entry point forwarding directly to mjlab."""

from __future__ import annotations

import sys


def main() -> int | None:
    from mjlab.scripts.train import main as mjlab_train_main

    return mjlab_train_main()


if __name__ == "__main__":
    sys.exit(main())
