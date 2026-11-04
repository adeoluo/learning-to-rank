"""Entrypoint: ``python -m learning_to_rank``."""

from __future__ import annotations

import sys

from learning_to_rank.cli import main

if __name__ == "__main__":
    sys.exit(main())
