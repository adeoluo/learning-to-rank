"""Entrypoint: ``python -m learning_to_rank``."""

from __future__ import annotations

from learning_to_rank import __version__


def main() -> None:
    print(f"learning-to-rank {__version__}")


if __name__ == "__main__":
    main()
