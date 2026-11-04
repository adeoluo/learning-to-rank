"""Entrypoint: `python -m learning_to_rank`."""

from __future__ import annotations

from learning_to_rank import __version__
from learning_to_rank.configuration import settings


def main() -> None:
    header = f"Learning To Rank v{__version__}"
    status = f"environment={settings.environment}, seed={settings.seed}"
    print(f"{header} - {status}")


if __name__ == "__main__":
    main()
