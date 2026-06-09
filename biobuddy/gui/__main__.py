from __future__ import annotations

import argparse

from .model_editor import launch_model_editor
from .yeadon_measurement_editor import launch_yeadon_measurement_editor


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch a BioBuddy desktop GUI.")
    parser.add_argument(
        "--yeadon-measurements",
        action="store_true",
        help="launch the Yeadon anthropometric measurement editor instead of the model editor",
    )
    args = parser.parse_args(argv)

    if args.yeadon_measurements:
        launch_yeadon_measurement_editor()
    else:
        launch_model_editor()


if __name__ == "__main__":
    main()
