from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_charts import discover_benchmark_runs
from benchmark_charts import save_benchmark_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build charts from benchmark detail logs."
    )
    parser.add_argument(
        "--details",
        type=Path,
        help="Path to one benchmark *_details.json file. Defaults to the newest run.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("benchmark_logs"),
        help="Directory with benchmark logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details_path = args.details
    if details_path is None:
        runs = discover_benchmark_runs(args.log_dir)
        if not runs:
            raise RuntimeError(f"No benchmark detail logs found in: {args.log_dir}")
        details_path = runs[0]

    output_path = save_benchmark_figure(details_path=details_path, output_path=args.output)
    print(f"Saved chart image to: {output_path}")


if __name__ == "__main__":
    main()
