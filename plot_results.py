import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _fail(message: str) -> None:
    raise SystemExit(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot x vs y from a Monte Carlo simulation result.csv."
    )
    parser.add_argument(
        "--sim-name",
        required=True,
        help="Simulation name (results/<sim-name>/result.csv).",
    )
    parser.add_argument(
        "--x",
        required=True,
        help="Column name for the x-axis.",
    )
    parser.add_argument(
        "--y",
        required=True,
        help="Column name for the y-axis.",
    )
    parser.add_argument(
        "--filename",
        default="",
        help="Optional output filename (default: plot_<x>_vs_<y>.png).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    sim_name = args.sim_name.strip()
    if not sim_name:
        _fail("Simulation name cannot be empty.")

    sim_dir = Path("results") / sim_name
    if not sim_dir.exists() or not sim_dir.is_dir():
        _fail(f"Simulation folder not found: {sim_dir}")

    csv_path = sim_dir / "result.csv"
    if not csv_path.exists():
        _fail(f"result.csv not found in: {sim_dir}")

    df = pd.read_csv(csv_path)
    if args.x not in df.columns:
        _fail(f"x-axis column not found in result.csv: {args.x}")
    if args.y not in df.columns:
        _fail(f"y-axis column not found in result.csv: {args.y}")

    x = df[args.x]
    y = df[args.y]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linestyle="-")
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.title(f"{args.y} vs {args.x}")
    plt.tight_layout()

    filename = args.filename.strip()
    if not filename:
        safe_x = args.x.replace(" ", "_")
        safe_y = args.y.replace(" ", "_")
        filename = f"plot_{safe_x}_vs_{safe_y}.png"

    out_path = sim_dir / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
