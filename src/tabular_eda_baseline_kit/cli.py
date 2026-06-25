from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tabular_eda_baseline_kit.analysis import run_baseline_analysis
from tabular_eda_baseline_kit.config import AnalysisConfig
from tabular_eda_baseline_kit.data import generate_demo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA and interpretable baseline modeling for tabular data.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv", type=Path, help="Path to an input CSV file.")
    input_group.add_argument("--demo", action="store_true", help="Run on a synthetic demo dataset.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"), help="Directory for reports and plots.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used for data generation and splitting.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for the test split.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Fraction of rows reserved for validation.")
    return parser.parse_args()


def load_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.demo:
        return generate_demo_dataset(random_state=args.random_state)
    return pd.read_csv(args.csv)


def main() -> None:
    args = parse_args()
    df = load_dataframe(args)
    config = AnalysisConfig(
        target=args.target,
        output_dir=args.output_dir,
        random_state=args.random_state,
        test_size=args.test_size,
        validation_size=args.validation_size,
    )
    result = run_baseline_analysis(df, config)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
