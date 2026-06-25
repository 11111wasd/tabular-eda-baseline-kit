from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


@dataclass(frozen=True)
class DataQualityReport:
    row_count: int
    column_count: int
    duplicate_rows: int
    missing_values: dict[str, int]
    missing_rates: dict[str, float]
    numeric_summary: dict[str, dict[str, float]]
    categorical_cardinality: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_demo_dataset(
    n_samples: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a small binary-classification dataset with mixed feature types."""

    features, target = make_classification(
        n_samples=n_samples,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        random_state=random_state,
        class_sep=1.1,
    )
    columns = [f"feature_{idx}" for idx in range(features.shape[1])]
    df = pd.DataFrame(features, columns=columns)

    rng = np.random.default_rng(random_state)
    df["segment"] = rng.choice(["small", "mid", "enterprise"], size=n_samples, p=[0.5, 0.35, 0.15])
    df["region"] = rng.choice(["north", "south", "east", "west"], size=n_samples)
    df["target"] = target

    missing_idx = rng.choice(df.index, size=max(1, n_samples // 25), replace=False)
    df.loc[missing_idx, "feature_2"] = np.nan
    return df


def inspect_dataframe(df: pd.DataFrame) -> DataQualityReport:
    """Summarize data completeness, duplicates, and simple feature diagnostics."""

    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(exclude=[np.number])

    numeric_summary: dict[str, dict[str, float]] = {}
    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        if series.empty:
            numeric_summary[column] = {}
            continue
        numeric_summary[column] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
        }

    missing_values = df.isna().sum().astype(int).to_dict()
    row_count = int(len(df))
    missing_rates = {
        column: (count / row_count if row_count else 0.0)
        for column, count in missing_values.items()
    }

    return DataQualityReport(
        row_count=row_count,
        column_count=int(df.shape[1]),
        duplicate_rows=int(df.duplicated().sum()),
        missing_values=missing_values,
        missing_rates=missing_rates,
        numeric_summary=numeric_summary,
        categorical_cardinality=categorical_df.nunique(dropna=True).astype(int).to_dict(),
    )
