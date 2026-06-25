from __future__ import annotations

import pandas as pd

from tabular_eda_baseline_kit import AnalysisConfig, generate_demo_dataset, inspect_dataframe, run_baseline_analysis


def test_inspect_dataframe_reports_missing_and_duplicates() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, None, 2.0],
            "b": ["x", "y", "z", "y"],
            "target": [0, 1, 0, 1],
        }
    )

    report = inspect_dataframe(df)

    assert report.row_count == 4
    assert report.column_count == 3
    assert report.duplicate_rows == 1
    assert report.missing_values["a"] == 1
    assert report.categorical_cardinality["b"] == 3


def test_run_baseline_analysis_demo(tmp_path) -> None:
    df = generate_demo_dataset(n_samples=160, random_state=123)
    config = AnalysisConfig(target="target", output_dir=tmp_path, random_state=123)

    result = run_baseline_analysis(df, config)

    assert result.problem_type == "classification"
    assert "dummy_majority" in result.metrics
    assert "logistic_regression" in result.metrics
    assert result.selected_model in result.metrics
    assert (tmp_path / "analysis_summary.json").exists()
