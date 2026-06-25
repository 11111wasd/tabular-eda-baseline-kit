"""Reproducible EDA and baseline modeling for tabular data."""

from tabular_eda_baseline_kit.analysis import AnalysisResult, run_baseline_analysis
from tabular_eda_baseline_kit.config import AnalysisConfig
from tabular_eda_baseline_kit.data import DataQualityReport, generate_demo_dataset, inspect_dataframe

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "DataQualityReport",
    "generate_demo_dataset",
    "inspect_dataframe",
    "run_baseline_analysis",
]
