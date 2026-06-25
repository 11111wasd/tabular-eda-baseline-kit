from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tabular_eda_baseline_kit.config import AnalysisConfig
from tabular_eda_baseline_kit.data import DataQualityReport, inspect_dataframe
from tabular_eda_baseline_kit.modeling import (
    ProblemType,
    attach_preprocessor,
    build_baseline_models,
    build_preprocessor,
    evaluate_classification,
    evaluate_regression,
    infer_problem_type,
    split_dataset,
)
from tabular_eda_baseline_kit.plots import (
    save_binary_curve,
    save_confusion_matrix,
    save_correlation_heatmap,
    save_distribution_plots,
)


@dataclass(frozen=True)
class AnalysisResult:
    problem_type: ProblemType
    data_quality: DataQualityReport
    metrics: dict[str, dict[str, Any]]
    selected_model: str
    artifacts: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_type": self.problem_type,
            "data_quality": self.data_quality.to_dict(),
            "metrics": self.metrics,
            "selected_model": self.selected_model,
            "artifacts": self.artifacts,
        }


def _select_model(problem_type: ProblemType, metrics: dict[str, dict[str, Any]]) -> str:
    if problem_type == "classification":
        return max(metrics, key=lambda name: float(metrics[name].get("balanced_accuracy", 0.0)))
    return min(metrics, key=lambda name: float(metrics[name].get("rmse", float("inf"))))


def run_baseline_analysis(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    output_dir = config.resolved_output_dir()
    data_quality = inspect_dataframe(df)
    problem_type = infer_problem_type(df[config.target])
    split = split_dataset(
        df=df,
        target=config.target,
        test_size=config.test_size,
        validation_size=config.validation_size,
        random_state=config.random_state,
    )

    preprocessor = build_preprocessor(split.x_train)
    models = attach_preprocessor(build_baseline_models(problem_type, config.random_state), preprocessor)
    metrics: dict[str, dict[str, Any]] = {}
    artifacts = []

    artifacts.extend(save_distribution_plots(df, output_dir, config.target))
    correlation_path = save_correlation_heatmap(df, output_dir)
    if correlation_path:
        artifacts.append(correlation_path)

    for name, model in models.items():
        model.fit(split.x_train, split.y_train)
        if problem_type == "classification":
            model_metrics = evaluate_classification(model, split.x_validation, split.y_validation)
            y_pred = model.predict(split.x_validation)
            artifacts.append(save_confusion_matrix(split.y_validation, y_pred, output_dir, name))
            if "roc_curve_fpr" in model_metrics:
                artifacts.append(
                    save_binary_curve(
                        model_metrics["roc_curve_fpr"],
                        model_metrics["roc_curve_tpr"],
                        output_dir,
                        f"roc_curve_{name}.png",
                        f"ROC Curve: {name}",
                        "False Positive Rate",
                        "True Positive Rate",
                    )
                )
                artifacts.append(
                    save_binary_curve(
                        model_metrics["pr_curve_recall"],
                        model_metrics["pr_curve_precision"],
                        output_dir,
                        f"precision_recall_curve_{name}.png",
                        f"Precision-Recall Curve: {name}",
                        "Recall",
                        "Precision",
                    )
                )
        else:
            model_metrics = evaluate_regression(model, split.x_validation, split.y_validation)
        metrics[name] = model_metrics

    selected_model = _select_model(problem_type, metrics)
    result = AnalysisResult(
        problem_type=problem_type,
        data_quality=data_quality,
        metrics=metrics,
        selected_model=selected_model,
        artifacts=artifacts,
    )

    summary_path = output_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    artifacts.append(str(summary_path))
    return result
