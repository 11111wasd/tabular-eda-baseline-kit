from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ProblemType = Literal["classification", "regression"]


@dataclass(frozen=True)
class SplitData:
    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def infer_problem_type(y: pd.Series) -> ProblemType:
    if y.dtype.kind in {"O", "b", "U", "S"}:
        return "classification"
    unique_count = y.nunique(dropna=True)
    if unique_count <= max(20, int(len(y) * 0.05)):
        return "classification"
    return "regression"


def split_dataset(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> SplitData:
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")

    x = df.drop(columns=[target])
    y = df[target]
    problem_type = infer_problem_type(y)
    stratify = y if problem_type == "classification" and y.nunique(dropna=True) > 1 else None

    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    relative_validation_size = validation_size / (1 - test_size)
    stratify_train_validation = (
        y_train_validation
        if problem_type == "classification" and y_train_validation.nunique(dropna=True) > 1
        else None
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=stratify_train_validation,
    )

    return SplitData(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = x.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_baseline_models(problem_type: ProblemType, random_state: int) -> dict[str, Pipeline]:
    if problem_type == "classification":
        estimators = {
            "dummy_majority": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=random_state),
        }
    else:
        estimators = {
            "dummy_mean": DummyRegressor(strategy="mean"),
            "linear_regression": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(max_depth=4, random_state=random_state),
        }
    return {name: Pipeline(steps=[("model", estimator)]) for name, estimator in estimators.items()}


def attach_preprocessor(models: dict[str, Pipeline], preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        name: Pipeline(steps=[("preprocessor", clone(preprocessor)), ("estimator", clone(pipeline.named_steps["model"]))])
        for name, pipeline in models.items()
    }


def evaluate_classification(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float | list[float]]:
    predictions = model.predict(x)
    metrics: dict[str, float | list[float]] = {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "f1_weighted": float(f1_score(y, predictions, average="weighted", zero_division=0)),
    }
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average="weighted", zero_division=0)
    metrics.update(
        {
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_from_prfs_weighted": float(f1),
        }
    )
    if hasattr(model, "predict_proba") and y.nunique(dropna=True) == 2:
        positive_scores = model.predict_proba(x)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y, positive_scores))
        fpr, tpr, _ = roc_curve(y, positive_scores)
        pr_precision, pr_recall, _ = precision_recall_curve(y, positive_scores)
        metrics["roc_curve_fpr"] = fpr.tolist()
        metrics["roc_curve_tpr"] = tpr.tolist()
        metrics["pr_curve_precision"] = pr_precision.tolist()
        metrics["pr_curve_recall"] = pr_recall.tolist()
    return metrics


def evaluate_regression(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    predictions = model.predict(x)
    return {
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(mean_squared_error(y, predictions, squared=False)),
        "r2": float(r2_score(y, predictions)),
    }
