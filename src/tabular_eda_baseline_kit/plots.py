from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_distribution_plots(df: pd.DataFrame, output_dir: Path, target: str) -> list[str]:
    paths: list[str] = []
    numeric_columns = [column for column in df.select_dtypes(include=[np.number]).columns if column != target]
    for column in numeric_columns[:8]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[column], kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        path = output_dir / f"distribution_{column}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def save_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> str | None:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), cmap="vlag", center=0, ax=ax)
    ax.set_title("Numeric Feature Correlation")
    path = output_dir / "correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_dir: Path, model_name: str) -> str:
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    fig, ax = plt.subplots(figsize=(5, 5))
    display.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix: {model_name}")
    path = output_dir / f"confusion_matrix_{model_name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def save_binary_curve(
    x_values: list[float],
    y_values: list[float],
    output_dir: Path,
    filename: str,
    title: str,
    x_label: str,
    y_label: str,
) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_values, y_values)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)
