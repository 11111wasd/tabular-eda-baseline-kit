from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration shared by EDA, splitting, modeling, and reporting."""

    target: str
    output_dir: str | Path = "outputs/baseline"
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    max_categories_for_one_hot: int = 30
    positive_label: str | int | float | None = None

    def resolved_output_dir(self) -> Path:
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
