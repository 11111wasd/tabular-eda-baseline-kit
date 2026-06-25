# Tabular EDA Baseline Kit

A small, reproducible Python toolkit for tabular data analysis. It helps you run a practical first pass before modeling:

- data quality checks for missing values, duplicates, and unusual numeric ranges
- distribution and correlation plots
- train/validation/test splitting with a fixed random seed
- interpretable baseline models for classification and regression
- model evaluation summaries and common diagnostic plots

This project is intentionally lightweight. It is useful for notebooks, teaching, quick project audits, and as a clean starting point for more advanced feature engineering.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m tabular_eda_baseline_kit.cli --demo --target target --output-dir outputs/demo
```

The demo command creates a synthetic binary-classification dataset, runs EDA, trains interpretable baselines, and writes reports plus plots under `outputs/demo`.

## Use Your Own CSV

```bash
python -m tabular_eda_baseline_kit.cli \
  --csv data/my_dataset.csv \
  --target target_column \
  --output-dir outputs/my_dataset
```

For classification, the toolkit reports accuracy, balanced accuracy, F1, ROC AUC when available, a confusion matrix, ROC curve, and precision-recall curve. For regression, it reports MAE, RMSE, and R2.

## Python API

```python
from tabular_eda_baseline_kit import AnalysisConfig, generate_demo_dataset, run_baseline_analysis

df = generate_demo_dataset(random_state=42)
config = AnalysisConfig(target="target", output_dir="outputs/demo", random_state=42)
result = run_baseline_analysis(df, config)

print(result.problem_type)
print(result.metrics)
```

## Reproducibility

- Default random seed: `42`
- Default split: 60 percent train, 20 percent validation, 20 percent test
- Models are from scikit-learn and use deterministic settings where available
- Output artifacts are written to the configured output directory

## Project Scope

The first goal is a reliable baseline, not maximum accuracy. Start with interpretable models, inspect the data, verify metric definitions, then iterate on feature engineering and stronger models only when the baseline is understood.

## License

MIT
