from tabular_eda_baseline_kit import AnalysisConfig, generate_demo_dataset, run_baseline_analysis


def main() -> None:
    df = generate_demo_dataset(random_state=42)
    config = AnalysisConfig(target="target", output_dir="outputs/example", random_state=42)
    result = run_baseline_analysis(df, config)
    print(f"Problem type: {result.problem_type}")
    print(f"Selected model: {result.selected_model}")


if __name__ == "__main__":
    main()
