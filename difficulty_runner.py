"""Experiment runner for difficulty rating model selection and reporting."""

from rating_runner_core import ExperimentRunner, make_default_config


def default_config():
    """Return the default runtime configuration for difficulty experiments."""
    return make_default_config(target_col="difficultyRating", output_csv="difficulty_results.csv", output_cv_folds_csv="difficulty_results_cv_folds.csv", output_model_summary_csv="difficulty_results_model_summary.csv")


def main() -> None:
    """Run the experiment with default configuration when called as a script."""
    runner = ExperimentRunner(default_config())
    runner.run()


if __name__ == "__main__":
    main()
