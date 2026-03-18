# PIC16B Project

This project looks at RateMyProfessors review data for UC campuses and builds models to predict two review targets:

- `clarityRating`
- `difficultyRating`

The repository is organized around one pipeline: collect raw data, clean and featurize reviews, run model selection, and analyze the saved results.

## Pipeline Overview

1. `get_professors.py`
   Downloads professor metadata from RateMyProfessors and writes `all_professors_rmp.csv`.

2. `get_reviews.py`
   Uses the professor list to download review-level data and writes `rmp_all_schools_reviews_small.csv`.

3. `rmp_preprocess.py`
   Cleans raw review text, removes noisy rows, and expands `ratingTags` into one-hot tag columns.

4. `rating_data.py`, `rating_features.py`, `rating_models.py`, `rating_types.py`
   Define the reusable training pipeline:
   - dataset loading and grouped train/test split by `profId`
   - TF-IDF + NMF text features
   - structured tag features
   - model registry and shared run configuration

5. `clarity_runner.py` and `difficulty_runner.py`
   Call the shared logic in `rating_runner_core.py` to run grouped cross-validation, rank model configurations, and save result tables.

6. `data_analysis/`
   Contains the analysis notebooks and helper code used to inspect the saved runner outputs and rebuild the selected models for deeper evaluation.

## Main Files

- [get_professors.py]: fetch professor metadata
- [get_reviews.py]: fetch review data
- [rmp_preprocess.py]: clean raw review text and tags
- [rating_runner_core.py]: shared experiment engine
- [clarity_runner.py]: clarity model pipeline
- [difficulty_runner.py]: difficulty model pipeline
- [data_analysis/clarity_results_analysis.ipynb]: clarity analysis notebook
- [data_analysis/difficulty_results_analysis.ipynb]: difficulty analysis notebook

## Data and Output Folders

- `raw_data/`
  Stores the raw CSV snapshots currently used in the project, including:
  - `all_professors_rmp.csv`
  - `rmp_all_schools_reviews_small.csv`

- `runner_results/`
  Stores saved model-selection outputs, including:
  - `clarity_results.csv`
  - `clarity_results_cv_folds.csv`
  - `clarity_results_model_summary.csv`
  - `difficulty_results.csv`
  - `difficulty_results_cv_folds.csv`
  - `difficulty_results_model_summary.csv`

- `data_analysis/`
  Stores notebooks and helper functions for post-run evaluation.

## Typical Workflow

Run the project from the repository root:

```bash
python get_professors.py
python get_reviews.py
python clarity_runner.py
python difficulty_runner.py
```

Then open the notebooks in `data_analysis/` to inspect the saved outputs and compare model choices.

## Important Note About Paths

The core scripts currently default to root-level filenames such as `all_professors_rmp.csv`, `rmp_all_schools_reviews_small.csv`, and `clarity_results.csv`. The repository also keeps organized copies of those artifacts inside `raw_data/` and `runner_results/`. That means the conceptual pipeline is:

`raw professor list -> raw review table -> runner outputs -> analysis notebooks`

but the scripts themselves still read and write the default filenames defined in code.
