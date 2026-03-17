"""Shared helpers for rating analysis notebooks."""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error

from rating_data import ReviewDatasetManager
from rating_features import FeaturePipeline
from rating_models import ModelRegistry
from rating_runner_core import ModelEvaluator

METRIC_COLS = ["accuracy", "macro_f1", "mae", "mean_accuracy", "mean_macro_f1", "mean_mae", "std_accuracy", "std_macro_f1", "std_mae", "fold", "k"]
LEADER_SORTS = [
    ("Best by Accuracy", ["mean_accuracy", "mean_macro_f1", "mean_mae"], [False, False, True]),
    ("Best by Macro-F1", ["mean_macro_f1", "mean_accuracy", "mean_mae"], [False, False, True]),
    ("Best by MAE", ["mean_mae", "mean_accuracy", "mean_macro_f1"], [True, False, False]),
]


def load_results_tables(results_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the result table and isolate CV summary rows."""
    results = pd.read_csv(results_csv)
    for col in METRIC_COLS:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")

    cv_summary = results.copy()
    if "result_type" in cv_summary.columns:
        cv_summary = cv_summary[cv_summary["result_type"] == "cv_summary"].copy()
    if cv_summary.empty:
        raise ValueError(f"No cv_summary rows found in {results_csv}")
    return results, cv_summary.reset_index(drop=True)


def select_metric_leaders(cv_summary: pd.DataFrame) -> pd.DataFrame:
    """Pick the top CV row under each ranking metric."""
    leaders = []
    for label, sort_cols, ascending in LEADER_SORTS:
        row = cv_summary.sort_values(sort_cols, ascending=ascending).reset_index(drop=True).iloc[0].copy()
        row["selection_label"] = label
        leaders.append(row)
    return pd.DataFrame(leaders).reset_index(drop=True)


def get_optional_test_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Return optional holdout rows if the runner wrote them."""
    if "result_type" not in results.columns:
        return pd.DataFrame()
    return results[results["result_type"].isin(["test", "test_row"])].copy().reset_index(drop=True)


def build_analysis_state(target_col: str, input_csv: str = "rmp_all_schools_reviews_small.csv", test_size: float = 0.2, random_state: int = 42, max_rows: int | None = None) -> Dict[str, Any]:
    """Recreate the dataset split and reusable feature objects from the runner."""
    dataset_manager = ReviewDatasetManager(target_col=target_col)
    feature_pipeline = FeaturePipeline(random_state=random_state)
    spec_map = {spec.name: spec for spec in ModelRegistry(random_state=random_state).get_specs()}

    df = dataset_manager.load_and_validate(input_csv, max_rows)
    tag_cols = dataset_manager.get_tag_columns(df)
    df_train, df_test = dataset_manager.split_train_test(df, test_size=test_size, random_state=random_state)

    train_comments = df_train["comment_clean"].astype(str).tolist()
    test_comments = df_test["comment_clean"].astype(str).tolist()
    y_train = df_train[target_col].to_numpy()
    y_test = df_test[target_col].to_numpy()
    train_tags = feature_pipeline.tag_matrix(df_train, tag_cols)
    test_tags = feature_pipeline.tag_matrix(df_test, tag_cols)

    return {
        "target_col": target_col,
        "feature_pipeline": feature_pipeline,
        "spec_map": spec_map,
        "df_train": df_train,
        "df_test": df_test,
        "tag_cols": tag_cols,
        "train_comments": train_comments,
        "test_comments": test_comments,
        "y_train": y_train,
        "y_test": y_test,
        "train_tags": train_tags,
        "test_tags": test_tags,
    }


def evaluate_leader(row: pd.Series, state: Dict[str, Any]) -> Dict[str, Any]:
    """Fit one selected config on the shared split and return diagnostics."""
    model_name = str(row["model"])
    k = int(row["k"])
    spec = state["spec_map"][model_name]
    pipeline = state["feature_pipeline"]

    vectorizer, nmf, x_train = pipeline.fit_transform(state["train_comments"], state["train_tags"], k=k)
    x_test = pipeline.transform(state["test_comments"], state["test_tags"], vectorizer, nmf)
    y_pred = ModelEvaluator.fit_predict_model(spec, x_train, state["y_train"], x_test, min_label=1, max_label=5)

    labels = [1, 2, 3, 4, 5]
    cm = confusion_matrix(state["y_test"], y_pred, labels=labels)
    row_totals = cm.sum(axis=1)
    diag = np.diag(cm)
    per_rating_accuracy = np.divide(diag, row_totals, out=np.zeros_like(diag, dtype=float), where=row_totals != 0)
    per_rating_df = pd.DataFrame({"true_rating": labels, "support": row_totals, "correct_predictions": diag, "accuracy_when_true": per_rating_accuracy})
    cm_norm = cm / np.clip(row_totals[:, None], 1, None)

    return {
        "model_name": model_name,
        "k": k,
        "vectorizer": vectorizer,
        "nmf": nmf,
        "y_pred": y_pred,
        "test_accuracy": accuracy_score(state["y_test"], y_pred),
        "test_macro_f1": f1_score(state["y_test"], y_pred, average="macro"),
        "test_mae": mean_absolute_error(state["y_test"], y_pred),
        "per_rating_df": per_rating_df,
        "cm": cm,
        "cm_norm": cm_norm,
        "classification_report": classification_report(state["y_test"], y_pred, labels=labels, digits=4),
    }


def summarize_topics(result: Dict[str, Any], state: Dict[str, Any]) -> pd.DataFrame:
    """Describe fitted NMF topics for one selected configuration."""
    x_train_tfidf = result["vectorizer"].transform(state["train_comments"])
    W_train = result["nmf"].transform(x_train_tfidf)
    H = result["nmf"].components_
    feature_names = np.array(result["vectorizer"].get_feature_names_out())
    dominant_topic = W_train.argmax(axis=1)
    rows = []

    for topic_id in range(result["nmf"].n_components):
        mask = dominant_topic == topic_id
        support = int(mask.sum())
        if support == 0:
            continue
        top_words = feature_names[np.argsort(H[topic_id])[::-1][:8]].tolist()
        rows.append({"topic": topic_id, "support": support, "mean_rating": float(state["y_train"][mask].mean()), "top_words": ", ".join(top_words)})

    return pd.DataFrame(rows).sort_values(["mean_rating", "support"], ascending=[False, False]).reset_index(drop=True)


def _plot_confusion_matrices(result: Dict[str, Any]) -> None:
    """Render raw and row-normalized confusion matrices."""
    labels = [1, 2, 3, 4, 5]
    title_suffix = f"{result['model_name']} (k={result['k']})"

    plt.figure(figsize=(7, 6))
    sns.heatmap(result["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix: {title_suffix}")
    plt.xlabel("Predicted rating")
    plt.ylabel("True rating")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 6))
    sns.heatmap(result["cm_norm"], annot=True, fmt=".2f", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.title(f"Row-normalized Confusion Matrix: {title_suffix}")
    plt.xlabel("Predicted rating")
    plt.ylabel("True rating")
    plt.tight_layout()
    plt.show()


def run_metric_leader_analysis(metric_leaders: pd.DataFrame, state: Dict[str, Any], rating_label: str | None = None) -> Dict[str, Dict[str, Any]]:
    """Run the full evaluation/reporting flow for each selected metric leader."""
    all_results = {}

    for _, row in metric_leaders.iterrows():
        selection_label = str(row["selection_label"])
        display(Markdown(f"## {selection_label}"))
        print("Selected CV summary row:")
        display(row[["model", "k", "mean_accuracy", "mean_macro_f1", "mean_mae"]].to_frame("value"))

        result = evaluate_leader(row, state)
        topic_summary = summarize_topics(result, state)

        print(f"Rebuilt model: {result['model_name']} (k={result['k']})")
        print(f"Holdout accuracy: {result['test_accuracy']:.4f}")
        print(f"Holdout macro F1: {result['test_macro_f1']:.4f}")
        print(f"Holdout MAE: {result['test_mae']:.4f}")
        print("Per-rating accuracy (conditioned on true class):")
        display(result["per_rating_df"])
        print("Classification report:")
        print(result["classification_report"])
        _plot_confusion_matrices(result)
        print("Topic-level descriptive patterns:")
        display(topic_summary[["topic", "support", "mean_rating", "top_words"]])

        all_results[selection_label] = {"leader_row": row.copy(), "evaluation": result, "topic_summary": topic_summary.copy()}

    return all_results
