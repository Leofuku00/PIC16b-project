"""Generic experiment runner for rating model selection and reporting."""

from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import GroupKFold

from rating_data import ReviewDatasetManager
from rating_features import FeaturePipeline
from rating_models import ModelRegistry
from rating_types import FoldEval, ModelSpec, RunConfig


def make_default_config(
    target_col: str,
    output_csv: str,
    output_cv_folds_csv: str,
    output_model_summary_csv: str,
) -> RunConfig:
    """Return the default runtime configuration for local experiments."""
    return RunConfig(
        input_csv="rmp_all_schools_reviews_small.csv",
        output_csv=output_csv,
        output_cv_folds_csv=output_cv_folds_csv,
        output_model_summary_csv=output_model_summary_csv,
        target_col=target_col,
        k_values=[3, 5, 8, 10],
        lsvc_safe_k_values=[3, 5],
        test_size=0.2,
        cv_splits=5,
        lsvc_safe_cv_splits=2,
        random_state=42,
        max_rows=None,
        requested_models=None,
        include_test_row=False,
    )


class ModelEvaluator:
    """Fit models and normalize prediction behavior across model families."""

    @staticmethod
    def fit_predict_model(
        spec: ModelSpec,
        x_train: csr_matrix,
        y_train: np.ndarray,
        x_valid: csr_matrix,
        min_label: int = 1,
        max_label: int = 5,
    ) -> np.ndarray:
        """Fit one model and return predictions on the validation/test matrix."""
        model = spec.build()
        if spec.family in {"classifier_dense", "regressor_dense"}:
            x_train_fit = x_train.toarray()
            x_valid_pred = x_valid.toarray()
        else:
            x_train_fit = x_train
            x_valid_pred = x_valid

        if spec.name == "xgboost_cls":
            y_train_fit = y_train.astype(int) - min_label
        else:
            y_train_fit = y_train

        model.fit(x_train_fit, y_train_fit)

        if spec.family in {"regressor", "regressor_dense"}:
            preds = model.predict(x_valid_pred)
            preds = np.rint(preds).astype(int)
            preds = np.clip(preds, min_label, max_label)
            return preds

        preds = model.predict(x_valid_pred).astype(int)
        if spec.name == "xgboost_cls":
            preds = preds + min_label
        return preds.astype(int)


class CrossValidationRunner:
    """Run grouped cross-validation searches and summarize candidate quality."""

    def __init__(self, feature_pipeline: FeaturePipeline, target_col: str):
        """Bind the reusable feature pipeline and supervised target."""
        self.feature_pipeline = feature_pipeline
        self.target_col = target_col

    def grouped_cv_search(
        self,
        df_train: pd.DataFrame,
        tag_cols: Sequence[str],
        k_values: Sequence[int],
        specs: Sequence[ModelSpec],
        cv_splits: int,
    ) -> pd.DataFrame:
        """Evaluate every (k, model) pair under grouped K-fold CV."""
        groups = df_train["profId"].to_numpy()
        y_all = df_train[self.target_col].to_numpy()

        gkf = GroupKFold(n_splits=cv_splits)
        rows: List[FoldEval] = []

        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(df_train, y_all, groups=groups), start=1):
            tr_df = df_train.iloc[tr_idx]
            va_df = df_train.iloc[va_idx]

            tr_comments = tr_df["comment_clean"].astype(str).tolist()
            y_tr = tr_df[self.target_col].to_numpy()
            tr_tags_block = self.feature_pipeline.tag_matrix(tr_df, tag_cols)

            va_comments = va_df["comment_clean"].astype(str).tolist()
            y_va = va_df[self.target_col].to_numpy()
            va_tags_block = self.feature_pipeline.tag_matrix(va_df, tag_cols)

            for k in k_values:
                vectorizer, nmf, x_tr = self.feature_pipeline.fit_transform(
                    comments=tr_comments,
                    tags_block=tr_tags_block,
                    k=k,
                )
                x_va = self.feature_pipeline.transform(va_comments, va_tags_block, vectorizer, nmf)

                for spec in specs:
                    y_pred = ModelEvaluator.fit_predict_model(
                        spec,
                        x_tr,
                        y_tr,
                        x_va,
                        min_label=1,
                        max_label=5,
                    )
                    acc = accuracy_score(y_va, y_pred)
                    macro = f1_score(y_va, y_pred, average="macro")
                    mae = mean_absolute_error(y_va, y_pred)
                    rows.append(
                        FoldEval(
                            fold=fold_idx,
                            k=k,
                            model=spec.name,
                            accuracy=acc,
                            macro_f1=macro,
                            mae=mae,
                        )
                    )
                    print(
                        f"[CV] fold={fold_idx} k={k} model={spec.name} "
                        f"accuracy={acc:.4f} macro_f1={macro:.4f} mae={mae:.4f}"
                    )

        return pd.DataFrame([r.__dict__ for r in rows])

    @staticmethod
    def summarize_cv(df_cv: pd.DataFrame) -> pd.DataFrame:
        """Aggregate mean/std metrics by (k, model) and rank best-first."""
        summary = (
            df_cv.groupby(["k", "model"], as_index=False)
            .agg(
                mean_accuracy=("accuracy", "mean"),
                std_accuracy=("accuracy", "std"),
                mean_macro_f1=("macro_f1", "mean"),
                std_macro_f1=("macro_f1", "std"),
                mean_mae=("mae", "mean"),
                std_mae=("mae", "std"),
            )
            .sort_values(["mean_accuracy", "mean_macro_f1", "mean_mae"], ascending=[False, False, True])
            .reset_index(drop=True)
        )

        print("\n[CV] Summary by (k, model)")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return summary


class ExperimentRunner:
    """Orchestrate data prep, CV search, optional test eval, and CSV outputs."""

    def __init__(self, config: RunConfig):
        """Initialize all experiment services from a single run config."""
        self.config = config
        self.dataset_manager = ReviewDatasetManager(target_col=config.target_col)
        self.feature_pipeline = FeaturePipeline(random_state=config.random_state)
        self.cv_runner = CrossValidationRunner(
            feature_pipeline=self.feature_pipeline,
            target_col=config.target_col,
        )
        self.registry = ModelRegistry(random_state=config.random_state)

    def _resolve_models(self) -> List[ModelSpec]:
        """Resolve requested models or default to the full registry."""
        specs = self.registry.get_specs()
        if not self.config.requested_models:
            return specs

        available = {s.name: s for s in specs}
        chosen = [available[m] for m in self.config.requested_models]
        print(f"Requested models: {[s.name for s in chosen]}")
        return chosen

    def _run_cv_block(
        self,
        df_train: pd.DataFrame,
        tag_cols: Sequence[str],
        specs: Sequence[ModelSpec],
        k_values: Sequence[int],
        cv_splits: int,
        label: str,
    ) -> pd.DataFrame | None:
        """Execute one CV block and return fold metrics, if any models exist."""
        if not specs:
            return None
        print(f"Running {label} with cv_splits={cv_splits}, k_values={list(k_values)}")
        return self.cv_runner.grouped_cv_search(
            df_train=df_train,
            tag_cols=tag_cols,
            k_values=k_values,
            specs=specs,
            cv_splits=cv_splits,
        )

    def _evaluate_test(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        tag_cols: Sequence[str],
        best_k: int,
        best_spec: ModelSpec,
    ) -> dict:
        """Train the selected best model on train split and evaluate on test split."""
        target_col = self.config.target_col

        train_comments = df_train["comment_clean"].astype(str).tolist()
        y_train = df_train[target_col].to_numpy()
        train_tags_block = self.feature_pipeline.tag_matrix(df_train, tag_cols)

        test_comments = df_test["comment_clean"].astype(str).tolist()
        y_test = df_test[target_col].to_numpy()
        test_tags_block = self.feature_pipeline.tag_matrix(df_test, tag_cols)

        vectorizer, nmf, x_train = self.feature_pipeline.fit_transform(
            comments=train_comments,
            tags_block=train_tags_block,
            k=best_k,
        )
        x_test = self.feature_pipeline.transform(test_comments, test_tags_block, vectorizer, nmf)

        y_pred = ModelEvaluator.fit_predict_model(best_spec, x_train, y_train, x_test, min_label=1, max_label=5)

        test_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "mae": mean_absolute_error(y_test, y_pred),
        }

        print("\n[Test] Best configuration")
        print(f"target={target_col}, k={best_k}, model={best_spec.name}")
        print("\n[Test] Metrics")
        print(f"accuracy={test_metrics['accuracy']:.4f}")
        print(f"macro_f1={test_metrics['macro_f1']:.4f}")
        print(f"mae={test_metrics['mae']:.4f}")
        print("\n[Test] Classification report")
        print(classification_report(y_test, y_pred, digits=4))
        print("[Test] Confusion matrix (rows=true, cols=pred)")
        print(confusion_matrix(y_test, y_pred))

        return test_metrics

    @staticmethod
    def _build_split_output_tables(
        df_cv: pd.DataFrame,
        summary: pd.DataFrame,
        test_metrics: dict | None,
        best_k: int | None,
        best_model_name: str | None,
        include_test_row: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build lean output tables for fold-level and model-summary CSVs."""
        cv_out = df_cv.loc[:, ["fold", "k", "model", "accuracy", "macro_f1", "mae"]].copy()
        cv_out.insert(0, "result_type", "cv_fold")
        cv_out = cv_out.loc[:, ["result_type", "fold", "k", "model", "accuracy", "macro_f1", "mae"]]

        summary_out = summary.loc[
            :,
            [
                "k",
                "model",
                "mean_accuracy",
                "std_accuracy",
                "mean_macro_f1",
                "std_macro_f1",
                "mean_mae",
                "std_mae",
            ],
        ].copy()
        summary_out.insert(0, "result_type", "cv_summary")
        summary_out = summary_out.loc[
            :,
            [
                "result_type",
                "k",
                "model",
                "mean_accuracy",
                "std_accuracy",
                "mean_macro_f1",
                "std_macro_f1",
                "mean_mae",
                "std_mae",
            ],
        ]

        if include_test_row:
            test_out = pd.DataFrame(
                [
                    {
                        "result_type": "test",
                        "k": best_k,
                        "model": best_model_name,
                        "accuracy": test_metrics["accuracy"],
                        "macro_f1": test_metrics["macro_f1"],
                        "mae": test_metrics["mae"],
                    }
                ]
            )
            model_summary_out = pd.concat([summary_out, test_out], ignore_index=True)
        else:
            model_summary_out = summary_out
        return cv_out, model_summary_out

    def run(self) -> None:
        """Run the full experiment lifecycle and write all configured outputs."""
        df = self.dataset_manager.load_and_validate(self.config.input_csv, self.config.max_rows)
        tag_cols = self.dataset_manager.get_tag_columns(df)
        df_train, df_test = self.dataset_manager.split_train_test(
            df=df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        print(f"Target column: {self.config.target_col}")
        print(f"Rows after cleaning: {len(df):,}")
        print(f"Train rows: {len(df_train):,}, Test rows: {len(df_test):,}")
        print(
            "Unique profs -> "
            f"train: {df_train['profId'].nunique():,}, "
            f"test: {df_test['profId'].nunique():,}"
        )
        print(f"Tag feature columns: {len(tag_cols)}")
        print(f"full k_values: {self.config.k_values}")
        print(f"linear_svc safe k_values: {self.config.lsvc_safe_k_values}")
        print(f"include_test_row: {self.config.include_test_row}")

        specs = self._resolve_models()
        safe_specs = [s for s in specs if s.name == "linear_svc"]
        full_specs = [s for s in specs if s.name != "linear_svc"]
        cv_parts: List[pd.DataFrame] = []

        df_cv_full = self._run_cv_block(
            df_train=df_train,
            tag_cols=tag_cols,
            specs=full_specs,
            k_values=self.config.k_values,
            cv_splits=self.config.cv_splits,
            label=f"full mode for models {[s.name for s in full_specs]}",
        )
        if df_cv_full is not None:
            cv_parts.append(df_cv_full)

        df_cv_safe = self._run_cv_block(
            df_train=df_train,
            tag_cols=tag_cols,
            specs=safe_specs,
            k_values=self.config.lsvc_safe_k_values,
            cv_splits=self.config.lsvc_safe_cv_splits,
            label="linear_svc safe mode",
        )
        if df_cv_safe is not None:
            cv_parts.append(df_cv_safe)

        df_cv = pd.concat(cv_parts, ignore_index=True)
        summary = self.cv_runner.summarize_cv(df_cv)

        best_row = summary.iloc[0]
        best_k = int(best_row["k"])
        best_model_name = str(best_row["model"])
        best_spec = next(s for s in specs if s.name == best_model_name)

        test_metrics = None
        if self.config.include_test_row:
            test_metrics = self._evaluate_test(
                df_train=df_train,
                df_test=df_test,
                tag_cols=tag_cols,
                best_k=best_k,
                best_spec=best_spec,
            )

        cv_folds_out, model_summary_out = self._build_split_output_tables(
            df_cv=df_cv,
            summary=summary,
            test_metrics=test_metrics,
            best_k=best_k if self.config.include_test_row else None,
            best_model_name=best_model_name if self.config.include_test_row else None,
            include_test_row=self.config.include_test_row,
        )
        results = pd.concat([cv_folds_out, model_summary_out], ignore_index=True)
        cv_folds_out.to_csv(self.config.output_cv_folds_csv, index=False)
        model_summary_out.to_csv(self.config.output_model_summary_csv, index=False)
        results.to_csv(self.config.output_csv, index=False)
        print(f"Saved CV fold scores to: {self.config.output_cv_folds_csv}")
        print(f"Saved model summary to: {self.config.output_model_summary_csv}")
        print(f"Saved consolidated results to: {self.config.output_csv}")
