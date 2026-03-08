"""Dataset loading, cleaning, and train/test split utilities."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from rmp_preprocess import ReviewPreprocessor


class ReviewDatasetManager:
    """Manage input resolution, preprocessing, and grouped data splits."""

    @staticmethod
    def resolve_input_csv(path: str) -> str:
        """Resolve relative CSV paths against this module's directory."""
        base = Path(path)
        if base.is_absolute():
            return str(base)
        return str(Path(__file__).resolve().parent / base)

    @staticmethod
    def clean_reviews(df_raw: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing and enforce the required modeling schema."""
        processor = ReviewPreprocessor()
        out = processor.preprocess(df_raw)

        out["clarityRating"] = pd.to_numeric(out["clarityRating"], errors="coerce")
        out = out.dropna(subset=["clarityRating", "profId", "comment_clean"]).copy()
        out = out[out["comment_clean"].astype(str).str.strip().ne("")].copy()

        out = out[out["clarityRating"].between(1, 5)].copy()
        out["clarityRating"] = out["clarityRating"].astype(int)
        out["profId"] = out["profId"].astype(str)
        return out

    @classmethod
    def load_and_validate(cls, input_csv: str, max_rows: int | None) -> pd.DataFrame:
        """Load raw CSV and return a cleaned dataframe ready for modeling."""
        resolved_input_csv = cls.resolve_input_csv(input_csv)
        df_raw = pd.read_csv(resolved_input_csv, nrows=max_rows)
        return cls.clean_reviews(df_raw)

    @staticmethod
    def get_tag_columns(df: pd.DataFrame) -> List[str]:
        """Return sorted one-hot tag feature column names."""
        return sorted([c for c in df.columns if c.startswith("tag_")])

    @staticmethod
    def split_train_test(
        df: pd.DataFrame,
        test_size: float,
        random_state: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data by professor group to avoid train/test leakage."""
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        idx = np.arange(len(df))
        groups = df["profId"].to_numpy()
        train_idx, test_idx = next(gss.split(idx, groups=groups))
        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
        )
