"""Preprocessing transforms for raw RateMyProfessor review exports."""

import re

import pandas as pd


class ReviewPreprocessor:
    """Preprocess RateMyProfessor-style review data into model-ready features.

    The pipeline removes noisy comments, normalizes text fields, one-hot encodes
    `ratingTags`, and adds a boolean label flag indicating whether tags exist.
    """

    def __init__(self, comment_col="comment", rating_tags_col="ratingTags", comment_no_numbers_col="comment_no_numbers", comment_clean_col="comment_clean", tag_prefix="tag", labeled_col="labeled"):
        """Initialize configurable column names used by the preprocessing steps."""
        self.comment_col = comment_col
        self.rating_tags_col = rating_tags_col
        self.comment_no_numbers_col = comment_no_numbers_col
        self.comment_clean_col = comment_clean_col
        self.tag_prefix = tag_prefix
        self.labeled_col = labeled_col

    @staticmethod
    def _split_tags(series):
        """Split raw tag strings on '--' and return normalized tag lists."""
        return (
            series.fillna("")
            .astype(str)
            .str.split("--")
            .map(lambda items: [t.strip() for t in items if t and t.strip()])
        )

    @staticmethod
    def _safe_tag_name(raw_tag):
        """Normalize a raw tag into a filesystem/column-safe identifier."""
        safe_tag = re.sub(r"[^0-9A-Za-z]+", "_", raw_tag).strip("_").lower()
        return safe_tag or "unknown"

    def drop_no_comments_rows(self, df):
        """Return a copy of `df` excluding rows where comment is exactly 'no comments'."""
        no_comment_mask = (
            df[self.comment_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("no comments")
        )
        return df.loc[~no_comment_mask].copy()

    def drop_html_escaped_comments(self, df):
        """Return a copy of `df` without comments containing HTML-escaped entities."""
        escaped_pattern = r"&(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);"
        has_escaped = df[self.comment_col].astype(str).str.contains(escaped_pattern, regex=True, na=False)
        return df.loc[~has_escaped].copy()

    def remove_standalone_numbers_from_comments(self, df):
        """Create the no-number comment column by removing digits and collapsing spaces."""
        out = df.copy()
        out[self.comment_no_numbers_col] = out[self.comment_col].fillna("").astype(str).str.replace(r"\d+", " ", regex=True)
        out[self.comment_no_numbers_col] = out[self.comment_no_numbers_col].str.replace(r"\s+", " ", regex=True).str.strip()
        return out

    def remove_repeated_char_words_from_comments(self, df):
        """Create the clean comment column by removing tokens like 'zzz' or '__'."""
        out = df.copy()
        out[self.comment_clean_col] = out[self.comment_no_numbers_col].fillna("").astype(str)
        out[self.comment_clean_col] = out[self.comment_clean_col].str.replace(r"(?u)\b(\w)\1+\b", " ", regex=True)
        out[self.comment_clean_col] = out[self.comment_clean_col].str.replace(r"\s+", " ", regex=True).str.strip()
        return out

    def one_hot_encode_rating_tags(self, df):
        """Split `ratingTags` on '--' and add one-hot columns using `tag_prefix`."""
        out = df.copy()
        tags = self._split_tags(out[self.rating_tags_col])

        all_tags = sorted({tag for items in tags for tag in items})
        for tag in all_tags:
            col_name = f"{self.tag_prefix}_{self._safe_tag_name(tag)}"
            out[col_name] = tags.map(lambda items: int(tag in items))

        return out

    def add_labeled_column_from_rating_tags(self, df):
        """Add boolean `labeled_col`: True when a row has at least one rating tag."""
        out = df.copy()
        tags = self._split_tags(out[self.rating_tags_col])
        out[self.labeled_col] = tags.map(lambda items: len(items) >= 1)
        return out

    def preprocess(self, df):
        """Run the full preprocessing pipeline and return the transformed dataframe."""
        # Keep a copy so all transforms are side-effect free.
        out = df.copy()
        out = self.drop_no_comments_rows(out)
        out = self.drop_html_escaped_comments(out)
        out = self.remove_standalone_numbers_from_comments(out)
        out = self.remove_repeated_char_words_from_comments(out)
        out = self.one_hot_encode_rating_tags(out)
        out = self.add_labeled_column_from_rating_tags(out)
        return out
