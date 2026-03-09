"""Feature engineering pipeline for text and tag features."""

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


DOMAIN_STOPWORDS = {
    "ucsd",
    "ucsb",
    "uci",
    "ucr",
    "ucb",
    "ucd",
    "ucsc",
    "ucm",
    "university",
    "california",
    "berkeley",
    "irvine",
    "davis",
    "san",
    "diego",
    "santa",
    "cruz",
    "barbara",
    "merced",
    "riverside",
}


class FeaturePipeline:
    """Create sparse model features from comments and structured tags."""

    def __init__(self, random_state: int):
        """Initialize the feature pipeline with a deterministic random seed."""
        self.random_state = random_state

    @staticmethod
    def tag_matrix(df: pd.DataFrame, tag_cols: Sequence[str]) -> csr_matrix:
        """Build a sparse tag matrix from one-hot tag columns."""
        if len(tag_cols) == 0:
            return csr_matrix((len(df), 0), dtype=np.float32)
        return csr_matrix(df.loc[:, tag_cols].to_numpy(dtype=np.float32), dtype=np.float32)

    @staticmethod
    def build_vectorizer() -> TfidfVectorizer:
        """Construct the TF-IDF vectorizer used across experiments."""
        all_stopwords = ENGLISH_STOP_WORDS.union(DOMAIN_STOPWORDS)
        return TfidfVectorizer(lowercase=True, norm="l2", binary=True, token_pattern=r"(?u)\b\w{3,}\b", strip_accents="unicode", stop_words=list(all_stopwords), ngram_range=(1, 3), min_df=200, max_df=0.6, sublinear_tf=True, max_features=200_000, analyzer="word")

    def build_nmf(self, k: int) -> NMF:
        """Create the NMF projection model for a requested topic dimension."""
        return NMF(n_components=k, init="random", random_state=self.random_state, max_iter=800, solver="cd", beta_loss="frobenius")

    def fit_transform(self, comments: Sequence[str], tags_block: csr_matrix, k: int) -> Tuple[TfidfVectorizer, NMF, csr_matrix]:
        """Fit vectorizer and NMF on train comments and return combined features."""
        vectorizer = self.build_vectorizer()
        x_tfidf = vectorizer.fit_transform(comments)

        nmf = self.build_nmf(k=k)
        w = nmf.fit_transform(x_tfidf)

        x = hstack([csr_matrix(w), tags_block], format="csr")
        return vectorizer, nmf, x

    @staticmethod
    def transform(comments: Sequence[str], tags_block: csr_matrix, vectorizer: TfidfVectorizer, nmf: NMF) -> csr_matrix:
        """Transform comments/tags using a fitted vectorizer and NMF model."""
        x_tfidf = vectorizer.transform(comments)
        w = nmf.transform(x_tfidf)
        return hstack([csr_matrix(w), tags_block], format="csr")
