"""2-D maps of the flows: PCA (linear, keeps global distances) and t-SNE (non-linear, keeps neighbourhoods).

Both are fitted on a class-capped sample of the preprocessed training rows, so the small classes
are visible; the map therefore does not show the real class proportions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import settings
from nids.utils.logger import get_logger

logger = get_logger("mining.projection")


def pca_analysis(X: np.ndarray, feature_names: list[str], n_loadings: int = 5) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    """Full PCA: 2-D coordinates, the variance curve, and the top loadings of PC1 and PC2."""
    pca = PCA(random_state=settings.RANDOM_SEED).fit(X)
    ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    info = {
        "explained_variance_ratio": ratio.tolist(),
        "cumulative": cumulative.tolist(),
        "pc1_pc2_variance": float(cumulative[1]),
        "components_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
        "components_for_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
        "n_features": int(X.shape[1]),
    }
    rows = []
    for pc in (0, 1):
        loadings = pca.components_[pc]
        for i in np.argsort(-np.abs(loadings))[:n_loadings]:
            rows.append({"component": f"PC{pc + 1}", "feature": feature_names[i], "loading": float(loadings[i])})
    return pca.transform(X)[:, :2], info, pd.DataFrame(rows)


def tsne_map(X: np.ndarray, perplexity: float = 40.0, pca_dims: int = 20) -> np.ndarray:
    """t-SNE coordinates (after a PCA to ``pca_dims`` dimensions, the usual speed-up)."""
    reduced = PCA(n_components=min(pca_dims, X.shape[1]), random_state=settings.RANDOM_SEED).fit_transform(X)
    logger.info("t-SNE on %d rows (perplexity %.0f)", X.shape[0], perplexity)
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=settings.RANDOM_SEED, n_jobs=settings.N_JOBS).fit_transform(reduced)
