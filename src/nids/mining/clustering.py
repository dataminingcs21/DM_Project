"""Clustering of network flows: do attack types form natural groups without labels?

The labels are **never used to fit** a clustering.  They are used afterwards only, to judge
how well the clusters match the known classes:

* external scores: adjusted Rand index (ARI), normalised mutual information (NMI),
  homogeneity, completeness and purity;
* internal scores (no labels): silhouette (on a random subset, it is O(n^2)) and Davies-Bouldin;
* "cluster-majority" classifier: each cluster is named after the majority class of its
  training rows and applied to *other* rows.  Its macro F1 says how much of the class
  structure the clusters capture.

Algorithms: k-means for a sweep over k, Ward agglomerative clustering (hierarchical) on a
smaller sample, and DBSCAN (density based, finds noise points) with ``eps`` taken from the
k-distance curve.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    f1_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from config import settings
from nids.utils.logger import get_logger

logger = get_logger("mining.clustering")

NOISE = -1  # DBSCAN label of noise points


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def purity(y_true: np.ndarray, clusters: np.ndarray) -> float:
    """Fraction of rows whose class is the majority class of their cluster (noise counts as its own cluster)."""
    y_true, clusters = np.asarray(y_true), np.asarray(clusters)
    if y_true.shape != clusters.shape or y_true.size == 0:
        raise ValueError("y_true and clusters must be non-empty and have the same shape")
    table = pd.crosstab(clusters, y_true)
    return float(table.max(axis=1).sum() / y_true.size)


def cluster_majority_map(y_true: np.ndarray, clusters: np.ndarray) -> dict[int, int]:
    """For every cluster id, the most frequent class among its rows (ties: lowest class code)."""
    table = pd.crosstab(np.asarray(clusters), np.asarray(y_true))
    return {int(c): int(table.columns[np.argmax(row)]) for c, row in zip(table.index, table.to_numpy())}


def apply_majority_map(clusters: np.ndarray, mapping: dict[int, int], fallback: int) -> np.ndarray:
    """Class codes predicted from cluster ids; an unseen cluster id gets ``fallback``."""
    return np.array([mapping.get(int(c), fallback) for c in np.asarray(clusters)], dtype="int64")


def cluster_scores(
    X: np.ndarray,
    y_true: np.ndarray,
    clusters: np.ndarray,
    silhouette_sample: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """External and internal scores of one clustering. Noise points (-1) are excluded from the internal scores."""
    seed = settings.RANDOM_SEED if seed is None else seed
    silhouette_sample = settings.SILHOUETTE_SAMPLE_SIZE if silhouette_sample is None else silhouette_sample
    clusters = np.asarray(clusters)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, clusters)
    core = clusters != NOISE
    n_clusters = int(np.unique(clusters[core]).size)
    result: dict[str, Any] = {
        "n_clusters": n_clusters,
        "noise_fraction": float(1.0 - core.mean()),
        "ari": float(adjusted_rand_score(y_true, clusters)),
        "nmi": float(normalized_mutual_info_score(y_true, clusters)),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "purity": purity(y_true, clusters),
        "silhouette": None,
        "davies_bouldin": None,
    }
    if 2 <= n_clusters < int(core.sum()):
        Xc, cc = X[core], clusters[core]
        size = min(silhouette_sample, cc.size)
        result["silhouette"] = float(silhouette_score(Xc, cc, sample_size=size, random_state=seed))
        result["davies_bouldin"] = float(davies_bouldin_score(Xc, cc))
    return result


def holdout_macro_f1(
    y_fit: np.ndarray, clusters_fit: np.ndarray, y_eval: np.ndarray, clusters_eval: np.ndarray, n_classes: int
) -> float:
    """Macro F1 of the cluster-majority classifier: map learnt on the *fit* rows, scored on the *eval* rows."""
    mapping = cluster_majority_map(y_fit, clusters_fit)
    fallback = int(np.bincount(np.asarray(y_fit)).argmax())
    predicted = apply_majority_map(clusters_eval, mapping, fallback)
    return float(f1_score(y_eval, predicted, labels=np.arange(n_classes), average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------
def kmeans_sweep(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    k_values: tuple[int, ...] | list[int] | None = None,
    n_classes: int | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict[int, KMeans]]:
    """k-means for every k: scores on the fit rows, cluster-majority macro F1 on the eval rows."""
    seed = settings.RANDOM_SEED if seed is None else seed
    k_values = tuple(settings.CLUSTER_K_VALUES if k_values is None else k_values)
    n_classes = int(max(y_fit.max(), y_eval.max()) + 1) if n_classes is None else n_classes
    rows, models = [], {}
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X_fit)
        scores = cluster_scores(X_fit, y_fit, model.labels_, seed=seed)
        scores.update(
            k=k,
            inertia=float(model.inertia_),
            holdout_macro_f1=holdout_macro_f1(y_fit, model.labels_, y_eval, model.predict(X_eval), n_classes),
        )
        rows.append(scores)
        models[k] = model
        logger.info("k-means k=%d: ARI %.3f, NMI %.3f, purity %.3f, silhouette %s", k, scores["ari"], scores["nmi"], scores["purity"], scores["silhouette"])
    columns = ["k", "inertia", "silhouette", "davies_bouldin", "ari", "nmi", "homogeneity", "completeness", "purity", "holdout_macro_f1"]
    return pd.DataFrame(rows)[columns], models


def agglomerative(X: np.ndarray, y: np.ndarray, k: int, linkage: str = "ward") -> tuple[np.ndarray, dict[str, Any]]:
    """Hierarchical clustering (Ward by default) into ``k`` clusters; returns labels and scores."""
    labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)
    scores = cluster_scores(X, y, labels)
    scores.update(algorithm=f"agglomerative ({linkage})", k=k)
    return labels, scores


def k_distance(X: np.ndarray, min_samples: int) -> np.ndarray:
    """Sorted distance of every row to its ``min_samples``-th neighbour (the DBSCAN 'elbow' curve)."""
    nn = NearestNeighbors(n_neighbors=min_samples, n_jobs=settings.N_JOBS).fit(X)
    distances, _ = nn.kneighbors(X)
    return np.sort(distances[:, -1])


def dbscan_grid(
    X: np.ndarray, y: np.ndarray, min_samples: int = 10, quantiles: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95)
) -> tuple[pd.DataFrame, np.ndarray, dict[float, np.ndarray]]:
    """DBSCAN with ``eps`` at several quantiles of the k-distance curve. Returns scores, the curve and labels per eps."""
    curve = k_distance(X, min_samples)
    rows, labels_by_eps = [], {}
    for q in quantiles:
        eps = float(np.quantile(curve, q))
        if eps <= 0:
            continue
        labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=settings.N_JOBS).fit_predict(X)
        scores = cluster_scores(X, y, labels)
        scores.update(algorithm="DBSCAN", eps=eps, eps_quantile=q, min_samples=min_samples)
        rows.append(scores)
        labels_by_eps[eps] = labels
        logger.info("DBSCAN eps=%.3f (q=%.2f): %d clusters, noise %.1f%%, ARI %.3f", eps, q, scores["n_clusters"], 100 * scores["noise_fraction"], scores["ari"])
    return pd.DataFrame(rows), curve, labels_by_eps


def cluster_composition(y_names: np.ndarray, clusters: np.ndarray) -> pd.DataFrame:
    """One row per cluster: size, majority class, its share, and the second class."""
    table = pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(y_names, name="class"))
    rows = []
    for cluster, counts in table.iterrows():
        ordered = counts.sort_values(ascending=False)
        size = int(ordered.sum())
        second = ordered.index[1] if len(ordered) > 1 and ordered.iloc[1] > 0 else ""
        rows.append(
            {
                "cluster": "noise" if cluster == NOISE else int(cluster),
                "size": size,
                "majority_class": ordered.index[0],
                "majority_share": float(ordered.iloc[0] / size),
                "second_class": second,
                "second_share": float(ordered.iloc[1] / size) if second else 0.0,
                "n_classes_present": int((ordered > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
