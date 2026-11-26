"""Step 9: clustering - do the attack types form natural groups without labels?

Usage (from the project root, with the virtual environment active):

    python scripts/09_cluster.py                 # k-means sweep, Ward, DBSCAN on the default sample
    python scripts/09_cluster.py --sample-size 20000 --skip-dbscan   # quick run

Data: a class-capped sample of the preprocessed TRAINING rows (every small class kept whole, the
large ones capped), split in two halves: clusters are fitted on one half; the cluster-majority
classifier is scored on the other.  The test set is not used.

Outputs (reports/mining/clustering/):
    clustering_report.md / .json      readable report and machine-readable numbers
    kmeans_sweep.csv                  scores for every k
    composition_k<k>.csv              which classes fall in which cluster
    figures/*.png                     elbow, scores vs k, composition heatmap, DBSCAN k-distance

Exit codes: 0 done; 1 data missing or stale.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import settings
from nids.mining import clustering as C
from nids.mining.reporting import class_names, markdown_table, plt, save_figure, write_report
from nids.training.data_access import DataNotReadyError, class_capped_indices, load_training_data
from nids.utils.logger import configure_logging, get_logger

logger = get_logger("scripts.cluster")
OUT = settings.MINING_REPORT_DIR / "clustering"


def plot_sweep(sweep: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(sweep["k"], sweep["inertia"], "o-", color="#3b6ea8")
    ax.set(xlabel="k", ylabel="inertia (within-cluster sum of squares)", title="Elbow curve")
    ax2 = ax.twinx()
    ax2.plot(sweep["k"], sweep["silhouette"], "s--", color="#c0504d")
    ax2.set_ylabel("silhouette", color="#c0504d")
    ax = axes[1]
    for col, style in (("ari", "o-"), ("nmi", "s-"), ("purity", "^-"), ("holdout_macro_f1", "d--")):
        ax.plot(sweep["k"], sweep[col], style, label=col)
    ax.set(xlabel="k", ylabel="score", ylim=(0, 1.02), title="Agreement of k-means clusters with the true classes")
    ax.legend()
    save_figure(fig, OUT / "figures" / "kmeans_sweep.png")


def plot_composition(y_names: np.ndarray, clusters: np.ndarray, k: int) -> None:
    table = pd.crosstab(pd.Series(y_names, name="class"), pd.Series(clusters, name="cluster"), normalize="index")
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * table.shape[1] + 4), 6.5))
    im = ax.imshow(table.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns])
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set(xlabel="k-means cluster", title=f"Where each class goes (k={k}; row = class, share of its rows)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    save_figure(fig, OUT / "figures" / f"composition_k{k}.png")


def plot_k_distance(curve: np.ndarray, dbscan: pd.DataFrame, min_samples: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(curve.size), curve, color="#3b6ea8")
    for eps in dbscan["eps"]:
        ax.axhline(eps, ls="--", lw=0.8, color="#c0504d")
    ax.set(xlabel="rows sorted by distance", ylabel=f"distance to {min_samples}th neighbour", yscale="log", title="DBSCAN k-distance curve (dashed: eps tried)")
    save_figure(fig, OUT / "figures" / "dbscan_k_distance.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample-size", type=int, default=settings.MINING_SAMPLE_SIZE)
    parser.add_argument("--k-composition", type=int, default=15, help="k whose cluster composition is reported (default 15 = number of classes)")
    parser.add_argument("--skip-dbscan", action="store_true")
    args = parser.parse_args(argv)
    configure_logging()

    try:
        data = load_training_data(mmap=True)
    except DataNotReadyError as exc:
        print(f"ERROR: {exc}")
        return 1
    names = class_names(data.codec)
    idx = class_capped_indices(data.y, 2 * args.sample_size)
    X_all, y_all = np.asarray(data.X[idx], dtype="float32"), data.y[idx]
    X_fit, X_eval, y_fit, y_eval = train_test_split(X_all, y_all, test_size=0.5, stratify=y_all, random_state=settings.RANDOM_SEED)
    print(f"Clustering sample: {len(y_fit):,} fit rows, {len(y_eval):,} evaluation rows, {X_fit.shape[1]} features")

    sweep, models = C.kmeans_sweep(X_fit, y_fit, X_eval, y_eval, n_classes=data.n_classes)
    OUT.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(OUT / "kmeans_sweep.csv", index=False)
    plot_sweep(sweep)
    best_sil = sweep.loc[sweep["silhouette"].idxmax()]
    k_comp = args.k_composition if args.k_composition in models else int(best_sil["k"])
    comp_labels = models[k_comp].labels_
    composition = C.cluster_composition(np.asarray(names, dtype=object)[y_fit], comp_labels)
    composition.to_csv(OUT / f"composition_k{k_comp}.csv", index=False)
    plot_composition(np.asarray(names, dtype=object)[y_fit], comp_labels, k_comp)

    # hierarchical (Ward) on a smaller sample: memory is O(n^2)
    n_ward = min(10_000, len(y_fit))
    ward_idx = class_capped_indices(y_fit, n_ward)
    _, ward_scores = C.agglomerative(X_fit[ward_idx], y_fit[ward_idx], k_comp)
    kmeans_same = C.cluster_scores(X_fit[ward_idx], y_fit[ward_idx], models[k_comp].predict(X_fit[ward_idx]))

    dbscan = pd.DataFrame()
    if not args.skip_dbscan:
        db_idx = class_capped_indices(y_fit, min(settings.DBSCAN_SAMPLE_SIZE, len(y_fit)))
        dbscan, curve, _ = C.dbscan_grid(X_fit[db_idx], y_fit[db_idx], min_samples=10)
        plot_k_distance(curve, dbscan, 10)

    compare = pd.DataFrame(
        [
            {"algorithm": f"k-means (k={k_comp})", **{k: kmeans_same[k] for k in ("n_clusters", "ari", "nmi", "purity", "silhouette")}},
            {"algorithm": f"Ward agglomerative (k={k_comp})", **{k: ward_scores[k] for k in ("n_clusters", "ari", "nmi", "purity", "silhouette")}},
        ]
    )
    pure = composition[composition["majority_share"] >= 0.95]
    mixed = composition[composition["majority_share"] < 0.6]

    md = [
        "# Clustering of network flows",
        "",
        f"Sample: {len(y_fit):,} class-capped **training** rows to fit, {len(y_eval):,} other training rows to score the "
        f"cluster-majority classifier; {X_fit.shape[1]} preprocessed features (log + standard scaling). The labels are "
        "never used to fit a clustering, only to judge it afterwards. The sample is class-capped, so small attack "
        "classes are over-represented compared with real traffic.",
        "",
        "## k-means sweep",
        "",
        markdown_table(sweep),
        "",
        "* **silhouette / Davies-Bouldin** use no labels (silhouette: higher is better; Davies-Bouldin: lower is better).",
        "* **ARI, NMI, purity** compare the clusters with the true classes.",
        "* **holdout_macro_f1**: each cluster is named after its majority class (fit rows) and that rule is scored on the evaluation rows.",
        "",
        f"Best k by silhouette (no labels used): **k = {int(best_sil['k'])}** (silhouette {best_sil['silhouette']:.3f}). "
        f"Best agreement with the classes: k = {int(sweep.loc[sweep['ari'].idxmax(), 'k'])} by ARI, "
        f"k = {int(sweep.loc[sweep['holdout_macro_f1'].idxmax(), 'k'])} by cluster-majority macro F1 "
        f"({sweep['holdout_macro_f1'].max():.3f}).",
        "",
        "![sweep](figures/kmeans_sweep.png)",
        "",
        f"## What the clusters contain (k = {k_comp})",
        "",
        f"{len(pure)} of {len(composition)} clusters are at least 95% one class; {len(mixed)} are mixed (majority below 60%).",
        "",
        markdown_table(composition, digits=3),
        "",
        f"![composition](figures/composition_k{k_comp}.png)",
        "",
        "## k-means vs hierarchical clustering (same rows)",
        "",
        f"Ward needs O(n^2) memory, so both are scored on the same {n_ward:,} class-capped rows.",
        "",
        markdown_table(compare, digits=3),
        "",
    ]
    if not dbscan.empty:
        md += [
            "## DBSCAN (density-based)",
            "",
            f"{len(db_idx):,} rows, min_samples = 10, eps taken at quantiles of the k-distance curve. "
            "Noise points (label -1) are not forced into a cluster.",
            "",
            markdown_table(dbscan[["eps_quantile", "eps", "n_clusters", "noise_fraction", "ari", "nmi", "purity", "silhouette"]], digits=3),
            "",
            "![k-distance](figures/dbscan_k_distance.png)",
            "",
        ]
    md += [
        "## Limits",
        "",
        "* The sample is class-capped; on real traffic proportions BENIGN would dominate every cluster.",
        "* Euclidean distance on scaled log-features is one choice among many; another metric or feature set gives other clusters.",
        "* Silhouette is computed on a random subset of rows (it is O(n^2)).",
    ]
    summary = {
        "fit_rows": int(len(y_fit)),
        "eval_rows": int(len(y_eval)),
        "kmeans_sweep": sweep.to_dict(orient="records"),
        "best_k_silhouette": int(best_sil["k"]),
        "composition_k": k_comp,
        "kmeans_vs_ward": compare.to_dict(orient="records"),
        "dbscan": dbscan.to_dict(orient="records"),
    }
    md_path, _ = write_report(OUT, "clustering_report", "\n".join(md), summary)
    print(markdown_table(sweep[["k", "silhouette", "ari", "nmi", "purity", "holdout_macro_f1"]], digits=3))
    print(f"\nReport: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
