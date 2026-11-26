"""Step 12: what the data and the models look like - PCA / t-SNE maps and feature importance.

Usage (from the project root, with the virtual environment active):

    python scripts/12_explain.py
    python scripts/12_explain.py --skip-tsne --importance-rows 10000      # quick run

Part 1 (projection): PCA and t-SNE of a class-capped sample of TRAINING rows, coloured by class.
Part 2 (importance): built-in importance of the saved baseline Random Forest and XGBoost, and
permutation importance (macro-F1 drop) on a class-capped slice of the TEST set.

Outputs (reports/mining/explain/): explain_report.md / .json, importance_*.csv, figures/*.png

Exit codes: 0 done; 1 data or models missing.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from config import settings
from nids.mining import importance as I
from nids.mining import projection as P
from nids.mining.reporting import class_names, markdown_table, plt, save_figure, write_report
from nids.persistence import model_store
from nids.training.data_access import DataNotReadyError, class_capped_indices, load_test_data, load_training_data
from nids.utils.io_utils import read_json
from nids.utils.logger import configure_logging, get_logger

logger = get_logger("scripts.explain")
OUT = settings.MINING_REPORT_DIR / "explain"
IMPORTANCE_MODELS = ("random_forest", "xgboost")


def scatter_by_class(coords: np.ndarray, y_names: np.ndarray, title: str, path_name: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7))
    tab20 = plt.get_cmap("tab20").colors
    palette = list(tab20[0::2]) + list(tab20[1::2])  # 10 distinct dark hues first, then their light partners
    classes = pd.Series(y_names).value_counts().index  # large classes first, small ones drawn on top
    for i, cls in enumerate(classes):
        mask = y_names == cls
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4 if mask.sum() > 200 else 14, alpha=0.6, color=palette[i % 20], label=f"{cls} ({mask.sum()})")
    ax.set(title=title, xticks=[], yticks=[])
    ax.legend(markerscale=3, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save_figure(fig, OUT / "figures" / path_name)


def importance_plot(frames: dict[str, pd.DataFrame], value: str, title: str, path_name: str, top: int = 15) -> None:
    fig, axes = plt.subplots(1, len(frames), figsize=(6.5 * len(frames), 5.5))
    for ax, (name, frame) in zip(np.atleast_1d(axes), frames.items()):
        head = frame.head(top).iloc[::-1]
        err = head["f1_drop_std"] if "f1_drop_std" in head else None
        ax.barh(head["feature"], head[value], xerr=err, color="#3b6ea8")
        ax.set(title=name, xlabel=value)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(title)
    save_figure(fig, OUT / "figures" / path_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--projection-rows", type=int, default=settings.PROJECTION_SAMPLE_SIZE)
    parser.add_argument("--importance-rows", type=int, default=settings.IMPORTANCE_SAMPLE_SIZE)
    parser.add_argument("--skip-tsne", action="store_true")
    args = parser.parse_args(argv)
    configure_logging()

    try:
        train, test = load_training_data(mmap=True), load_test_data(mmap=True)
        models = {k: model_store.load_model(k, "trained") for k in IMPORTANCE_MODELS}
    except (DataNotReadyError, model_store.ModelStoreError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    names = np.asarray(class_names(train.codec), dtype=object)
    features = read_json(settings.FEATURE_NAMES_FILE)["model_feature_names"]
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Part 1: projections
    idx = class_capped_indices(train.y, args.projection_rows)
    X, y_names = np.asarray(train.X[idx]), names[train.y[idx]]
    coords, pca_info, loadings = P.pca_analysis(X, features)
    scatter_by_class(coords, y_names, f"PCA (PC1+PC2 = {pca_info['pc1_pc2_variance']:.1%} of the variance)", "pca_map.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(pca_info["explained_variance_ratio"]) + 1), pca_info["explained_variance_ratio"], color="#3b6ea8", label="per component")
    ax.plot(range(1, len(pca_info["cumulative"]) + 1), pca_info["cumulative"], "o-", color="#c0504d", ms=3, label="cumulative")
    ax.axhline(0.95, ls="--", lw=0.8, color="grey")
    ax.set(xlabel="principal component", ylabel="explained variance", title="PCA scree plot")
    ax.legend()
    save_figure(fig, OUT / "figures" / "pca_scree.png")
    loadings.to_csv(OUT / "pca_loadings.csv", index=False)
    if not args.skip_tsne:
        scatter_by_class(P.tsne_map(X), y_names, "t-SNE map (neighbourhoods preserved; distances between groups are not meaningful)", "tsne_map.png")

    # ---- Part 2: importance
    tidx = class_capped_indices(test.y, args.importance_rows)
    X_imp, y_imp = np.asarray(test.X[tidx]), test.y[tidx]
    builtin, permutation = {}, {}
    for key, model in models.items():
        b = I.builtin_importance(model, features)
        if b is not None:
            builtin[model.display_name] = b
            b.to_csv(OUT / f"importance_builtin_{key}.csv", index=False)
        logger.info("Permutation importance for %s on %d test rows ...", key, len(y_imp))
        p = I.permutation_table(model, X_imp, y_imp, features)
        permutation[model.display_name] = p
        p.to_csv(OUT / f"importance_permutation_{key}.csv", index=False)
    importance_plot(builtin, "importance", "Built-in importance (normalised)", "importance_builtin.png")
    importance_plot(permutation, "f1_drop_mean", f"Permutation importance: macro-F1 drop on {len(y_imp):,} test flows", "importance_permutation.png")

    agreement = {}
    for name in permutation:
        if name in builtin:
            agreement[name] = I.rank_agreement(builtin[name], permutation[name], "importance", "f1_drop_mean")
    rf_name, xgb_name = (models[k].display_name for k in IMPORTANCE_MODELS)
    between = I.rank_agreement(permutation[rf_name], permutation[xgb_name], "f1_drop_mean", "f1_drop_mean")
    port_rank = {n: int(f.index[f["feature"] == "Destination Port"][0]) + 1 for n, f in permutation.items() if "Destination Port" in set(f["feature"])}

    top_table = pd.concat(
        [permutation[n].head(10).reset_index(drop=True).add_prefix(f"{n}: ")[[f"{n}: feature", f"{n}: f1_drop_mean"]] for n in permutation], axis=1
    )
    md = [
        "# Projections and feature importance",
        "",
        "## PCA",
        "",
        f"{len(idx):,} class-capped training rows, {pca_info['n_features']} preprocessed features. PC1 and PC2 explain "
        f"{pca_info['pc1_pc2_variance']:.1%} of the variance; {pca_info['components_for_90pct']} components are needed for 90% "
        f"and {pca_info['components_for_95pct']} for 95%, so the classes cannot be fully separated in a 2-D linear view.",
        "",
        "![PCA](figures/pca_map.png)",
        "",
        "![scree](figures/pca_scree.png)",
        "",
        "Largest loadings of the first two components:",
        "",
        markdown_table(loadings, digits=3),
        "",
    ]
    if not args.skip_tsne:
        md += [
            "## t-SNE",
            "",
            "Same rows. t-SNE keeps neighbourhoods: compact islands of one colour are attack types that are easy to "
            "recognise; colours mixed inside one island (for example the web attacks) are the classes the models confuse.",
            "",
            "![t-SNE](figures/tsne_map.png)",
            "",
        ]
    md += [
        "## Feature importance",
        "",
        f"Models: the saved **baseline** Random Forest and XGBoost (the versions used by the final set). Permutation "
        f"importance = mean drop of macro F1 when one feature is shuffled ({settings.IMPORTANCE_REPEATS} repeats) on "
        f"{len(y_imp):,} class-capped **test** flows.",
        "",
        "Top 10 features by permutation importance:",
        "",
        markdown_table(top_table, digits=4),
        "",
        "![permutation](figures/importance_permutation.png)",
        "",
        "![builtin](figures/importance_builtin.png)",
        "",
        "Agreement between rankings:",
        "",
        *[f"* {n}: built-in vs permutation share {a['n_shared']} of their top 10; Spearman rank correlation {a['spearman']:.2f}." for n, a in agreement.items()],
        f"* Random Forest vs XGBoost (permutation): {between['n_shared']} shared in the top 10; Spearman {between['spearman']:.2f}.",
        *[f"* `Destination Port` ranks #{r} for {n} (permutation)." for n, r in port_rank.items()],
        "",
        "## Limits",
        "",
        "* Correlated features share importance: shuffling one leaves its partner, so both look less important than the pair is.",
        "* Built-in (impurity / gain) importance favours features with many distinct values and is measured on training data.",
        "* The test slice is class-capped, so the macro F1 weights small classes more than real traffic would.",
    ]
    summary = {
        "pca": pca_info,
        "permutation_top10": {n: f.head(10).to_dict(orient="records") for n, f in permutation.items()},
        "agreement": agreement,
        "rf_vs_xgb": between,
        "destination_port_rank": port_rank,
        "importance_test_rows": int(len(y_imp)),
    }
    md_path, _ = write_report(OUT, "explain_report", "\n".join(md), summary)
    print(markdown_table(top_table, digits=4))
    print(f"\nReport: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
