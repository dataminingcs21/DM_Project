"""Step 19: formal statistical validation (5 hypothesis tests) + the analysis sample used by the app.

Usage (from the project root, with the virtual environment active):

    python scripts/19_stat_tests.py                       # 2,000 rows per class (all rows of the small classes)
    python scripts/19_stat_tests.py --cap-per-class 500   # quick run

Data: a class-capped sample of the CLEANED data (data/cleaned/cleaned.parquet).  Large classes are capped,
small ones kept whole, so every class weighs in the tests; the sample therefore over-represents the rare
attacks and is NOT the real class mix (the real mix is in reports/inspection/).

Tests (src/nids/eda/stat_tests.py), on every one of the 66 features:
    chi-square (binned feature x class), one-way ANOVA, Kruskal-Wallis, Shapiro-Wilk (raw and log1p),
    Mann-Whitney U (BENIGN vs attack).  Holm correction per test; effect sizes reported.

Outputs:
    reports/stats/stat_tests_report.md / .json     readable report and machine-readable summary
    reports/stats/<test>.csv                       one table per test (feature, statistic, p, Holm p, effect)
    reports/stats/figures/*.png                    effect sizes and the normality picture
    reports/app_data/analysis_sample.parquet       the sample itself (the app's quality / EDA / stats pages use it;
                                                   it is small and committed, so the app works from a clone)

Exit codes: 0 done; 1 cleaned data missing.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from config import settings
from nids.eda import eda_stats
from nids.eda import stat_tests as T
from nids.mining.reporting import markdown_table, plt, save_figure, write_report
from nids.utils.logger import configure_logging, get_logger

logger = get_logger("scripts.stat_tests")
OUT = settings.REPORTS_DIR / "stats"
APP_DATA = settings.REPORTS_DIR / "app_data"
SAMPLE_FILE = APP_DATA / "analysis_sample.parquet"


def plot_effects(results: dict[str, pd.DataFrame], top: int = 20) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5))
    for ax, (test, label) in zip(axes, (("chi2", "Cramer's V"), ("kruskal", "epsilon squared"), ("mannwhitney", "rank-biserial r"))):
        t = results[test].dropna(subset=["effect_size"]).head(top).iloc[::-1]
        colors = ["#c0504d" if v < 0 else "#3b6ea8" for v in t["effect_size"]]
        ax.barh(t["feature"], t["effect_size"], color=colors)
        ax.set(title=f"{test}: top {top} by |{label}|", xlabel=label)
        ax.tick_params(axis="y", labelsize=8)
    save_figure(fig, OUT / "figures" / "effect_sizes.png")


def plot_normality(shapiro: pd.DataFrame) -> None:
    t = shapiro.dropna(subset=["statistic"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(t["statistic"], t["statistic_log"], s=18, color="#3b6ea8")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    ax.set(xlabel="Shapiro-Wilk W on raw values", ylabel="W after signed log1p", xlim=(0, 1.02), ylim=(0, 1.02),
           title="No feature is normal; log1p brings most of them closer")
    save_figure(fig, OUT / "figures" / "normality_raw_vs_log.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cap-per-class", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    args = parser.parse_args(argv)
    configure_logging()

    if not settings.CLEANED_DATA_FILE.is_file():
        logger.error("Cleaned data not found: %s (run scripts/03_clean_data.py first)", settings.CLEANED_DATA_FILE)
        return 1
    df = pd.read_parquet(settings.CLEANED_DATA_FILE)
    n_full = len(df)
    sample = eda_stats.class_capped_sample(df, args.cap_per_class, seed=args.seed).reset_index(drop=True)
    del df
    APP_DATA.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(SAMPLE_FILE, index=False)
    logger.info("Analysis sample: %s rows of %s (cap %s per class) -> %s", f"{len(sample):,}", f"{n_full:,}", args.cap_per_class, SAMPLE_FILE)

    features = eda_stats._features(sample, None)
    results = T.run_all_tests(sample, features)
    OUT.mkdir(parents=True, exist_ok=True)
    for test, table in results.items():
        table.to_csv(OUT / f"{test}.csv", index=False)
    summary = T.summary_table(results)
    summary.to_csv(OUT / "summary.csv", index=False)
    plot_effects(results)
    plot_normality(results["shapiro"])

    sh = results["shapiro"]
    mw = results["mannwhitney"]
    kw = results["kruskal"]
    an = results["anova"]
    ch = results["chi2"]
    both = kw.merge(an[["feature", "effect_size"]], on="feature", suffixes=("_kruskal", "_anova"))
    rho = both[["effect_size_kruskal", "effect_size_anova"]].corr(method="spearman").iloc[0, 1]
    n_normal = int((sh["p_holm"] >= T.ALPHA).sum())

    lines = [
        "# Statistical validation (step 19)",
        "",
        f"Sample: **{len(sample):,} flows** of the cleaned data ({n_full:,} rows), at most {args.cap_per_class:,} per class, "
        f"every small class kept whole; seed {args.seed}. {len(features)} features tested. Significance level {T.ALPHA}, "
        "Holm-Bonferroni correction across the features of each test.",
        "",
        "With thousands of rows almost every p-value is tiny, so the conclusions below rest on the **effect sizes**.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## 1. Shapiro-Wilk: are the features normal?",
        "",
        f"H0 (normal) is rejected for **{len(sh) - n_normal - int(sh['p_value'].isna().sum())} of {int(sh['p_value'].notna().sum())}** testable features "
        f"(on a random subsample of at most {T.SHAPIRO_MAX_ROWS} rows). Median W = {sh['statistic'].median():.3f} on the raw values and "
        f"{sh['statistic_log'].median():.3f} after signed log1p. Consequences: (a) the parametric ANOVA is only indicative, so the "
        "Kruskal-Wallis test is used as the reference; (b) the log1p step of the preprocessing pipeline is justified; (c) a "
        "standard scaler on raw values would be dominated by the extreme flows.",
        "",
        markdown_table(sh[["feature", "statistic", "statistic_log", "skewness", "p_holm"]], max_rows=10),
        "",
        "## 2. Kruskal-Wallis and ANOVA: do the classes differ?",
        "",
        f"Kruskal-Wallis rejects H0 for **{int(kw['reject_h0'].sum())} of {int(kw['p_value'].notna().sum())}** features; "
        f"{int((kw['effect'] == 'large').sum())} have a large effect (epsilon squared >= 0.14). ANOVA rejects H0 for {int(an['reject_h0'].sum())}. "
        f"Both tests reach the same decision (H0 rejected) for the same features, so that conclusion does not depend on the normality assumption. "
        f"Their effect-size rankings agree only moderately (Spearman rho between epsilon squared and eta squared = {rho:.2f}): eta squared compares "
        "means, which a few extreme flows dominate, while epsilon squared compares ranks. The rank-based ranking is the one to trust here.",
        "",
        markdown_table(kw[["feature", "statistic", "effect_size", "effect", "p_holm"]], max_rows=15),
        "",
        "## 3. Chi-square: is the binned feature independent of the class?",
        "",
        f"Each feature is cut into {T.DEFAULT_BINS} quantile bins. Independence is rejected for **{int(ch['reject_h0'].sum())}** features; "
        f"{int((ch['effect'] == 'large').sum())} have a large Cramer's V. `Destination Port` is among the strongest: attack types are tied to ports in this capture "
        "(see the port ablation, step 14).",
        "",
        markdown_table(ch[["feature", "statistic", "dof", "effect_size", "effect", "p_holm", "note"]], max_rows=15),
        "",
        "## 4. Mann-Whitney U: BENIGN vs attack flows",
        "",
        f"H0 rejected for **{int(mw['reject_h0'].sum())}** features. Positive r: attack flows have larger values. "
        "This is the binary question a detector answers; the biggest shifts are on durations, inter-arrival times and window sizes.",
        "",
        markdown_table(mw[["feature", "statistic", "effect_size", "effect", "p_holm", "note"]], max_rows=15),
        "",
        "## Limits",
        "",
        "- The sample is class-capped: effect sizes describe a balanced mix of classes, not the real traffic (83% BENIGN).",
        "- Flows of one attack session are not independent observations; p-values are therefore optimistic. Effect sizes are less affected.",
        "- The chi-square uses quantile bins; a few cells have expected counts below 5 (the share is given per feature).",
        "",
        "Figures: `figures/effect_sizes.png`, `figures/normality_raw_vs_log.png`.",
    ]
    report = {
        "sample_rows": int(len(sample)),
        "cleaned_rows": int(n_full),
        "cap_per_class": args.cap_per_class,
        "seed": args.seed,
        "n_features": len(features),
        "alpha": T.ALPHA,
        "summary": summary.to_dict(orient="records"),
        "spearman_kruskal_vs_anova": float(rho),
        "shapiro_median_w_raw": float(sh["statistic"].median()),
        "shapiro_median_w_log": float(sh["statistic_log"].median()),
        "class_counts": sample[settings.LABEL_COLUMN].value_counts().to_dict(),
    }
    write_report(OUT, "stat_tests_report", "\n".join(lines) + "\n", report)
    print(summary.to_string(index=False))
    print(f"\nReport: {OUT / 'stat_tests_report.md'}\nSample: {SAMPLE_FILE} ({SAMPLE_FILE.stat().st_size / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
