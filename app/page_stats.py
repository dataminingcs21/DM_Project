"""Page 3 - Statistical validation: five hypothesis tests (precomputed on every feature + live on one feature)."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import analysis_data as D
import style
from config import settings
from nids.eda import stat_tests as T

LABEL = settings.LABEL_COLUMN

H0 = {
    "chi2": "the binned feature and the class are independent",
    "anova": "every class has the same mean",
    "kruskal": "every class has the same distribution (location)",
    "shapiro": "the feature is normally distributed",
    "mannwhitney": "BENIGN and attack flows have the same distribution",
}
EFFECT = {"chi2": "Cramer's V", "anova": "eta squared", "kruskal": "epsilon squared", "shapiro": "W", "mannwhitney": "rank-biserial r"}


def precomputed_section() -> None:
    summary = D.read_csv(str(D.STATS_DIR / "summary.csv"))
    report = D.read_json(str(D.STATS_DIR / "stat_tests_report.json"))
    if summary is None or report is None:
        D.missing_hint(D.STATS_DIR / "summary.csv", "19_stat_tests.py")
        return
    st.markdown(
        f"All 66 features, on a class-capped sample of **{report['sample_rows']:,} flows**. Significance level {report['alpha']}; "
        "p-values are **Holm-corrected** across the features of each test. With thousands of rows almost every p-value is tiny, "
        "so read the **effect size** columns."
    )
    st.dataframe(summary.rename(columns={"reject_h0_holm": "H0 rejected (Holm)", "large_effect": "large effect"}), hide_index=True, width="stretch")
    tabs = st.tabs([t.split(" (")[0] for t in T.TEST_TITLES.values()])
    for tab, test in zip(tabs, T.TESTS):
        with tab:
            table = D.read_csv(str(D.STATS_DIR / f"{test}.csv"))
            if table is None:
                continue
            st.markdown(f"**H0:** {H0[test]}. Effect size: {EFFECT[test]}.")
            if test == "shapiro":
                plot = table.dropna(subset=["statistic"])
                chart = alt.Chart(plot).mark_circle(size=50, color=style.ACCENT).encode(
                    x=alt.X("statistic:Q", title="W, raw values", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("statistic_log:Q", title="W after signed log1p", scale=alt.Scale(domain=[0, 1])), tooltip=["feature", "statistic", "statistic_log", "skewness"],
                ).properties(height=320, title="Normality before and after log1p (W = 1 is perfectly normal)")
                line = alt.Chart(pd.DataFrame({"a": [0, 1]})).mark_line(strokeDash=[4, 4], color="grey").encode(x="a:Q", y="a:Q")
                st.altair_chart(chart + line, width="stretch")
                st.caption(f"Median W: {report['shapiro_median_w_raw']:.3f} raw, {report['shapiro_median_w_log']:.3f} after log1p. No feature is normal, so the ANOVA is only indicative and Kruskal-Wallis is the reference test.")
            else:
                top = table.dropna(subset=["effect_size"]).head(20)
                chart = alt.Chart(top).mark_bar().encode(
                    x=alt.X("effect_size:Q", title=EFFECT[test]), y=alt.Y("feature:N", sort=None, title=None),
                    color=alt.Color("effect:N", scale=alt.Scale(domain=["large", "medium", "small", "negligible"], range=[style.ATTACK, style.WARN, style.ACCENT, style.MUTED])),
                    tooltip=["feature", "statistic", "p_holm", "effect_size", "effect"],
                ).properties(height=430, title=f"Top 20 features by {EFFECT[test]}")
                st.altair_chart(chart, width="stretch")
                if test == "anova":
                    st.caption(f"ANOVA and Kruskal-Wallis reject H0 for the same features, but their effect rankings agree only moderately (Spearman {report['spearman_kruskal_vs_anova']:.2f}): eta squared compares means, which a few extreme flows dominate. Trust the rank-based Kruskal-Wallis ranking.")
            st.dataframe(table, hide_index=True, width="stretch", height=280)


def live_section() -> None:
    df = D.load_sample()
    if df is None:
        D.missing_hint(D.SAMPLE_FILE, "19_stat_tests.py")
        return
    feats = D.feature_names(df)
    c1, c2 = st.columns([2, 1])
    feat = c1.selectbox("Feature", feats, index=feats.index("Flow IAT Mean") if "Flow IAT Mean" in feats else 0, key="st_feat")
    classes = sorted(df[LABEL].astype(str).unique())
    chosen = c2.multiselect("Classes compared (Kruskal / ANOVA / Chi-square)", classes, default=classes, key="st_classes")
    view = df[df[LABEL].astype(str).isin(chosen)] if len(chosen) >= 2 else df
    if len(chosen) < 2:
        st.caption("Fewer than two classes chosen: all classes are used.")
    rows = [T.run_test(t, view, feat) for t in T.TESTS]
    table = pd.DataFrame(rows)
    table["H0"] = [H0[t] for t in table["test"]]
    table["decision"] = np.where(table["p_value"] < T.ALPHA, "reject H0", "keep H0")
    table["effect"] = [T.interpret_effect(t, v) for t, v in zip(table["test"], table["effect_size"])]
    table.loc[table["test"] == "shapiro", "effect_size"] = table.loc[table["test"] == "shapiro", "statistic"]
    table["effect size"] = [EFFECT[t] for t in table["test"]]
    st.dataframe(table[["test", "H0", "statistic", "p_value", "decision", "effect size", "effect_size", "effect", "n", "note"]], hide_index=True, width="stretch")
    sh = rows[3]
    mw = rows[4]
    st.markdown(
        f"- **Normality:** W = {sh['statistic']:.3f} (after log1p {sh.get('statistic_log', float('nan')):.3f}), so the distribution is {'far from' if sh['statistic'] < 0.9 else 'close to'} normal.\n"
        f"- **Classes:** Kruskal-Wallis epsilon squared = {rows[2]['effect_size']:.3f} ({T.interpret_effect('kruskal', rows[2]['effect_size'])} effect).\n"
        f"- **BENIGN vs attack:** r = {mw['effect_size']:+.3f} ({T.interpret_effect('mannwhitney', mw['effect_size'])}); {mw['note']}."
    )
    data = view[[feat, LABEL]].replace([np.inf, -np.inf], np.nan).dropna()
    data["log1p"] = np.sign(data[feat]) * np.log1p(np.abs(data[feat]))
    q = np.sort(data["log1p"].sample(n=min(len(data), 2000), random_state=0).to_numpy())
    from scipy import stats as sps

    theo = sps.norm.ppf((np.arange(1, len(q) + 1) - 0.5) / len(q))
    qq = pd.DataFrame({"theoretical": theo, "sample": (q - q.mean()) / (q.std() or 1)})
    c1, c2 = st.columns(2)
    c1.altair_chart(
        alt.Chart(qq).mark_circle(size=10, color=style.ACCENT).encode(x="theoretical:Q", y="sample:Q").properties(title=f"Q-Q plot of {feat} (log1p, standardised)", height=320)
        + alt.Chart(pd.DataFrame({"a": [-4, 4]})).mark_line(color="grey", strokeDash=[4, 4]).encode(x="a:Q", y="a:Q"),
        width="stretch",
    )
    c2.altair_chart(
        alt.Chart(data).mark_boxplot(extent="min-max").encode(x=alt.X("log1p:Q", title=f"{feat} (log1p)"), y=alt.Y(f"{LABEL}:N", title=None), color=alt.Color(f"{LABEL}:N", legend=None)).properties(title="Groups compared", height=320),
        width="stretch",
    )


def render() -> None:
    st.title("Statistical validation")
    style.hero("Five formal hypothesis tests check what the exploratory figures suggest: chi-square, ANOVA, Kruskal-Wallis, Shapiro-Wilk and Mann-Whitney U.", ["5 tests", "Holm correction", "effect sizes", "66 features"])
    st.subheader("A. Every feature (step 19)")
    precomputed_section()
    st.divider()
    st.subheader("B. Run the tests on one feature")
    live_section()
    st.markdown(
        "**Conclusions used by the project.** Shapiro-Wilk: no feature is normal, which justifies the signed log1p and "
        "non-parametric tests. Kruskal-Wallis and Chi-square: almost every feature depends on the class, many with a large "
        "effect, so the supervised task is learnable. `Destination Port` ranks at the top, which links to the port ablation "
        "(step 14). Mann-Whitney: the BENIGN / attack shift is smaller than the shifts between attack types, so one binary "
        "threshold on a single feature cannot detect attacks, and multivariate models are needed. **Caution:** flows of one "
        "attack session are not independent, so the p-values are optimistic; the effect sizes are the robust part."
    )
