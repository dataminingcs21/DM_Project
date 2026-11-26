"""Page 2 - Exploratory data analysis: interactive figures on the full-data tables and on the analysis sample."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import analysis_data as D
import style
from config import settings

LABEL = settings.LABEL_COLUMN
BENIGN = settings.BENIGN_LABEL


def _signed_log(s: pd.Series) -> pd.Series:
    return np.sign(s) * np.log1p(np.abs(s))


def _short_day(name: str) -> str:
    return str(name).replace(".pcap_ISCX.csv", "").replace("-WorkingHours", "").replace("-workingHours", "")


def full_data_figures() -> int:
    """Figures from the tables computed on the full data (step 02). Returns the number of figures drawn."""
    n = 0
    labels = D.read_csv(str(D.EDA_TABLES / "label_distribution.csv"))
    by_file = D.read_csv(str(D.EDA_TABLES / "label_by_file.csv"))
    desc = D.read_csv(str(D.EDA_TABLES / "descriptive_statistics.csv"))
    outliers = D.read_csv(str(D.EDA_TABLES / "outlier_summary.csv"))
    sep = D.read_csv(str(D.EDA_TABLES / "feature_class_separation.csv"))
    corr_pairs = D.read_csv(str(D.EDA_TABLES / "high_correlation_pairs.csv"))
    if labels is None:
        D.missing_hint(D.EDA_TABLES / "label_distribution.csv", "02_run_eda.py")
        return 0

    c1, c2 = st.columns(2)
    with c1:
        log = st.toggle("Log scale", value=True, key="eda_log_classes")
        chart = alt.Chart(labels).mark_bar().encode(
            x=alt.X("count:Q", scale=alt.Scale(type="log") if log else alt.Scale(), title="flows"),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.condition(alt.datum.label == BENIGN, alt.value(style.BENIGN), alt.value(style.ATTACK)),
            tooltip=["label", "count", alt.Tooltip("percent:Q", format=".3f")],
        ).properties(title="Fig. 1 - Class distribution (raw data): BENIGN is 80%", height=360)
        st.altair_chart(chart, width="stretch")
        n += 1
    with c2:
        if by_file is not None:
            long = by_file.melt(id_vars=by_file.columns[0], var_name="file", value_name="flows")
            long = long[(long["flows"] > 0) & (long[by_file.columns[0]] != BENIGN)]
            long["day"] = long["file"].map(_short_day)
            chart = alt.Chart(long).mark_bar().encode(
                x=alt.X("flows:Q", stack="normalize", title="share of the attack flows of the file"),
                y=alt.Y("day:N", title=None),
                color=alt.Color(f"{by_file.columns[0]}:N", title="attack"),
                tooltip=["day", by_file.columns[0], "flows"],
            ).properties(title="Fig. 2 - Attacks per capture file: each attack type lives in one file", height=360)
            st.altair_chart(chart, width="stretch")
            n += 1

    c1, c2 = st.columns(2)
    with c1:
        if desc is not None:
            top = desc.assign(abs_skew=desc["skewness"].abs()).nlargest(20, "abs_skew")
            chart = alt.Chart(top).mark_bar(color=style.ACCENT).encode(
                x=alt.X("skewness:Q"), y=alt.Y("column:N", sort="-x", title=None), tooltip=["column", "mean", "median", "max", alt.Tooltip("skewness:Q", format=".1f")]
            ).properties(title="Fig. 3 - The 20 most skewed features (hence the log1p step)", height=380)
            st.altair_chart(chart, width="stretch")
            n += 1
    with c2:
        if outliers is not None:
            top = outliers.nlargest(20, "outlier_percent")
            chart = alt.Chart(top).mark_bar(color=style.WARN).encode(
                x=alt.X("outlier_percent:Q", title="% of rows outside 1.5 IQR"), y=alt.Y("column:N", sort="-x", title=None), tooltip=["column", "n_outliers", "outlier_percent"]
            ).properties(title="Fig. 4 - IQR outliers per feature (kept: attacks ARE the outliers)", height=380)
            st.altair_chart(chart, width="stretch")
            n += 1

    c1, c2 = st.columns(2)
    with c1:
        if sep is not None:
            top = sep.dropna().nlargest(20, "f_statistic")
            chart = alt.Chart(top).mark_bar(color=style.BENIGN).encode(
                x=alt.X("f_statistic:Q", title="ANOVA F (class-capped sample)"), y=alt.Y("column:N", sort="-x", title=None), tooltip=["column", "f_statistic"]
            ).properties(title="Fig. 5 - Features that separate the classes best", height=380)
            st.altair_chart(chart, width="stretch")
            n += 1
    with c2:
        if corr_pairs is not None:
            thr = st.slider("Correlation threshold", 0.80, 1.0, float(settings.CORRELATION_THRESHOLD), 0.01, key="eda_corr_thr")
            pairs = corr_pairs[corr_pairs["correlation"].abs() >= thr]
            hist = alt.Chart(corr_pairs).mark_bar(color=style.MUTED).encode(
                x=alt.X("correlation:Q", bin=alt.Bin(step=0.01), title="|r| of the highly correlated pairs"), y="count()"
            ).properties(title=f"Fig. 6 - Redundancy: {len(pairs)} pairs with |r| >= {thr:.2f}", height=380)
            st.altair_chart(hist, width="stretch")
            n += 1
    return n


def sample_figures(df: pd.DataFrame) -> int:
    n = 0
    feats = D.feature_names(df)
    classes = sorted(df[LABEL].astype(str).unique())
    st.caption(f"Analysis sample: {len(df):,} flows, at most 2,000 per class (rare classes whole), so the rare attacks are visible.")
    pick = st.multiselect("Classes shown", classes, default=classes, key="eda_classes")
    view = df[df[LABEL].astype(str).isin(pick)] if pick else df
    default = "Flow Duration" if "Flow Duration" in feats else feats[0]

    c1, c2 = st.columns(2)
    with c1:
        feat = st.selectbox("Feature", feats, index=feats.index(default), key="eda_feat")
        logx = st.toggle("signed log1p", value=True, key="eda_feat_log")
        data = view[[feat, LABEL]].replace([np.inf, -np.inf], np.nan).dropna()
        data["value"] = _signed_log(data[feat]) if logx else data[feat]
        data["group"] = np.where(data[LABEL] == BENIGN, "BENIGN", "attack")
        hist = alt.Chart(data).mark_bar(opacity=0.6, binSpacing=0).encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=60), title=f"{feat}{' (log1p)' if logx else ''}"),
            y=alt.Y("count()", stack=None, title="flows"),
            color=alt.Color("group:N", scale=alt.Scale(domain=["BENIGN", "attack"], range=[style.BENIGN, style.ATTACK])),
        ).properties(title=f"Fig. 7 - Distribution of {feat}: BENIGN vs attack", height=330)
        st.altair_chart(hist, width="stretch")
        n += 1
    with c2:
        box = alt.Chart(data).mark_boxplot(extent="min-max", size=12).encode(
            x=alt.X("value:Q", title=f"{feat}{' (log1p)' if logx else ''}"), y=alt.Y(f"{LABEL}:N", title=None), color=alt.Color(f"{LABEL}:N", legend=None)
        ).properties(title=f"Fig. 8 - {feat} per class (box plot)", height=400)
        st.altair_chart(box, width="stretch")
        n += 1

    c1, c2 = st.columns(2)
    with c1:
        fx = st.selectbox("X axis", feats, index=feats.index("Destination Port") if "Destination Port" in feats else 0, key="eda_x")
        fy = st.selectbox("Y axis", feats, index=feats.index("Init_Win_bytes_forward") if "Init_Win_bytes_forward" in feats else 1, key="eda_y")
        pts = view[[fx, fy, LABEL]].replace([np.inf, -np.inf], np.nan).dropna()
        pts = pts.sample(n=min(len(pts), 4000), random_state=0)
        pts["x"], pts["y"] = _signed_log(pts[fx]), _signed_log(pts[fy])
        sel = alt.selection_point(fields=[LABEL], bind="legend")
        scatter = alt.Chart(pts).mark_circle(size=14).encode(
            x=alt.X("x:Q", title=f"{fx} (log1p)"), y=alt.Y("y:Q", title=f"{fy} (log1p)"),
            color=alt.Color(f"{LABEL}:N"), opacity=alt.condition(sel, alt.value(0.75), alt.value(0.05)), tooltip=[LABEL, fx, fy],
        ).add_params(sel).interactive().properties(title="Fig. 9 - Two features at once (click the legend to isolate a class)", height=420)
        st.altair_chart(scatter, width="stretch")
        n += 1
    with c2:
        k = st.slider("Features in the heatmap", 8, min(40, len(feats)), 20, key="eda_k")
        method = st.radio("Correlation", ["spearman", "pearson"], horizontal=True, key="eda_corr_method")
        top = view[feats].replace([np.inf, -np.inf], np.nan).var().nlargest(k).index.tolist() if method == "pearson" else feats[:k]
        corr = view[top].replace([np.inf, -np.inf], np.nan).corr(method=method).round(2)
        long = corr.reset_index(names="a").melt(id_vars="a", var_name="b", value_name="r")
        heat = alt.Chart(long).mark_rect().encode(
            x=alt.X("a:N", sort=top, title=None, axis=alt.Axis(labelAngle=-60, labelLimit=120)), y=alt.Y("b:N", sort=top, title=None, axis=alt.Axis(labelLimit=120)),
            color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True)), tooltip=["a", "b", "r"],
        ).properties(title=f"Fig. 10 - {method.title()} correlation of {k} features", height=470)
        st.altair_chart(heat, width="stretch")
        n += 1

    c1, c2 = st.columns(2)
    with c1:
        ports = view.assign(port=view["Destination Port"].astype(int).astype(str)).groupby(["port", LABEL]).size().reset_index(name="flows")
        top_ports = ports.groupby("port")["flows"].sum().nlargest(15).index
        chart = alt.Chart(ports[ports["port"].isin(top_ports)]).mark_bar().encode(
            x=alt.X("flows:Q"), y=alt.Y("port:N", sort="-x", title="destination port"), color=alt.Color(f"{LABEL}:N"), tooltip=["port", LABEL, "flows"]
        ).properties(title="Fig. 11 - Top destination ports and their classes (port 80 = most attacks)", height=380)
        st.altair_chart(chart, width="stretch")
        n += 1
    with c2:
        prof_feats = st.multiselect("Features of the class profile", feats, default=[f for f in ("Flow Duration", "Total Fwd Packets", "Packet Length Mean", "Flow IAT Mean", "Init_Win_bytes_forward", "Destination Port") if f in feats], key="eda_prof")
        if prof_feats:
            med = view.groupby(LABEL)[prof_feats].median()
            z = _signed_log(med)
            z = (z - z.mean()) / z.std(ddof=0).replace(0, 1)
            long = z.reset_index().melt(id_vars=LABEL, var_name="feature", value_name="z")
            heat = alt.Chart(long).mark_rect().encode(
                x=alt.X("feature:N", title=None, axis=alt.Axis(labelAngle=-30)), y=alt.Y(f"{LABEL}:N", title=None),
                color=alt.Color("z:Q", scale=alt.Scale(scheme="purpleorange", domainMid=0), title="z (log median)"), tooltip=[LABEL, "feature", alt.Tooltip("z:Q", format=".2f")],
            ).properties(title="Fig. 12 - Class profile: median per class, standardised", height=380)
            st.altair_chart(heat, width="stretch")
            n += 1

    if settings.SOURCE_FILE_COLUMN in view.columns:
        days = view.assign(day=view[settings.SOURCE_FILE_COLUMN].map(_short_day)).groupby(["day", LABEL]).size().reset_index(name="flows")
        chart = alt.Chart(days).mark_rect().encode(
            x=alt.X("day:N", title=None, axis=alt.Axis(labelAngle=-20)), y=alt.Y(f"{LABEL}:N", title=None),
            color=alt.Color("flows:Q", scale=alt.Scale(type="log", scheme="blues")), tooltip=["day", LABEL, "flows"],
        ).properties(title="Fig. 13 - Class x capture day (sample): why a split by day is much harder than a random split", height=330)
        st.altair_chart(chart, width="stretch")
        n += 1
    return n


def render() -> None:
    st.title("Exploratory analysis")
    style.hero("Interactive figures: hover for values, change features and classes. Part A uses tables computed on all 2.8 million raw rows; part B uses the class-capped analysis sample.", ["13 figures", "interactive", "distributions", "correlations", "class profiles"])
    st.subheader("A. Full data (step 02)")
    n = full_data_figures()
    st.divider()
    st.subheader("B. Analysis sample (interactive)")
    df = D.load_sample()
    if df is None:
        D.missing_hint(D.SAMPLE_FILE, "19_stat_tests.py")
    else:
        n += sample_figures(df)
    st.caption(f"{n} figures on this page. Static versions of the step-02 figures are in reports/eda/figures/.")
    st.markdown(
        "**What the figures show.** (1) Strong imbalance: BENIGN 80%, three classes under 40 flows. "
        "(2) Each attack type comes from a single capture file. (3-4) Heavy right skew and many IQR outliers, which motivates the "
        "signed log1p and the median imputation, and outliers are kept because the attacks are outliers. (5-6) Many near-duplicate "
        "features (|r| >= 0.95) are removed by the correlation filter. (7-12) Classes differ strongly on ports, window sizes, "
        "durations and inter-arrival times, which the hypothesis tests on the next page confirm."
    )
