"""Page 4 - Model benchmark and batch inference.

A. Hermetic benchmark (step 20): baselines vs ensembles, whole pipeline cross-validated (stratified k-fold).
B. Full-data evaluation of the 11 models (steps 06-07) and the ensembles (step 13).
C. Batch inference with the saved hermetic pipeline: upload a CSV, get predictions, download them.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score

import analysis_data as D
import style
from config import settings

FAMILY_COLORS = alt.Scale(domain=["floor", "baseline", "ensemble"], range=[style.MUTED, style.WARN, style.ACCENT])
MAX_ROWS = 100_000


def hermetic_section() -> None:
    summary = D.read_csv(str(D.BENCH_DIR / "cv_summary.csv"))
    folds = D.read_csv(str(D.BENCH_DIR / "cv_folds.csv"))
    report = D.read_json(str(D.BENCH_DIR / "benchmark_report.json"))
    if summary is None or folds is None or report is None:
        D.missing_hint(D.BENCH_DIR / "cv_summary.csv", "20_pipeline_benchmark.py")
        return
    best = summary.iloc[0]
    style.kpi_row(
        [
            ("Models compared", f"{len(summary)}", "floor, baselines, ensembles", None),
            ("CV", f"{report['folds']}-fold stratified", f"{report['dev_rows']:,} rows, pipeline refitted per fold", None),
            ("Best CV macro F1", f"{best['macro_f1_mean']:.4f}", f"{best['display']} (std {best['macro_f1_std']:.4f})", style.ACCENT),
            ("Hold-out macro F1", f"{report['holdout']['macro_f1']:.4f}", f"{report['holdout_rows']:,} untouched rows", style.BENIGN),
        ]
    )
    st.write("")
    metric = st.selectbox("Metric", ["macro_f1", "balanced_accuracy", "mcc", "weighted_f1", "accuracy", "detection_rate", "false_alarm_rate", "fit_seconds"], key="b_metric")
    c1, c2 = st.columns(2)
    with c1:
        bars = alt.Chart(summary).mark_bar().encode(
            x=alt.X(f"{metric}_mean:Q", title=f"{metric} (mean over folds)"), y=alt.Y("display:N", sort="-x", title=None),
            color=alt.Color("family:N", scale=FAMILY_COLORS), tooltip=["display", "family", alt.Tooltip(f"{metric}_mean:Q", format=".4f"), alt.Tooltip(f"{metric}_std:Q", format=".4f")],
        )
        err = alt.Chart(summary).mark_errorbar(color="white").encode(x=f"{metric}_mean:Q", xError=f"{metric}_std:Q", y=alt.Y("display:N", sort="-x"))
        st.altair_chart((bars + err).properties(title="Baselines vs ensembles", height=320), width="stretch")
    with c2:
        f = folds[folds["model"] != "dummy"] if metric != "fit_seconds" else folds
        st.altair_chart(
            alt.Chart(f).mark_line(point=True).encode(x=alt.X("fold:O"), y=alt.Y(f"{metric}:Q", scale=alt.Scale(zero=False)), color=alt.Color("display:N", title="model"), tooltip=["display", "fold", alt.Tooltip(f"{metric}:Q", format=".4f")]).properties(title="Per fold (same folds for every model)", height=320),
            width="stretch",
        )
    cols = ["display", "family"] + [c for c in summary.columns if c.endswith("_mean") or c.endswith("_std")]
    st.dataframe(summary[cols].round(4), hide_index=True, width="stretch")
    paired = D.read_csv(str(D.BENCH_DIR / "paired_vs_baseline.csv"))
    if paired is not None:
        st.markdown(f"**Paired fold differences against the best baseline ({report['best_baseline'].replace('_', ' ')})**")
        st.dataframe(paired.round(4), hide_index=True, width="stretch")
    per_class = D.read_csv(str(D.BENCH_DIR / "per_class_f1_oof.csv"))
    if per_class is not None:
        long = per_class.melt(id_vars=["class", "support"], var_name="model", value_name="F1")
        st.altair_chart(
            alt.Chart(long).mark_rect().encode(
                x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=-30)), y=alt.Y("class:N", title=None),
                color=alt.Color("F1:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1])), tooltip=["class", "support", "model", alt.Tooltip("F1:Q", format=".3f")],
            ).properties(title="Error analysis: out-of-fold F1 per class", height=420),
            width="stretch",
        )
    cm_files = sorted(D.BENCH_DIR.glob("confusion_*_oof.csv"))
    if cm_files:
        cm = pd.read_csv(cm_files[0], index_col=0)
        long = cm.reset_index(names="true").melt(id_vars="true", var_name="predicted", value_name="flows")
        long["row_share"] = long["flows"] / long.groupby("true")["flows"].transform("sum").replace(0, 1)
        st.altair_chart(
            alt.Chart(long[long["flows"] > 0]).mark_rect().encode(
                x=alt.X("predicted:N", axis=alt.Axis(labelAngle=-50)), y="true:N", color=alt.Color("row_share:Q", scale=alt.Scale(scheme="blues")), tooltip=["true", "predicted", "flows", alt.Tooltip("row_share:Q", format=".2%")],
            ).properties(title=f"Out-of-fold confusion matrix of {best['display']} (colour = share of the true class)", height=460),
            width="stretch",
        )
    with st.expander("Full benchmark report (reports/benchmark/benchmark_report.md)"):
        p = D.BENCH_DIR / "benchmark_report.md"
        st.markdown(p.read_text(encoding="utf-8") if p.is_file() else "not found")


def full_data_section() -> None:
    comp = D.read_csv(str(D.EVAL_DIR / "model_comparison_tuned.csv"))
    ens = D.read_csv(str(settings.EXPERIMENTS_REPORT_DIR / "ensembles" / "ensembles_comparison.csv"))
    if comp is None:
        D.missing_hint(D.EVAL_DIR / "model_comparison_tuned.csv", "06_evaluate.py --stage tuned")
        return
    st.markdown("The 11 models of the detection page, trained on 2 million rows and scored on the 504,473 held-out flows (real class mix).")
    long = comp.melt(id_vars=["model"], value_vars=["baseline_macro_f1", "tuned_macro_f1"], var_name="version", value_name="macro F1")
    long["version"] = long["version"].str.replace("_macro_f1", "")
    st.altair_chart(
        alt.Chart(long).mark_bar().encode(
            y=alt.Y("model:N", sort="-x", title=None), x=alt.X("macro F1:Q", scale=alt.Scale(domain=[0, 1])), yOffset="version:N",
            color=alt.Color("version:N", scale=alt.Scale(range=[style.MUTED, style.ACCENT])), tooltip=["model", "version", alt.Tooltip("macro F1:Q", format=".4f")],
        ).properties(title="Test macro F1, baseline vs tuned (11 models)", height=420),
        width="stretch",
    )
    if ens is not None:
        st.markdown("**Ensembles (step 13): votes and stacking learnt from out-of-fold predictions**")
        st.dataframe(ens.round(4), hide_index=True, width="stretch")


def inference_section() -> None:
    pipe, meta = D.load_benchmark_pipeline()
    if pipe is None:
        D.missing_hint(D.BENCH_DIR / "best_pipeline.joblib", "20_pipeline_benchmark.py")
        return
    st.markdown(
        f"Model: **{meta['display']}** in its hermetic pipeline (trained on {meta['trained_rows']:,} rows). Upload a CSV of flows with the "
        f"CIC-IDS2017 column names: the pipeline selects the {len(meta['features'])} feature columns itself and handles infinite and missing "
        "values. If the file has a `Label` column, the page also scores the predictions. The 11-model vote is on the **Detection** page."
    )
    hold = D.BENCH_DIR / "holdout_flows.csv"
    if hold.is_file():
        st.download_button("Download a labelled test file (the benchmark hold-out) to try it", data=hold.read_bytes(), file_name="holdout_flows.csv", mime="text/csv")
    upload = st.file_uploader("CSV of flows", type=["csv"], key="b_upload")
    if upload is None:
        return
    try:
        df = pd.read_csv(upload, low_memory=False, nrows=MAX_ROWS)
    except (ValueError, UnicodeDecodeError) as error:
        st.error(f"The file could not be read as CSV: {error}")
        return
    df.columns = [str(c).strip() for c in df.columns]
    missing = [f for f in meta["features"] if f not in df.columns]
    if missing:
        st.error(f"{len(missing)} required column(s) are missing, e.g. {', '.join(missing[:5])}.")
        return
    X = df.copy()
    for f in meta["features"]:
        X[f] = pd.to_numeric(X[f], errors="coerce")
    with st.spinner(f"Predicting {len(X):,} flows ..."):
        codes = np.asarray(pipe.predict(X)).astype(int).ravel()
        proba = pipe.predict_proba(X).max(axis=1) if hasattr(pipe, "predict_proba") else np.full(len(X), np.nan)
    classes = np.array(meta["classes"])
    out = pd.DataFrame({"row": np.arange(1, len(df) + 1), "predicted": classes[codes], "confidence": np.round(proba, 4)})
    label = settings.LABEL_COLUMN
    if label in df.columns:
        out.insert(1, "true", df[label].astype(str).str.strip().to_numpy())
        known = out["true"].isin(classes)
        out["correct"] = out["true"] == out["predicted"]
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(out):,}")
        b.metric("Accuracy", f"{accuracy_score(out.loc[known, 'true'], out.loc[known, 'predicted']):.4f}" if known.any() else "n/a")
        c.metric("Macro F1", f"{f1_score(out.loc[known, 'true'], out.loc[known, 'predicted'], average='macro', zero_division=0):.4f}" if known.any() else "n/a")
    counts = out["predicted"].value_counts().rename_axis("class").reset_index(name="flows")
    st.altair_chart(alt.Chart(counts).mark_bar(color=style.ACCENT).encode(x="flows:Q", y=alt.Y("class:N", sort="-x"), tooltip=["class", "flows"]).properties(title="Predicted classes", height=280), width="stretch")
    st.dataframe(out, hide_index=True, width="stretch", height=320)
    st.download_button("Download the predictions as CSV", data=out.to_csv(index=False).encode("utf-8"), file_name="batch_predictions.csv", mime="text/csv", type="primary")


def render() -> None:
    st.title("Model benchmark")
    style.hero("Baselines vs ensembles under stratified cross-validation with a leak-free pipeline, the full-data scores of the 11 models, and batch inference with CSV export.", ["stratified k-fold", "ColumnTransformer + Pipeline", "XGBoost", "LightGBM", "CSV in / CSV out"])
    tab_a, tab_b, tab_c = st.tabs(["Hermetic CV benchmark", "11 models on the full test set", "Batch inference"])
    with tab_a:
        hermetic_section()
    with tab_b:
        full_data_section()
    with tab_c:
        inference_section()
