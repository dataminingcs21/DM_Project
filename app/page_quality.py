"""Page 1 - Data quality: duplicates, missing and infinite values, types, impossible values, class balance."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import analysis_data as D
import style
from config import settings
from nids.data import quality


def raw_dataset_section() -> None:
    st.subheader("1. The raw CIC-IDS2017 files (inspection, step 01)")
    summary = D.read_json(str(D.INSPECTION / "inspection_summary.json"))
    profile = D.read_csv(str(D.INSPECTION / "column_profile.csv"))
    labels = D.read_csv(str(D.INSPECTION / "label_distribution.csv"))
    if summary is None or profile is None:
        D.missing_hint(D.INSPECTION / "inspection_summary.json", "01_inspect_dataset.py")
        return
    dup = summary.get("duplicates", {})
    val = summary.get("values", {})
    style.kpi_row(
        [
            ("Rows", f"{summary['n_rows']:,}", f"{summary['n_source_files']} CSV files", None),
            ("Feature columns", f"{summary['n_feature_columns']}", f"{summary['memory_mb']:,.0f} MB in memory", None),
            ("Duplicate rows", f"{dup.get('n_duplicate_rows', 0):,}", f"{100 * dup.get('n_duplicate_rows', 0) / summary['n_rows']:.2f}% of the rows", style.WARN),
            ("Missing / infinite", f"{val.get('n_missing', 0):,} / {val.get('n_infinite', 0):,}", f"{val.get('rows_with_missing_or_infinite', 0):,} rows affected", style.ATTACK),
        ]
    )
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Columns with a problem**")
        issues = profile[(profile["n_missing"] > 0) | (profile["n_inf"] > 0) | (profile["n_negative"] > 0) | profile["is_constant"]]
        st.dataframe(issues, hide_index=True, width="stretch", height=320)
        st.caption(
            f"{len(issues)} of {len(profile)} columns. Constant columns: {', '.join(val.get('constant_columns', [])[:8]) or 'none'}. "
            f"{dup.get('n_conflicting_feature_vectors', 0):,} feature vectors appear with two different labels."
        )
    with c2:
        st.markdown("**Data types**")
        types = profile["dtype"].value_counts().rename_axis("dtype").reset_index(name="columns")
        st.altair_chart(alt.Chart(types).mark_bar().encode(x="columns:Q", y=alt.Y("dtype:N", sort="-x"), tooltip=["dtype", "columns"]).properties(height=140), width="stretch")
        if labels is not None:
            st.markdown("**Labels with an odd character**")
            odd = summary.get("labels", {}).get("labels_with_odd_characters", [])
            st.write(", ".join(f"`{x}`" for x in odd) or "none")
    st.markdown("**What the cleaning (step 03) did about it**")
    st.markdown(
        "- 308,381 exact duplicate rows (10.89%) removed; 12 identical copy columns dropped.\n"
        "- The three Web Attack labels with a damaged character renamed.\n"
        "- Impossible negatives (durations, rates, header lengths) and infinite rates set to **missing**; "
        "they are filled later by the median *learned on the training rows only* (inside the preprocessing pipeline).\n"
        "- The 698 conflicting feature vectors are kept: they are a real, unavoidable error floor (0.055% of the rows).\n"
        "- Result: 2,522,362 rows, 66 feature columns, 15 classes."
    )
    report = settings.CLEANING_REPORT_FILE
    if report.is_file():
        with st.expander("Full cleaning report (data/cleaned/cleaning_report.json)"):
            st.json(D.read_json(str(report)))


def interactive_section() -> None:
    st.subheader("2. Check a table yourself")
    st.markdown("Run the same checks on the analysis sample or on any CSV of flows (for example before uploading it on the detection page).")
    source = st.radio("Table to check", ["Analysis sample (cleaned data)", "Upload a CSV file"], horizontal=True, key="q_source")
    if source.startswith("Upload"):
        upload = st.file_uploader("CSV file", type=["csv"], key="q_upload")
        if upload is None:
            return
        try:
            df = pd.read_csv(upload, low_memory=False, nrows=200_000)
        except (ValueError, UnicodeDecodeError) as error:
            st.error(f"The file could not be read as CSV: {error}")
            return
        df.columns = [str(c).strip() for c in df.columns]
    else:
        df = D.load_sample()
        if df is None:
            D.missing_hint(D.SAMPLE_FILE, "19_stat_tests.py")
            return
    rep = quality.quality_report(df)
    dup = rep["duplicates"]
    style.kpi_row(
        [
            ("Rows x columns", f"{rep['n_rows']:,} x {rep['n_columns']}", f"{rep['n_numeric']} numeric", None),
            ("Duplicate rows", f"{dup['n_duplicate_rows']:,}", f"{dup['pct_duplicate_rows']:.2f}%", style.WARN if dup["n_duplicate_rows"] else style.BENIGN),
            ("Missing cells", f"{int(rep['columns']['n_missing'].sum()):,}", f"{int((rep['columns']['n_missing'] > 0).sum())} column(s)", None),
            ("Infinite values", f"{int(rep['columns']['n_inf'].sum()):,}", f"{int((rep['columns']['n_inf'] > 0).sum())} column(s)", None),
        ]
    )
    st.write("")
    if rep["issues"]:
        st.warning("**Findings:**\n" + "\n".join(f"- {i}" for i in rep["issues"]))
    else:
        st.success("No quality problem found.")
    cols = rep["columns"]
    c1, c2 = st.columns([3, 2])
    with c1:
        only = st.toggle("Only columns with a finding", value=True, key="q_only")
        view = cols[(cols["n_missing"] > 0) | (cols["n_inf"] > 0) | ((cols["n_negative"] > 0) & cols["negative_impossible"]) | cols["is_constant"] | (cols["kind"] != "numeric")] if only else cols
        st.dataframe(view, hide_index=True, width="stretch", height=360)
    with c2:
        bad = cols[(cols["n_missing"] + cols["n_inf"]) > 0].melt(id_vars="column", value_vars=["n_missing", "n_inf"], var_name="kind", value_name="cells")
        bad = bad[bad["cells"] > 0]
        if len(bad):
            st.altair_chart(
                alt.Chart(bad).mark_bar().encode(x="cells:Q", y=alt.Y("column:N", sort="-x"), color="kind:N", tooltip=["column", "kind", "cells"]).properties(title="Missing and infinite cells", height=260),
                width="stretch",
            )
        bal = rep["balance"]
        if len(bal):
            st.altair_chart(
                alt.Chart(bal).mark_bar().encode(x=alt.X("rows:Q", scale=alt.Scale(type="log")), y=alt.Y("class:N", sort="-x"), tooltip=["class", "rows", alt.Tooltip("share_pct:Q", format=".2f")]).properties(title="Class balance (log scale)", height=300),
                width="stretch",
            )
    if source.startswith("Analysis"):
        st.caption("The sample is drawn from the cleaned data, so duplicates and infinite values are already gone; the remaining missing cells are the impossible values the cleaning turned into missing. Upload a raw CIC-IDS2017 CSV to see the raw problems.")


def render() -> None:
    st.title("Data quality")
    style.hero("Duplicates, missing and infinite values, types, impossible values and class balance: what was wrong with the raw data and how it was handled.", ["inspection", "cleaning", "duplicates", "missing values", "types"])
    raw_dataset_section()
    st.divider()
    interactive_section()
