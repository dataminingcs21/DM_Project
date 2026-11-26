"""The "Analyze a capture" tab: upload a .pcap / .pcapng, turn it into flows, let the models vote on every flow."""

from __future__ import annotations

import time
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import style
import ui_components as ui
from config import settings
from nids.capture.demo import build_demo_pcap
from nids.capture.pcap_flows import DEFAULT_MAX_PACKETS, PcapError, extract_flows
from nids.prediction import predictor as predictor_module

MAX_FLOWS = ui.MAX_UPLOAD_ROWS
PACKET_LIMITS = [100_000, 500_000, 1_000_000, DEFAULT_MAX_PACKETS, 5_000_000, 10_000_000, 20_000_000]
STATE = "capture::result"


def _analyse(name: str, data: bytes, predictor: predictor_module.Predictor, max_packets: int, flag_mode: str) -> dict[str, Any]:
    started = time.time()
    with st.status("Analysing the capture ...", expanded=True) as status:
        st.write(f"Reading packets from **{name}** ({len(data) / 1e6:.1f} MB) and building flows ...")
        flows = extract_flows(data, max_packets=max_packets, flag_mode=flag_mode)
        st.write(f"{flows.stats['packets_read']:,} packets -> {flows.n_flows:,} TCP/UDP flows.")
        if flows.n_flows == 0:
            status.update(label="No TCP or UDP flow found in this capture", state="error")
            return {"name": name, "flows": flows, "batch": None, "seconds": time.time() - started, "capped": False}
        features, meta, capped = flows.features, flows.meta, False
        if len(features) > MAX_FLOWS:
            features, meta, capped = features.head(MAX_FLOWS), meta.head(MAX_FLOWS), True
        st.write(f"Running the {len(predictor.models)} models on {len(features):,} flows ...")
        batch = predictor.predict_frame(features, missing="median")
        status.update(label=f"Done in {time.time() - started:.1f} s", state="complete", expanded=False)
    return {"name": name, "flows": flows, "features": features, "meta": meta, "batch": batch, "seconds": time.time() - started, "capped": capped}


def _results_frame(result: dict[str, Any]) -> pd.DataFrame:
    batch: predictor_module.BatchPrediction = result["batch"]
    frame = result["meta"].reset_index(drop=True).copy()
    votes = batch.frame.reset_index(drop=True)
    frame.insert(1, "prediction", votes["prediction"].fillna("not predicted"))
    frame.insert(2, "agreement", (votes["top_votes"].astype("float") / votes["n_voters"].astype("float")).round(3))
    frame["start_time"] = frame["start_time"].dt.tz_convert(None)
    frame["is_attack"] = ~frame["prediction"].isin([settings.BENIGN_LABEL, "not predicted"])
    return frame


def _class_chart(frame: pd.DataFrame) -> alt.Chart:
    counts = frame.groupby("prediction").size().rename("flows").reset_index()
    counts["kind"] = np.where(counts["prediction"] == settings.BENIGN_LABEL, "benign", "attack")
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("flows:Q", title="flows", scale=alt.Scale(type="symlog")),
            y=alt.Y("prediction:N", sort="-x", title=None),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["benign", "attack"], range=[style.BENIGN, style.ATTACK]), legend=None),
            tooltip=["prediction", "flows"],
        )
        .properties(height=max(90, 34 * len(counts)))
    )


def _timeline_chart(frame: pd.DataFrame) -> alt.Chart:
    data = frame[["start_time", "prediction", "dst_port", "src_ip", "dst_ip", "packets", "is_attack"]].copy()
    data["kind"] = np.where(data["is_attack"], "attack", "benign")
    return (
        alt.Chart(data)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("start_time:T", title="flow start (UTC)"),
            y=alt.Y("dst_port:Q", title="destination port", scale=alt.Scale(type="symlog")),
            size=alt.Size("packets:Q", legend=None, scale=alt.Scale(range=[18, 260])),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["benign", "attack"], range=[style.BENIGN, style.ATTACK]), title=None),
            tooltip=["start_time:T", "prediction", "src_ip", "dst_ip", "dst_port", "packets"],
        )
        .properties(height=280)
        .interactive()
    )


def _render_result(result: dict[str, Any]) -> None:
    flows = result["flows"]
    stats = flows.stats
    if result["batch"] is None:
        st.warning("This capture contains no TCP or UDP flow, so there is nothing to classify.")
        return
    frame = _results_frame(result)
    n_attack = int(frame["is_attack"].sum())
    attack_classes = frame.loc[frame["is_attack"], "prediction"].value_counts()
    if n_attack:
        top = ", ".join(f"{c} ({n:,})" for c, n in attack_classes.head(3).items())
        style.verdict(False, f"{n_attack:,} of {len(frame):,} flows match an attack class", f"Most frequent: {top}. Review the flagged flows below.")
    else:
        style.verdict(True, "No attack class detected", f"All {len(frame):,} flows were classified as benign traffic by the majority vote.")

    span = (frame["start_time"].max() - frame["start_time"].min()).total_seconds()
    style.kpi_row(
        [
            ("Packets", f"{stats['packets_read']:,}", f"{stats['packets_skipped']:,} non TCP/UDP skipped", None),
            ("Flows", f"{len(frame):,}", f"over {span:,.0f} s", None),
            ("Flagged flows", f"{n_attack:,}", f"{n_attack / len(frame):.1%} of the flows", style.ATTACK if n_attack else style.BENIGN),
            ("Hosts involved", f"{pd.concat([frame['src_ip'], frame['dst_ip']]).nunique():,}", f"analysed in {result['seconds']:.1f} s", None),
        ]
    )
    if stats.get("truncated"):
        st.info(f"Only the first {stats['packets_read']:,} packets were read (limit set in the options).")
    if result["capped"]:
        st.info(f"Only the first {MAX_FLOWS:,} flows were classified.")

    left, right = st.columns([2, 3])
    with left:
        st.markdown("##### Predicted classes")
        st.altair_chart(_class_chart(frame), width="stretch")
    with right:
        st.markdown("##### Flows over time")
        st.altair_chart(_timeline_chart(frame), width="stretch")

    if n_attack:
        st.markdown("##### Who is involved in the flagged flows")
        talkers = (
            frame[frame["is_attack"]]
            .groupby(["src_ip", "dst_ip", "prediction"])
            .agg(flows=("flow_id", "size"), ports=("dst_port", "nunique"), first_seen=("start_time", "min"), mean_agreement=("agreement", "mean"))
            .reset_index()
            .sort_values("flows", ascending=False)
        )
        st.dataframe(
            talkers, hide_index=True, width="stretch",
            column_config={"mean_agreement": st.column_config.ProgressColumn("model agreement", min_value=0.0, max_value=1.0, format="%.2f"),
                           "ports": st.column_config.NumberColumn("distinct ports")},
        )

    st.markdown("##### Every flow")
    flagged_only = st.toggle("Flagged flows only", value=bool(n_attack), key="capture::flagged_only")
    view = frame[frame["is_attack"]] if flagged_only else frame
    st.dataframe(
        view.drop(columns=["is_attack"]), hide_index=True, width="stretch", height=min(420, 36 * (len(view) + 1)),
        column_config={"agreement": st.column_config.ProgressColumn("agreement", min_value=0.0, max_value=1.0, format="%.2f"),
                       "start_time": st.column_config.DatetimeColumn("start (UTC)", format="HH:mm:ss.SSS")},
    )
    export = pd.concat([frame.drop(columns=["is_attack"]), result["features"].reset_index(drop=True)], axis=1)
    st.download_button("Download flows, features and predictions (CSV)", data=ui.to_csv_bytes(export), file_name=f"{result['name'].rsplit('.', 1)[0]}_flows.csv", mime="text/csv", key="capture::download")

    findings = result["batch"].validation
    with st.expander(f"Input findings ({sum(findings.counts.values()):,})", expanded=False):
        st.caption("Rates of zero-length flows are undefined and were filled with the training median, exactly like the infinite values of the dataset.")
        ui.render_findings(findings, show_rows=True)


def capture_tab(predictor: predictor_module.Predictor) -> None:
    style.steps(
        [
            ("1", "Upload a .pcap or .pcapng capture (Wireshark, tcpdump), or try the demo."),
            ("2", "Packets are grouped into flows and summarised with CICFlowMeter's 40 features."),
            ("3", "The 11 models vote on every flow; flagged flows are listed with their hosts."),
        ]
    )
    upload_col, demo_col = st.columns([3, 1], vertical_alignment="center")
    with upload_col:
        upload = st.file_uploader("Packet capture", type=["pcap", "pcapng", "cap"], key="capture::upload", label_visibility="collapsed")
    with demo_col:
        demo = st.button("Try the demo capture", icon=":material/science:", width="stretch", key="capture::demo")
        style.note("Synthetic: normal web and DNS traffic plus one SYN port scan.")

    with st.expander("Options", icon=":material/tune:"):
        real_flags = st.toggle("Use the real TCP flags", value=False, key="capture::real_flags")
        st.caption(
            "Off (recommended): the TCP flag columns reproduce the pattern of CIC-IDS2017, whose CICFlowMeter version filled them "
            "incorrectly (Engelen et al., 2021) - that is what the models learnt. On: the true flags, which the models may misread."
        )
        max_packets = st.select_slider(
            "Read at most this many packets", options=PACKET_LIMITS, value=DEFAULT_MAX_PACKETS,
            format_func=lambda n: f"{n / 1e6:g} M" if n >= 1_000_000 else f"{n // 1000} k", key="capture::max_packets",
        )
    flag_mode = "real" if real_flags else "cic"

    source: tuple[str, bytes] | None = None
    if demo:
        source = ("demo_capture.pcap", build_demo_pcap())
    elif upload is not None and st.button("Analyze this capture", type="primary", icon=":material/radar:", key="capture::analyze"):
        source = (upload.name, upload.getvalue())
    if source is not None:
        try:
            st.session_state[STATE] = _analyse(source[0], source[1], predictor, int(max_packets), flag_mode)
        except PcapError as error:
            st.session_state.pop(STATE, None)
            st.error(f"This file could not be read as a packet capture: {error}")
        except predictor_module.PredictorError as error:
            st.session_state.pop(STATE, None)
            st.error(f"The prediction failed: {error}")

    result = st.session_state.get(STATE)
    if result is not None:
        st.divider()
        st.markdown(f"#### Results for `{result['name']}`")
        _render_result(result)
    style.note(
        "Flows are rebuilt with a re-implementation of CICFlowMeter, and the models learnt one lab network (2017). "
        "A flagged flow is a lead to investigate, not proof of an attack; on other networks, unseen attack types are often missed."
    )
