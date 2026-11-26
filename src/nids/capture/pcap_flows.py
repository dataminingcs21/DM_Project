"""Packet capture -> network flows -> the 40 CICFlowMeter features used by the models.

CIC-IDS2017 was not distributed as packets but as *flows* computed by CICFlowMeter.  To classify a capture
of your own, the packets must be grouped into flows and summarised the same way.  This module follows
CICFlowMeter's definitions as closely as is practical:

* A **flow** is bidirectional, keyed by protocol and the two (address, port) endpoints.  The *forward*
  direction is the direction of the flow's first packet; its destination port is ``Destination Port``.
* A flow ends on a TCP **FIN or RST** (that packet included), or when it has lasted **120 s** (flow
  timeout); the next packet of the same endpoints starts a new flow.
* **Packet length** = transport payload length (not the frame size), *plus the Ethernet padding of
  minimum-size frames*, which CICFlowMeter counted as payload; **header length** = TCP header
  (data offset x 4) or 8 for UDP.
* **TCP flag columns: two modes.**  The CICFlowMeter version used for CIC-IDS2017 filled several flag columns
  incorrectly (Engelen et al., "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study",
  IEEE S&P Workshops 2021).  In the cleaned data, ``PSH Flag Count`` and ``ACK Flag Count`` are 0/1 and almost
  mutually exclusive (both 1 in 1,330 of 1.6 million TCP flows); ``PSH`` is 1 for every PortScan flow (which
  starts with a SYN) and for 96% of complete HTTPS sessions; ``FIN``, ``RST``, ``Fwd PSH`` and ``Fwd URG`` are
  almost always 0.  ``flag_mode="cic"`` (default) reproduces that pattern - ``PSH`` = the flow starts with a
  SYN, ``ACK`` = its complement, the other flag columns 0 - because it is what the models learnt.  It is an
  empirical approximation of the bug, not its exact rule.  ``flag_mode="real"`` gives the true flags.
* Times are in **microseconds**; standard deviations are sample deviations (n - 1), 0 for one value.
* **Active / idle** periods use an activity timeout of **5 s**.
* Flag counts are 1 when at least one packet of the flow carries the flag (the dataset stores 0/1).
* Rates (bytes/s, packets/s) are undefined for a zero-length flow and left missing, exactly like the
  infinite values of the dataset that the cleaning step turned into missing values.

Only IPv4/IPv6 TCP and UDP packets are used; everything else (ARP, ICMP, ...) is counted and skipped.

**Limit:** this is a re-implementation, not CICFlowMeter itself.  Small differences in edge cases
(flow termination, retransmissions) are expected, and the models were trained on one lab network in 2017:
predictions on other traffic are indicative, not proof.
"""

from __future__ import annotations

import io
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import dpkt
import numpy as np
import pandas as pd

from nids.utils.logger import get_logger

logger = get_logger("capture.pcap_flows")

FLOW_TIMEOUT_S = 120.0
ACTIVITY_TIMEOUT_S = 5.0
DEFAULT_MAX_PACKETS = 3_000_000

FIN, SYN, RST, PSH, ACK, URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20

FEATURES: tuple[str, ...] = (
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Min", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow IAT Std", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Std", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Std", "Bwd IAT Min", "Fwd PSH Flags", "Fwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Bwd Packets/s", "Max Packet Length", "FIN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "Down/Up Ratio", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
    "Idle Mean", "Idle Std",
)
META_COLUMNS: tuple[str, ...] = ("flow_id", "start_time", "protocol", "src_ip", "src_port", "dst_ip", "dst_port", "packets", "bytes")


class PcapError(ValueError):
    """The file is not a readable packet capture."""


# ---------------------------------------------------------------------------
# Packets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Packet:
    ts: float
    proto: str          # "TCP" or "UDP"
    src: str
    sport: int
    dst: str
    dport: int
    payload: int        # transport payload bytes
    header: int         # transport header bytes
    flags: int = 0      # TCP flags
    window: int = -1    # TCP window (-1 for UDP)


def _ip_text(raw: bytes) -> str:
    return socket.inet_ntop(socket.AF_INET6 if len(raw) == 16 else socket.AF_INET, raw)


def _network_layer(buf: bytes, datalink: int) -> tuple[Any, int]:
    """The IP / IPv6 object of one frame (or None) and the Ethernet padding bytes after it."""
    if datalink == dpkt.pcap.DLT_EN10MB:
        frame = dpkt.ethernet.Ethernet(buf)
        # A minimum-size Ethernet frame (60 bytes) is padded with zeros after the IP packet.  CICFlowMeter counted
        # that padding as payload (a bare RST shows 6 bytes in CIC-IDS2017), so it is measured here to match.
        padding = max(len(buf) - len(bytes(frame)), 0) if len(buf) <= 64 else 0
        return frame.data, padding
    if datalink == getattr(dpkt.pcap, "DLT_LINUX_SLL", 113):
        return dpkt.sll.SLL(buf).data, 0
    if datalink in (dpkt.pcap.DLT_NULL, getattr(dpkt.pcap, "DLT_LOOP", 108)):
        return dpkt.loopback.Loopback(buf).data, 0
    if datalink in (getattr(dpkt.pcap, "DLT_RAW", 12), 14, 101, 228, 229):
        if not buf:
            return None, 0
        return (dpkt.ip.IP(buf) if buf[0] >> 4 == 4 else dpkt.ip6.IP6(buf)), 0
    raise PcapError(f"unsupported link type {datalink} (supported: Ethernet, Linux cooked, loopback, raw IP)")


def parse_packet(ts: float, buf: bytes, datalink: int) -> Packet | None:
    """One TCP/UDP packet, or None for anything else (or a damaged frame)."""
    try:
        ip, padding = _network_layer(buf, datalink)
    except (dpkt.UnpackError, IndexError, ValueError):
        return None
    if isinstance(ip, dpkt.ip.IP):
        ip_payload_len = max(ip.len - ip.hl * 4, 0)
    elif isinstance(ip, dpkt.ip6.IP6):
        ip_payload_len = ip.plen  # extension headers (rare) are counted as payload
    else:
        return None
    segment = ip.data
    if isinstance(segment, dpkt.tcp.TCP):
        header = segment.off * 4
        payload = (max(ip_payload_len - header, 0) if ip_payload_len else len(segment.data)) + padding
        return Packet(ts, "TCP", _ip_text(ip.src), segment.sport, _ip_text(ip.dst), segment.dport, payload, header, segment.flags, segment.win)
    if isinstance(segment, dpkt.udp.UDP):
        return Packet(ts, "UDP", _ip_text(ip.src), segment.sport, _ip_text(ip.dst), segment.dport, max(segment.ulen - 8, 0) + padding, 8)
    return None


def _reader(stream: BinaryIO) -> Any:
    head = stream.read(4)
    stream.seek(0)
    if len(head) < 4:
        raise PcapError("the file is empty or too short to be a capture")
    try:
        if head == b"\x0a\x0d\x0d\x0a":
            return dpkt.pcapng.Reader(stream)
        return dpkt.pcap.Reader(stream)
    except (ValueError, dpkt.UnpackError) as exc:
        raise PcapError(f"not a pcap or pcapng file ({exc})") from exc


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------
@dataclass
class _Flow:
    proto: str
    src: str
    sport: int
    dst: str
    dport: int
    packets: list[Packet] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.packets[0].ts


def _std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def _min(values: np.ndarray) -> float:
    return float(values.min()) if values.size else 0.0


def _active_idle(times: np.ndarray) -> tuple[list[float], list[float]]:
    """Active and idle periods (microseconds) with CICFlowMeter's activity timeout."""
    timeout = ACTIVITY_TIMEOUT_S * 1e6
    active, idle = [], []
    start = end = times[0]
    for t in times[1:]:
        if t - end > timeout:
            if end - start > 0:
                active.append(end - start)
            idle.append(t - end)
            start = end = t
        else:
            end = t
    if end - start > 0:
        active.append(end - start)
    return active, idle


FLAG_MODES = ("cic", "real")


def _cic_flags(flow: _Flow) -> dict[str, int]:
    """Flag columns as they appear in CIC-IDS2017 (see the module docstring)."""
    tcp = flow.proto == "TCP"
    first = flow.packets[0].flags
    starts_with_syn = int(tcp and bool(first & SYN) and not first & ACK)
    return {
        "Fwd PSH Flags": 0, "Fwd URG Flags": 0, "FIN Flag Count": 0, "RST Flag Count": 0, "URG Flag Count": 0,
        "PSH Flag Count": starts_with_syn, "ACK Flag Count": int(tcp) - starts_with_syn,
    }


def flow_features(flow: _Flow, flag_mode: str = "cic") -> dict[str, float]:
    """The 40 model features of one finished flow."""
    if flag_mode not in FLAG_MODES:
        raise ValueError(f"flag_mode must be one of {FLAG_MODES}, got {flag_mode!r}")
    pk = flow.packets
    ts = np.array([round(p.ts * 1e6) for p in pk], dtype="int64")  # whole microseconds, like CICFlowMeter
    fwd = np.array([(p.src, p.sport) == (flow.src, flow.sport) for p in pk])
    payload = np.array([p.payload for p in pk], dtype="float64")
    header = np.array([p.header for p in pk], dtype="float64")
    flags = np.array([p.flags for p in pk], dtype="int64")
    f_len, b_len = payload[fwd], payload[~fwd]
    f_ts, b_ts = ts[fwd], ts[~fwd]
    duration_us = int(ts[-1] - ts[0])
    duration_s = duration_us / 1e6
    iat, f_iat, b_iat = (np.diff(t).astype("float64") for t in (ts, f_ts, b_ts))
    active, idle = _active_idle(ts)
    active_us, idle_us = np.array(active, dtype="float64"), np.array(idle, dtype="float64")
    n_fwd, n_bwd = int(fwd.sum()), int((~fwd).sum())
    first_bwd = next((p for p in pk if (p.src, p.sport) != (flow.src, flow.sport)), None)

    def any_flag(mask: np.ndarray, bit: int) -> int:
        return int(bool((flags[mask] & bit).any()))

    every = np.ones(len(pk), dtype=bool)
    features = {
        "Destination Port": flow.dport,
        "Flow Duration": float(duration_us),
        "Total Fwd Packets": n_fwd,
        "Total Backward Packets": n_bwd,
        "Total Length of Fwd Packets": float(f_len.sum()),
        "Total Length of Bwd Packets": float(b_len.sum()),
        "Fwd Packet Length Min": _min(f_len),
        "Fwd Packet Length Mean": float(f_len.mean()) if f_len.size else 0.0,
        "Fwd Packet Length Std": _std(f_len),
        "Bwd Packet Length Min": _min(b_len),
        "Bwd Packet Length Std": _std(b_len),
        "Flow Bytes/s": float(payload.sum() / duration_s) if duration_s > 0 else np.nan,
        "Flow IAT Std": _std(iat),
        "Flow IAT Min": _min(iat),
        "Fwd IAT Total": float(f_iat.sum()),
        "Fwd IAT Std": _std(f_iat),
        "Fwd IAT Min": _min(f_iat),
        "Bwd IAT Total": float(b_iat.sum()),
        "Bwd IAT Std": _std(b_iat),
        "Bwd IAT Min": _min(b_iat),
        "Fwd PSH Flags": any_flag(fwd, PSH),
        "Fwd URG Flags": any_flag(fwd, URG),
        "Fwd Header Length": float(header[fwd].sum()),
        "Bwd Header Length": float(header[~fwd].sum()),
        "Bwd Packets/s": n_bwd / duration_s if duration_s > 0 else np.nan,
        "Max Packet Length": float(payload.max()),
        "FIN Flag Count": any_flag(every, FIN),
        "RST Flag Count": any_flag(every, RST),
        "PSH Flag Count": any_flag(every, PSH),
        "ACK Flag Count": any_flag(every, ACK),
        "URG Flag Count": any_flag(every, URG),
        "Down/Up Ratio": float(n_bwd // n_fwd) if n_fwd else 0.0,
        "Init_Win_bytes_forward": pk[0].window if flow.proto == "TCP" else -1,
        "Init_Win_bytes_backward": first_bwd.window if (first_bwd is not None and flow.proto == "TCP") else -1,
        "act_data_pkt_fwd": int((f_len > 0).sum()),
        "min_seg_size_forward": _min(header[fwd]),
        "Active Mean": float(active_us.mean()) if active_us.size else 0.0,
        "Active Std": _std(active_us),
        "Idle Mean": float(idle_us.mean()) if idle_us.size else 0.0,
        "Idle Std": _std(idle_us),
    }
    if flag_mode == "cic":
        features.update(_cic_flags(flow))
    return features


def build_flows(packets: Iterator[Packet]) -> Iterator[_Flow]:
    """Group packets into flows (FIN/RST termination, 120 s flow timeout). Yields finished flows."""
    active: dict[tuple, _Flow] = {}
    for p in packets:
        a, b = (p.src, p.sport), (p.dst, p.dport)
        key = (p.proto, *sorted((a, b)))
        flow = active.get(key)
        if flow is not None and p.ts - flow.start > FLOW_TIMEOUT_S:
            yield active.pop(key)
            flow = None
        if flow is None:
            flow = active[key] = _Flow(p.proto, p.src, p.sport, p.dst, p.dport)
        flow.packets.append(p)
        if p.proto == "TCP" and p.flags & (FIN | RST):
            yield active.pop(key)
    yield from active.values()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
@dataclass
class CaptureFlows:
    features: pd.DataFrame          # one row per flow, the 40 model features
    meta: pd.DataFrame              # one row per flow: endpoints, protocol, start time, size
    stats: dict[str, Any]           # packets read / used / skipped, flows, time span, truncation

    @property
    def n_flows(self) -> int:
        return int(len(self.features))


def extract_flows(
    source: str | Path | bytes | BinaryIO, max_packets: int | None = DEFAULT_MAX_PACKETS, flag_mode: str = "cic"
) -> CaptureFlows:
    """Read a .pcap/.pcapng (path, bytes or binary file object) and return its flows and their features."""
    if isinstance(source, (str, Path)):
        stream: BinaryIO = open(source, "rb")
    elif isinstance(source, (bytes, bytearray)):
        stream = io.BytesIO(source)
    else:
        stream = source
    if flag_mode not in FLAG_MODES:
        raise ValueError(f"flag_mode must be one of {FLAG_MODES}, got {flag_mode!r}")
    stats = {"packets_read": 0, "packets_used": 0, "packets_skipped": 0, "truncated": False, "flag_mode": flag_mode}
    try:
        reader = _reader(stream)
        datalink = reader.datalink()

        def packets() -> Iterator[Packet]:
            for ts, buf in reader:
                if max_packets is not None and stats["packets_read"] >= max_packets:
                    stats["truncated"] = True
                    break
                stats["packets_read"] += 1
                p = parse_packet(float(ts), buf, datalink)
                if p is None:
                    stats["packets_skipped"] += 1
                    continue
                stats["packets_used"] += 1
                yield p

        rows, meta = [], []
        for i, flow in enumerate(build_flows(packets())):
            rows.append(flow_features(flow, flag_mode))
            meta.append(
                {
                    "flow_id": i,
                    "start_time": datetime.fromtimestamp(flow.start, tz=timezone.utc),
                    "protocol": flow.proto,
                    "src_ip": flow.src,
                    "src_port": flow.sport,
                    "dst_ip": flow.dst,
                    "dst_port": flow.dport,
                    "packets": len(flow.packets),
                    "bytes": int(sum(p.payload for p in flow.packets)),
                }
            )
    except dpkt.NeedData as exc:
        raise PcapError(f"the capture ends in the middle of a packet ({exc})") from exc
    finally:
        if isinstance(source, (str, Path)):
            stream.close()
    features = pd.DataFrame(rows, columns=list(FEATURES))
    meta_frame = pd.DataFrame(meta, columns=list(META_COLUMNS))
    if len(meta_frame):
        meta_frame = meta_frame.sort_values("start_time", kind="stable").reset_index(drop=True)
        features = features.loc[meta_frame["flow_id"]].reset_index(drop=True)
        meta_frame["flow_id"] = np.arange(len(meta_frame))
        stats["first_packet"] = meta_frame["start_time"].min().isoformat()
        stats["last_flow_start"] = meta_frame["start_time"].max().isoformat()
    stats["flows"] = int(len(features))
    stats["zero_duration_flows"] = int((features["Flow Duration"] == 0).sum()) if len(features) else 0
    logger.info("Capture: %d packets read, %d used, %d flows", stats["packets_read"], stats["packets_used"], stats["flows"])
    return CaptureFlows(features=features, meta=meta_frame, stats=stats)
