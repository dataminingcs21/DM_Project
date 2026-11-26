"""Tests of the packet-capture to flow-feature extractor (synthetic packets written with dpkt)."""

from __future__ import annotations

import io
import socket

import dpkt
import numpy as np
import pytest

from nids.capture import pcap_flows as P
from nids.capture.demo import SCANNER, build_demo_pcap

S, A, PSH, F, R = dpkt.tcp.TH_SYN, dpkt.tcp.TH_ACK, dpkt.tcp.TH_PUSH, dpkt.tcp.TH_FIN, dpkt.tcp.TH_RST
C, SRV = "10.0.0.1", "10.0.0.2"


def eth(ip: dpkt.ip.IP, pad: bool = False) -> bytes:
    raw = bytes(dpkt.ethernet.Ethernet(src=b"\x00" * 6, dst=b"\x11" * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=ip))
    return raw.ljust(60, b"\x00") if pad else raw


def tcp(src, dst, sport, dport, flags, payload=b"", win=8192, pad=False):
    seg = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, win=win, data=payload)
    seg.off = 5
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=6, data=seg)
    ip.len = 20 + len(bytes(seg))
    return eth(ip, pad)


def udp(src, dst, sport, dport, payload):
    dg = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
    dg.ulen = 8 + len(payload)
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=17, data=dg)
    ip.len = 20 + dg.ulen
    return eth(ip)


def capture(packets) -> bytes:
    buffer = io.BytesIO()
    writer = dpkt.pcap.Writer(buffer)
    for ts, frame in packets:
        writer.writepkt(frame, ts=ts)
    return buffer.getvalue()


SESSION = [
    (1.0, tcp(C, SRV, 40000, 80, S, win=29200)),
    (1.1, tcp(SRV, C, 80, 40000, S | A, win=28960)),
    (1.2, tcp(C, SRV, 40000, 80, A | PSH, b"x" * 100)),
    (1.4, tcp(SRV, C, 80, 40000, A, b"y" * 300)),
    (1.5, tcp(C, SRV, 40000, 80, F | A)),
]


class TestFeatures:
    def test_hand_worked_tcp_session(self):
        flows = P.extract_flows(capture(SESSION), flag_mode="real")
        assert flows.n_flows == 1
        f = flows.features.iloc[0]
        expected = {
            "Destination Port": 80, "Flow Duration": 500_000, "Total Fwd Packets": 3, "Total Backward Packets": 2,
            "Total Length of Fwd Packets": 100, "Total Length of Bwd Packets": 300, "Fwd Packet Length Min": 0,
            "Fwd Packet Length Mean": 100 / 3, "Bwd Packet Length Min": 0, "Flow Bytes/s": 800, "Flow IAT Min": 100_000,
            "Fwd IAT Total": 500_000, "Fwd IAT Min": 200_000, "Bwd IAT Total": 300_000, "Fwd Header Length": 60,
            "Bwd Header Length": 40, "Bwd Packets/s": 4, "Max Packet Length": 300, "Down/Up Ratio": 0,
            "Init_Win_bytes_forward": 29200, "Init_Win_bytes_backward": 28960, "act_data_pkt_fwd": 1,
            "min_seg_size_forward": 20, "FIN Flag Count": 1, "PSH Flag Count": 1, "ACK Flag Count": 1, "RST Flag Count": 0,
            "Fwd PSH Flags": 1, "Active Mean": 500_000, "Idle Mean": 0,
        }
        for name, value in expected.items():
            assert f[name] == pytest.approx(value), name
        assert f["Fwd Packet Length Std"] == pytest.approx(np.std([0, 100, 0], ddof=1))
        meta = flows.meta.iloc[0]
        assert (meta["src_ip"], meta["src_port"], meta["dst_ip"], meta["dst_port"], meta["protocol"]) == (C, 40000, SRV, 80, "TCP")

    def test_cic_flag_mode(self):
        f = P.extract_flows(capture(SESSION)).features.iloc[0]  # default: CIC-IDS2017 compatible
        assert (f["PSH Flag Count"], f["ACK Flag Count"], f["FIN Flag Count"], f["Fwd PSH Flags"]) == (1, 0, 0, 0)
        mid_stream = capture(SESSION[2:4])  # starts without a SYN
        g = P.extract_flows(mid_stream).features.iloc[0]
        assert (g["PSH Flag Count"], g["ACK Flag Count"]) == (0, 1)

    def test_bad_flag_mode(self):
        with pytest.raises(ValueError):
            P.extract_flows(capture(SESSION), flag_mode="other")

    def test_feature_columns_match_the_model_features(self):
        assert len(P.FEATURES) == 40 and len(set(P.FEATURES)) == 40
        assert list(P.extract_flows(capture(SESSION)).features.columns) == list(P.FEATURES)

    def test_udp_flow_and_skipped_packets(self):
        arp = bytes(dpkt.ethernet.Ethernet(type=dpkt.ethernet.ETH_TYPE_ARP, data=b"\x00" * 28))
        flows = P.extract_flows(capture([(1.0, udp(C, "8.8.8.8", 5353, 53, b"q" * 40)), (1.05, udp("8.8.8.8", C, 53, 5353, b"r" * 120)), (1.1, arp)]))
        assert flows.stats["packets_skipped"] == 1 and flows.n_flows == 1
        f = flows.features.iloc[0]
        assert (f["Destination Port"], f["Total Length of Fwd Packets"], f["Total Length of Bwd Packets"]) == (53, 40, 120)
        assert f["Init_Win_bytes_forward"] == -1 and f["Fwd Header Length"] == 8
        assert f["ACK Flag Count"] == 0 and f["PSH Flag Count"] == 0

    def test_ethernet_padding_counts_as_payload(self):
        flows = P.extract_flows(capture([(1.0, tcp(C, SRV, 40000, 81, S, pad=True)), (1.0001, tcp(SRV, C, 81, 40000, R | A, win=0, pad=True))]))
        f = flows.features.iloc[0]
        assert f["Total Length of Fwd Packets"] == 6 and f["Total Length of Bwd Packets"] == 6
        assert f["Init_Win_bytes_backward"] == 0

    def test_zero_length_flow_has_missing_rates(self):
        f = P.extract_flows(capture([(1.0, tcp(C, SRV, 40000, 80, R))])).features.iloc[0]
        assert f["Flow Duration"] == 0 and np.isnan(f["Flow Bytes/s"]) and np.isnan(f["Bwd Packets/s"])


class TestFlowBoundaries:
    def test_fin_ends_a_flow_and_the_next_packet_starts_another(self):
        packets = SESSION + [(2.0, tcp(C, SRV, 40000, 80, A))]
        assert P.extract_flows(capture(packets)).n_flows == 2

    def test_flow_timeout(self):
        packets = [(1.0, tcp(C, SRV, 40000, 80, A)), (1.0 + P.FLOW_TIMEOUT_S + 1, tcp(C, SRV, 40000, 80, A))]
        assert P.extract_flows(capture(packets)).n_flows == 2

    def test_idle_period(self):
        packets = [(1.0, tcp(C, SRV, 40000, 80, A)), (1.5, tcp(C, SRV, 40000, 80, A)), (10.0, tcp(C, SRV, 40000, 80, A)), (10.5, tcp(C, SRV, 40000, 80, A))]
        f = P.extract_flows(capture(packets)).features.iloc[0]
        assert f["Idle Mean"] == pytest.approx(8.5e6) and f["Active Mean"] == pytest.approx(0.5e6)

    def test_pcapng_is_read(self):
        buffer = io.BytesIO()
        writer = dpkt.pcapng.Writer(buffer)
        for ts, frame in SESSION:
            writer.writepkt(frame, ts=ts)
        assert P.extract_flows(buffer.getvalue()).n_flows == 1


class TestErrors:
    @pytest.mark.parametrize("data", [b"", b"abc", b"not a capture at all, just text"])
    def test_not_a_capture(self, data):
        with pytest.raises(P.PcapError):
            P.extract_flows(data)

    def test_packet_limit(self):
        flows = P.extract_flows(capture(SESSION), max_packets=2)
        assert flows.stats["truncated"] and flows.stats["packets_read"] == 2


class TestDemo:
    def test_demo_is_deterministic_and_has_a_scan(self):
        data = build_demo_pcap()
        assert data == build_demo_pcap()
        flows = P.extract_flows(data)
        scan = flows.meta["src_ip"] == SCANNER
        assert scan.sum() == 300 and (~scan).sum() > 50
        assert flows.stats["packets_skipped"] == 0
