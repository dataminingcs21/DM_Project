"""A small synthetic packet capture for trying the application without a capture of your own.

It contains ordinary traffic (HTTPS sessions and DNS lookups from a few workstations) and one SYN port
scan (one host probing ports 1-300 of a server, which answers RST for closed ports and SYN/ACK for open
ones).  The packets are generated, not recorded: it shows how the page works, it is not evidence of what
the models would do on real traffic.
"""

from __future__ import annotations

import io
import random
import socket

import dpkt

START = 1_764_064_800.0  # 2025-11-25 10:00:00 UTC, fixed so the demo is reproducible
SERVER, SCANNER, WEB, DNS = "192.168.10.50", "192.168.10.66", "93.184.216.34", "192.168.10.3"
CLIENTS = ("192.168.10.12", "192.168.10.14", "192.168.10.19", "192.168.10.25")
OPEN_PORTS = (22, 80, 443)


def _frame(ip: dpkt.ip.IP) -> bytes:
    raw = bytes(dpkt.ethernet.Ethernet(src=b"\x02\x00\x00\x00\x00\x01", dst=b"\x02\x00\x00\x00\x00\x02", type=dpkt.ethernet.ETH_TYPE_IP, data=ip))
    return raw.ljust(60, b"\x00")  # Ethernet pads short frames to the 60-byte minimum, as on a real wire


def _tcp(src: str, dst: str, sport: int, dport: int, flags: int, payload: bytes = b"", win: int = 64240, mss: bool = False) -> bytes:
    segment = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, win=win, data=payload)
    if mss:  # option MSS=1460, as nmap and most stacks put on a SYN
        segment.opts = b"\x02\x04\x05\xb4"
    segment.off = 5 + len(segment.opts) // 4
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=dpkt.ip.IP_PROTO_TCP, ttl=64, data=segment)
    ip.len = 20 + len(bytes(segment))
    return _frame(ip)


def _udp(src: str, dst: str, sport: int, dport: int, payload: bytes) -> bytes:
    datagram = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
    datagram.ulen = 8 + len(payload)
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=dpkt.ip.IP_PROTO_UDP, ttl=64, data=datagram)
    ip.len = 20 + datagram.ulen
    return _frame(ip)


def build_demo_pcap(seed: int = 7) -> bytes:
    """The demo capture as pcap bytes (deterministic for a given seed)."""
    rng = random.Random(seed)
    packets: list[tuple[float, bytes]] = []
    S, A, P, F, R = dpkt.tcp.TH_SYN, dpkt.tcp.TH_ACK, dpkt.tcp.TH_PUSH, dpkt.tcp.TH_FIN, dpkt.tcp.TH_RST

    # ordinary traffic: DNS lookup, then an HTTPS session, for every client, several times
    t = START
    for _ in range(6):
        for client in CLIENTS:
            t += rng.uniform(0.5, 3.0)
            port = rng.randint(49152, 65000)
            packets.append((t, _udp(client, DNS, port, 53, b"q" * rng.randint(28, 45))))
            packets.append((t + rng.uniform(0.002, 0.02), _udp(DNS, client, 53, port, b"r" * rng.randint(60, 180))))
            s = t + 0.05
            sport = rng.randint(49152, 65000)
            rtt = rng.uniform(0.01, 0.04)
            packets += [
                (s, _tcp(client, WEB, sport, 443, S)),
                (s + rtt, _tcp(WEB, client, 443, sport, S | A, win=65160)),
                (s + rtt * 1.5, _tcp(client, WEB, sport, 443, A)),
            ]
            s += rtt * 2
            for _ in range(rng.randint(3, 8)):  # request / response exchanges
                packets.append((s, _tcp(client, WEB, sport, 443, P | A, b"x" * rng.randint(200, 900))))
                s += rtt
                for _ in range(rng.randint(1, 4)):
                    packets.append((s, _tcp(WEB, client, 443, sport, A, b"y" * rng.randint(900, 1448))))
                    s += rng.uniform(0.0005, 0.003)
                packets.append((s, _tcp(client, WEB, sport, 443, A)))
                s += rng.uniform(0.05, 1.5)
            packets += [(s, _tcp(client, WEB, sport, 443, F | A)), (s + rtt, _tcp(WEB, client, 443, sport, F | A)), (s + rtt * 1.5, _tcp(client, WEB, sport, 443, A))]

    # one SYN scan of ports 1-300 of the server
    t = START + 20.0
    for port in range(1, 301):
        t += rng.uniform(0.0005, 0.003)
        sport = rng.randint(40000, 60000)
        packets.append((t, _tcp(SCANNER, SERVER, sport, port, S, win=1024, mss=True)))
        if port in OPEN_PORTS:
            packets.append((t + 0.0004, _tcp(SERVER, SCANNER, port, sport, S | A, win=29200)))
            packets.append((t + 0.0006, _tcp(SCANNER, SERVER, sport, port, R, win=0)))
        else:
            packets.append((t + 0.0003, _tcp(SERVER, SCANNER, port, sport, R | A, win=0)))

    packets.sort(key=lambda item: item[0])
    buffer = io.BytesIO()
    writer = dpkt.pcap.Writer(buffer)
    for ts, frame in packets:
        writer.writepkt(frame, ts=ts)
    return buffer.getvalue()
