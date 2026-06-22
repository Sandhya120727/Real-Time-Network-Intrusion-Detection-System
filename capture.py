"""
capture.py - Live Packet Capture using Scapy
Captures real network packets and converts to CICIDS2017 features
"""

from scapy.all import sniff, IP, TCP, UDP
import pandas as pd
import numpy as np
from collections import defaultdict
import time
import os

# Storage for flows
flows = defaultdict(list)
flow_stats = []

def get_flow_key(pkt):
    """Create a unique key for each network flow."""
    if IP in pkt:
        src = pkt[IP].src
        dst = pkt[IP].dst
        proto = pkt[IP].proto
        sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
        dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
        return (src, dst, sport, dport, proto)
    return None

def extract_features(pkt_list):
    """Extract CICIDS2017-like features from a list of packets in a flow."""
    if not pkt_list:
        return None

    lengths = [len(p) for p in pkt_list]
    times   = [p.time for p in pkt_list]

    # Inter-arrival times
    iats = [times[i+1] - times[i] for i in range(len(times)-1)] if len(times) > 1 else [0]

    # TCP flags
    syn_count = sum(1 for p in pkt_list if TCP in p and p[TCP].flags & 0x02)
    rst_count = sum(1 for p in pkt_list if TCP in p and p[TCP].flags & 0x04)
    psh_count = sum(1 for p in pkt_list if TCP in p and p[TCP].flags & 0x08)

    duration = times[-1] - times[0] if len(times) > 1 else 0.001
    total_bytes = sum(lengths)

    features = {
        'Destination Port'              : pkt_list[0][TCP].dport if TCP in pkt_list[0] else 0,
        'Flow Duration'                 : duration * 1e6,
        'Total Fwd Packets'             : len(pkt_list),
        'Total Backward Packets'        : 0,
        'Total Length of Fwd Packets'   : total_bytes,
        'Total Length of Bwd Packets'   : 0,
        'Fwd Packet Length Max'         : max(lengths),
        'Fwd Packet Length Mean'        : np.mean(lengths),
        'Bwd Packet Length Mean'        : 0,
        'Flow Bytes/s'                  : total_bytes / duration if duration > 0 else 0,
        'Flow Packets/s'                : len(pkt_list) / duration if duration > 0 else 0,
        'Flow IAT Mean'                 : np.mean(iats),
        'Flow IAT Std'                  : np.std(iats),
        'Fwd IAT Total'                 : sum(iats),
        'Bwd IAT Total'                 : 0,
        'Fwd PSH Flags'                 : psh_count,
        'SYN Flag Count'                : syn_count,
        'RST Flag Count'                : rst_count,
        'Avg Fwd Segment Size'          : np.mean(lengths),
        'Init_Win_bytes_forward'        : pkt_list[0][TCP].window if TCP in pkt_list[0] else 0,
    }
    return features


def capture_live(duration=30, output_file="data/live_capture.csv"):
    """
    Capture live packets for `duration` seconds.
    Saves result as CSV to output_file.
    """
    print(f"\n[INFO] Starting live capture for {duration} seconds...")
    print("[INFO] Capturing packets from your network...")
    print("[INFO] Press Ctrl+C to stop early.\n")

    captured = []

    def process_packet(pkt):
        key = get_flow_key(pkt)
        if key:
            flows[key].append(pkt)

    # Capture packets
    sniff(timeout=duration, prn=process_packet, store=False)

    print(f"\n[INFO] Capture done. Processing {len(flows)} flows...")

    for key, pkts in flows.items():
        if len(pkts) >= 2:
            features = extract_features(pkts)
            if features:
                captured.append(features)

    if not captured:
        print("[WARNING] No flows captured. Make sure you have network activity.")
        return None

    df = pd.DataFrame(captured)
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"[INFO] Saved {len(df)} flows to {output_file}")
    return output_file


if __name__ == "__main__":
    capture_live(duration=30)