"""
Generates a SYNTHETIC sample dataset that mimics the UNSW-NB15 schema
(same column names/types, plausible-looking distributions) so you can
run and debug the entire pipeline before your real UNSW-NB15 download
finishes.

IMPORTANT: This is NOT real network traffic data. Do not report results
trained on this sample data as your project's findings -- it exists only
to let you verify the code works end-to-end. Swap in the real CSVs and
re-run once you have them (see README.md).
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import SAMPLE_TRAIN_FILE, SAMPLE_TEST_FILE, ATTACK_CATEGORIES, SAMPLE_DATA_DIR

RNG = np.random.default_rng(42)

PROTOS = ["tcp", "udp", "icmp", "arp", "ospf"]
SERVICES = ["-", "http", "ftp", "dns", "smtp", "ssh", "pop3", "dhcp"]
STATES = ["FIN", "CON", "INT", "REQ", "RST", "ECO"]


def _make_split(n_rows, start_id=1):
    attack_cat = RNG.choice(ATTACK_CATEGORIES, size=n_rows, p=[
        0.55, 0.14, 0.10, 0.06, 0.05, 0.04, 0.03, 0.015, 0.01, 0.005
    ])
    label = (attack_cat != "Normal").astype(int)

    n = n_rows
    df = pd.DataFrame({
        "id": np.arange(start_id, start_id + n),
        "dur": RNG.exponential(0.5, n).round(6),
        "proto": RNG.choice(PROTOS, n, p=[0.6, 0.25, 0.1, 0.03, 0.02]),
        "service": RNG.choice(SERVICES, n),
        "state": RNG.choice(STATES, n),
        "spkts": RNG.poisson(10, n) + label * RNG.poisson(20, n),
        "dpkts": RNG.poisson(8, n),
        "sbytes": (RNG.exponential(500, n) + label * RNG.exponential(1500, n)).round(0),
        "dbytes": RNG.exponential(400, n).round(0),
        "rate": RNG.exponential(50, n).round(3),
        "sttl": RNG.choice([254, 62, 31, 128], n),
        "dttl": RNG.choice([254, 62, 31, 128], n),
        "sload": RNG.exponential(1000, n).round(3),
        "dload": RNG.exponential(800, n).round(3),
        "sloss": RNG.poisson(1, n),
        "dloss": RNG.poisson(1, n),
        "sinpkt": RNG.exponential(10, n).round(3),
        "dinpkt": RNG.exponential(10, n).round(3),
        "sjit": RNG.exponential(5, n).round(3),
        "djit": RNG.exponential(5, n).round(3),
        "swin": RNG.choice([0, 255], n),
        "stcpb": RNG.integers(0, 4294967295, n, dtype=np.int64),
        "dtcpb": RNG.integers(0, 4294967295, n, dtype=np.int64),
        "dwin": RNG.choice([0, 255], n),
        "tcprtt": RNG.exponential(0.05, n).round(6),
        "synack": RNG.exponential(0.02, n).round(6),
        "ackdat": RNG.exponential(0.02, n).round(6),
        "smean": RNG.exponential(100, n).round(1),
        "dmean": RNG.exponential(90, n).round(1),
        "trans_depth": RNG.poisson(0.5, n),
        "response_body_len": RNG.exponential(200, n).round(0),
        "ct_srv_src": RNG.poisson(3, n),
        "ct_state_ttl": RNG.poisson(2, n),
        "ct_dst_ltm": RNG.poisson(2, n),
        "ct_src_dport_ltm": RNG.poisson(2, n),
        "ct_dst_sport_ltm": RNG.poisson(2, n),
        "ct_dst_src_ltm": RNG.poisson(2, n),
        "is_ftp_login": RNG.choice([0, 1], n, p=[0.95, 0.05]),
        "ct_ftp_cmd": RNG.poisson(0.2, n),
        "ct_flw_http_mthd": RNG.poisson(0.3, n),
        "ct_src_ltm": RNG.poisson(2, n),
        "ct_srv_dst": RNG.poisson(3, n),
        "is_sm_ips_ports": RNG.choice([0, 1], n, p=[0.9, 0.1]),
        "attack_cat": attack_cat,
        "label": label,
    })
    return df


def main():
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
    train_df = _make_split(4000, start_id=1)
    test_df = _make_split(1000, start_id=100001)
    train_df.to_csv(SAMPLE_TRAIN_FILE, index=False)
    test_df.to_csv(SAMPLE_TEST_FILE, index=False)
    print(f"Wrote {len(train_df)} rows -> {SAMPLE_TRAIN_FILE}")
    print(f"Wrote {len(test_df)} rows -> {SAMPLE_TEST_FILE}")
    print("\nLabel balance (train):")
    print(train_df["label"].value_counts(normalize=True).round(3))
    print("\nAttack category balance (train):")
    print(train_df["attack_cat"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
