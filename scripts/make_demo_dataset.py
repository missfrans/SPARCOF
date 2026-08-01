#!/usr/bin/env python3
"""Create a small deterministic dataset for installation and CI smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/demo/demo.csv"))
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rows < 30:
        raise ValueError("Use at least 30 rows so stratified splitting and CV remain meaningful.")
    rng = np.random.default_rng(args.seed)
    signal_a = rng.normal(size=args.rows)
    signal_b = rng.normal(size=args.rows)
    category = rng.choice(["tcp", "udp", "icmp"], size=args.rows, p=[0.5, 0.35, 0.15])
    logits = 1.4 * signal_a - 0.7 * signal_b + (category == "icmp") * 1.1 + rng.normal(0, 0.7, args.rows)
    label = np.where(logits > np.median(logits), "attack", "normal")
    frame = pd.DataFrame(
        {
            "record_id": np.arange(args.rows),
            "signal_a": signal_a,
            "protocol": category,
            "signal_b": signal_b,
            "noise": rng.normal(size=args.rows),
            "label": label,
        }
    )
    frame.loc[frame.index[::47], "protocol"] = np.nan
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
