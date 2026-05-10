from __future__ import annotations

import argparse
from pathlib import Path

from hotel_agent.data.d1_loader import load_hotels_csv
from hotel_agent.data.d3_builder import D3BuildConfig, build_d3_pairs, save_pairs_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1_path", type=str, required=True, help="Path to D1 hotels CSV")
    ap.add_argument("--out_path", type=str, required=True, help="Output CSV path for D3 pairs")
    ap.add_argument("--n_queries", type=int, default=200)
    ap.add_argument("--candidates", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hotels = load_hotels_csv(Path(args.d1_path))
    cfg = D3BuildConfig(seed=args.seed, n_queries=args.n_queries, candidates_per_query=args.candidates)
    df = build_d3_pairs(hotels, cfg)
    save_pairs_csv(df, Path(args.out_path))

    pos_rate = df["label_relevant"].mean()
    print(f"Saved {len(df):,} pairs ({pos_rate*100:.1f}% positive) to {args.out_path}")


if __name__ == "__main__":
    main()
