from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, required=True, help="Path to D3 pairs CSV")
    ap.add_argument("--output_dir", type=str, default="models/reranker", help="Where to save fine-tuned model")
    ap.add_argument("--base_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder, InputExample

    df = pd.read_csv(Path(args.pairs))
    required = {"query", "hotel_card", "label_relevant"}
    if not required.issubset(df.columns):
        raise ValueError(f"Pairs CSV missing columns {required}. Found: {list(df.columns)}")

    samples = [
        InputExample(texts=[str(r.query), str(r.hotel_card)], label=float(r.label_relevant))
        for r in df.itertuples(index=False)
    ]

    train_loader = DataLoader(samples, shuffle=True, batch_size=args.batch_size)

    model = CrossEncoder(args.base_model, num_labels=1)

    # Heuristic warmup
    warmup_steps = int(len(train_loader) * args.epochs * 0.1)

    model.fit(
        train_dataloader=train_loader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        show_progress_bar=True,
    )

    print(f"Saved fine-tuned reranker to: {args.output_dir}")


if __name__ == "__main__":
    main()
