# Smart Hotel Reservation Agent (CECS 551 - AI)

Team **Almost Autonomous**

This repository contains our CECS 551 term project: a **Smart Hotel Reservation Agent** that supports:

- **Hotel search** (offline dataset by default; optional live search integration)
- **Price comparison simulation** across multiple booking sites
- **Deep-learning ranking** (E5 embeddings + Transformer cross-encoder reranking)
- **Conversational booking flow** with explicit state tracking (dates, guests, room type, confirmation)

The goal is to reduce "tab-hopping" and booking friction by returning a short, trustworthy shortlist and then guiding the user through booking steps.

---

## 1) Quickstart (runs with included sample data)

### Prereqs
- Python **3.10+**

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run app.py
```

The app will launch in **offline demo mode** using `data/sample/sample_hotels.csv`.

---

## 2) Optional: Use larger datasets

We designed the repository so you can use real datasets without committing large files.

### D1 (Hotel listings)
Place your full hotel listings CSV at:
- `data/raw/d1_hotels.csv`

The app will automatically prefer `data/raw/d1_hotels.csv` if it exists.

### D3 (Ranking pairs)
We generate training/evaluation pairs from D1:
```bash
python scripts/build_d3_pairs.py --d1_path data/raw/d1_hotels.csv --out_path data/processed/d3_pairs.csv
```

---

## 3) Deep ranking pipeline

We support 3 ranking modes:
- **BM25 baseline** (strong lexical baseline)
- **E5 embedding retrieval** (dense retrieval)
- **E5 + cross-encoder reranking** (best quality)

### Evaluate ranking (offline)
```bash
python scripts/eval_ranker.py --pairs data/processed/d3_pairs.csv
```

### Optional: Fine-tune the cross-encoder reranker
Fine-tuning is optional but recommended for best results.

```bash
python scripts/train_reranker.py \
  --pairs data/processed/d3_pairs.csv \
  --output_dir models/reranker
```

---

## 4) Repo structure

```
.
├── app.py
├── hotel_agent/
│   ├── booking/        # booking state machine
│   ├── data/           # loading + preprocessing
│   ├── ranking/        # BM25 + embeddings + reranker + metrics
│   ├── tools/          # offline search + price simulation
│   └── utils/
├── scripts/            # build pairs, train, eval
├── data/
│   ├── sample/         # small sample to run out-of-the-box
│   ├── raw/            # (gitignored) large datasets
│   └── processed/      # (gitignored) derived datasets
└── assets/
```

---

## 5) Notes on reproducibility

- Large datasets are not committed. The README explains where to place them.
- The app runs end-to-end with the sample dataset.
- All ranking evaluation is scripted via `scripts/`.

---

## 6) License

Educational project for CECS 551 (CSULB).
