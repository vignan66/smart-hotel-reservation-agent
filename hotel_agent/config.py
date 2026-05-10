from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a local .env if present
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Centralized configuration.

    This keeps the repository reproducible: the code runs in offline mode by default,
    and only enables external services if keys are configured.
    """

    # Data paths
    d1_raw_path: Path = Path(os.getenv("D1_RAW_PATH", "data/raw/d1_hotels.csv"))
    d1_sample_path: Path = Path(os.getenv("D1_SAMPLE_PATH", "data/sample/sample_hotels.csv"))

    d3_pairs_path: Path = Path(os.getenv("D3_PAIRS_PATH", "data/processed/d3_pairs.csv"))

    # Model names
    e5_model_name: str = os.getenv("E5_MODEL_NAME", "intfloat/e5-small-v2")
    cross_encoder_name: str = os.getenv("CROSS_ENCODER_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Optional: Google Custom Search (if you want live retrieval)
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    google_cse_id: str | None = os.getenv("GOOGLE_CSE_ID")

    # App behavior
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", "15"))


def get_settings() -> Settings:
    return Settings()
