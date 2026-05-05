"""Közös konfiguráció: path-ok, konstansok, környezet-detektálás."""

from __future__ import annotations

import os
from pathlib import Path

# Reprodukálhatóság
RANDOM_SEED: int = 42

# Train/test split arány
TEST_SIZE: float = 0.2

# Cross-validation fold-ok száma
CV_FOLDS: int = 5

# Környezet detektálása (Colab vs lokális)
IN_COLAB: bool = "google.colab" in os.environ.get("PYTHONPATH", "") or os.path.exists(
    "/content"
)


def get_project_root() -> Path:
    """Visszaadja a projekt gyökérkönyvtárát Colabon és lokálisan is."""
    if IN_COLAB:
        return Path("/content/gepitan-beadando")
    # Lokálisan: a src/config.py két szinttel a gyökér alatt van
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Az adatok elérési útja. Colabon Drive-ról, lokálisan a repo data/ mappájából."""
    if IN_COLAB:
        return Path("/content/drive/MyDrive/GepiTan_Beadando/data")
    return get_project_root() / "data"


def get_outputs_root() -> Path:
    """Modellek és eredmények gyökere. Colabon Drive-on lokálisan a repo gyökerében."""
    if IN_COLAB:
        return Path("/content/drive/MyDrive/GepiTan_Beadando")
    return get_project_root()


# Származtatott path-ok
PROJECT_ROOT: Path = get_project_root()
DATA_DIR: Path = get_data_dir()
SPLITS_DIR: Path = DATA_DIR / "splits"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = get_outputs_root() / "models"
RESULTS_DIR: Path = get_outputs_root() / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
METRICS_CSV: Path = RESULTS_DIR / "metrics.csv"

# Adat-verzió kapcsoló — a notebookok ezt használják
# Lehet: "v0" (200k sample, sanity check) vagy "v1" (teljes adat, feature engineered)
DATA_VERSION: str = "v0"

# Subsample fejlesztéshez (None = teljes adat a kiválasztott verzióból)
# Notebook tetején override-olható
SAMPLE_SIZE: int | None = None

# Target oszlop neve
TARGET_COLUMN: str = "Duration"

def ensure_dirs() -> None:
    """Biztosítja, hogy a kimeneti könyvtárak léteznek."""
    for directory in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
