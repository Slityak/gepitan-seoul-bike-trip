import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, PROJECT_ROOT, RANDOM_SEED, SPLITS_DIR, TEST_SIZE

SAMPLE_SIZES: dict[str, int | None] = {
    "v0": 200_000,
    "vKNN": 500_000,
    "v1": None,
}


def _find_raw_csv() -> Path:
    """For_modeling.csv: Colabon Drive-on (DATA_DIR), lokálisan a repo gyökerében."""
    for c in (DATA_DIR / "For_modeling.csv", PROJECT_ROOT / "For_modeling.csv"):
        if c.exists():
            return c
    raise FileNotFoundError(
        "For_modeling.csv sehol nincs. Próbáltam: "
        f"{DATA_DIR / 'For_modeling.csv'} és {PROJECT_ROOT / 'For_modeling.csv'}."
    )


def prepare(version: str = "v0", csv_path: str | Path | None = None) -> None:
    if version not in SAMPLE_SIZES:
        raise ValueError(
            f"Ismeretlen version: {version!r} (választható: {list(SAMPLE_SIZES)})"
        )

    csv_path = Path(csv_path) if csv_path is not None else _find_raw_csv()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nem található: {csv_path}")

    print(f"[{version}] Adatok beolvasása: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[{version}] Beolvasva {len(df):,} sor.")

    n_sample = SAMPLE_SIZES[version]
    if n_sample is not None:
        print(f"[{version}] Mintavételezés: {n_sample:,} sor.")
        df = df.sample(n=n_sample, random_state=RANDOM_SEED)

    leakage_cols = ["Dmonth", "Dday", "Dhour", "Dmin", "DDweek"]
    df = df.drop(columns=leakage_cols)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    os.makedirs(SPLITS_DIR, exist_ok=True)
    train_df.to_parquet(SPLITS_DIR / f"train_{version}.parquet")
    test_df.to_parquet(SPLITS_DIR / f"test_{version}.parquet")
    print(
        f"[{version}] Kész: train={len(train_df):,}, test={len(test_df):,} → {SPLITS_DIR}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=list(SAMPLE_SIZES), default="v0")
    parser.add_argument("--csv", default=None, help="Útvonal a For_modeling.csv-hez (opcionális)")
    args = parser.parse_args()
    prepare(args.version, csv_path=args.csv)