import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from config import PROJECT_ROOT, TEST_SIZE, RANDOM_SEED, SPLITS_DIR


def prepare():
    csv_path = PROJECT_ROOT / 'For_modeling.csv'

    if not csv_path.exists():
        raise FileNotFoundError(f"A fájl nem található: {csv_path}")

    print("Adatok beolvasása és típusoptimalizálás...")
    # Memória optimalizálás: float64 helyett float32 használata
    df = pd.read_csv(csv_path)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)

    print(f"Teljes adathalmaz betöltve: {len(df)} sor.")
    #df = df.sample(n=2000000, random_state=RANDOM_SEED)

    # JAVÍTÁS: A 'Distance' eltávolítása, mert ez predikciókor még nem ismert (Leakage)
    # A többi időpont-alapú leakage oszlop mellé bekerült a Distance is.
    leakage_cols = ['Dmonth', 'Dday', 'Dhour', 'Dmin', 'DDweek', 'Distance']

    print(f"Leakage oszlopok eltávolítása: {leakage_cols}")
    df = df.drop(columns=leakage_cols)

    print("Adatok felosztása (train/test split)...")
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    os.makedirs(SPLITS_DIR, exist_ok=True)
    print("Mentés Parquet formátumba...")
    # A Parquet fájlok már nem fogják tartalmazni a szivárgó oszlopokat
    train_df.to_parquet(SPLITS_DIR / "train_v0.parquet")
    test_df.to_parquet(SPLITS_DIR / "test_v0.parquet")

    print(f"Siker! A tisztított adathalmaz elkészült itt: {SPLITS_DIR}")


if __name__ == "__main__":
    prepare()