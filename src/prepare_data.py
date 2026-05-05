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
    # Memória optimalizálás: float64 helyett float32 használata a numerikus oszlopokhoz
    df = pd.read_csv(csv_path)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)

    # MÓDOSÍTÁS: Mintavételezés kikapcsolva, a teljes 9,6M sor feldolgozása
    print(f"Teljes adathalmaz betöltve: {len(df)} sor.")

    leakage_cols = ['Dmonth', 'Dday', 'Dhour', 'Dmin', 'DDweek']
    df = df.drop(columns=leakage_cols)

    print("Adatok felosztása (train/test split)...")
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    os.makedirs(SPLITS_DIR, exist_ok=True)
    print("Mentés Parquet formátumba...")
    train_df.to_parquet(SPLITS_DIR / "train_v0.parquet")
    test_df.to_parquet(SPLITS_DIR / "test_v0.parquet")

    print(f"Siker! A teljes adathalmaz elkészült itt: {SPLITS_DIR}")

if __name__ == "__main__":
    prepare()