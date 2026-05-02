import pandas as pd
import os
from sklearn.model_selection import train_test_split
from config import PROJECT_ROOT, TEST_SIZE, RANDOM_SEED, SPLITS_DIR


def prepare():
    csv_path = PROJECT_ROOT / 'For_modeling.csv'

    if not csv_path.exists():
        raise FileNotFoundError(f"A fájl nem található: {csv_path}")

    print("Adatok beolvasása...")
    df = pd.read_csv(csv_path)

    SAMPLE_SIZE_FOR_DEV = 200000
    print(f"Mintavételezés: {SAMPLE_SIZE_FOR_DEV} sor kiválasztása...")
    df = df.sample(n=SAMPLE_SIZE_FOR_DEV, random_state=RANDOM_SEED)

    leakage_cols = ['Dmonth', 'Dday', 'Dhour', 'Dmin', 'DDweek']
    df = df.drop(columns=leakage_cols)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    os.makedirs(SPLITS_DIR, exist_ok=True)
    train_df.to_parquet(SPLITS_DIR / "train_v0.parquet")
    test_df.to_parquet(SPLITS_DIR / "test_v0.parquet")

    print(f"Siker! A {SAMPLE_SIZE_FOR_DEV} soros minta elkészült itt: {SPLITS_DIR}")

if __name__ == "__main__":
    prepare()