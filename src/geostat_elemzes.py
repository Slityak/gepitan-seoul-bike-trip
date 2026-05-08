"""Feature engineering + scaling pipeline. Bemenet: train/test parquet.
Kimenet: X_*_{version}.npy + y_*_{version}.csv. vKNN esetén KNN fit + R²."""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import DATA_DIR, DATA_VERSION, SPLITS_DIR
from .data_io import load_train_test


def clean_data_logic(X, y):
    df = X.copy()
    df["target"] = y
    df = df[(df["target"] >= 1) & (df["target"] <= 180)]
    df = df[df["Distance"] >= 10]
    return df.drop(columns=["target"]), df["target"]


def feature_engineering_transform(df):
    df = df.copy()

    if "GroundTemp" in df.columns:
        df = df.drop(columns=["GroundTemp"])

    df["manhattan_dist"] = (df["PLatd"] - df["DLatd"]).abs() + (df["PLong"] - df["DLong"]).abs()
    df["route_directness"] = df["Distance"] / (df["Haversine"] * 1000 + 1e-6)

    df["hour_sin"] = np.sin(2 * np.pi * df["Phour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Phour"] / 24)

    df["is_rush_hour"] = df["Phour"].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)
    df["is_weekend"] = df["PDweek"].apply(lambda x: 1 if x >= 5 else 0)

    df = pd.get_dummies(df, columns=["PDweek"], prefix="day")

    return df


def build_features(version: str | None = None) -> dict:
    """Lefuttatja a feature engineering + scaling pipeline-t a megadott verzión.

    Args:
        version: "v0" | "vKNN" | "v1". None esetén a config DATA_VERSION-jét használja.

    Returns:
        dict shapes-szel és — ha vKNN — a KNN R² értékkel.
    """
    version = version or DATA_VERSION

    X_train, X_test, y_train, y_test = load_train_test(version=version)

    X_train, y_train = clean_data_logic(X_train, y_train)
    X_test, y_test = clean_data_logic(X_test, y_test)

    data_pipeline = Pipeline([
        ("feature_eng", FunctionTransformer(feature_engineering_transform)),
        ("scaler", StandardScaler()),
    ])

    print(f"[{version}] Pipeline fit/transform...")
    X_train_final = data_pipeline.fit_transform(X_train)
    X_test_final = data_pipeline.transform(X_test)

    result: dict = {
        "version": version,
        "X_train_shape": X_train_final.shape,
        "X_test_shape": X_test_final.shape,
    }

    if version == "vKNN":
        print(f"[{version}] KNN fit (n_neighbors=15, weights='distance')...")
        knn = KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)
        knn.fit(X_train_final, y_train)
        y_pred = knn.predict(X_test_final)
        r2 = r2_score(y_test, y_pred)
        print(f"[{version}] R²: {r2:.4f}")
        result["r2"] = r2
    else:
        print(f"[{version}] KNN fit kihagyva (csak vKNN-en fut).")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / f"X_train_{version}.npy", X_train_final)
    np.save(DATA_DIR / f"X_test_{version}.npy", X_test_final)
    y_train.to_csv(DATA_DIR / f"y_train_{version}.csv", index=False)
    y_test.to_csv(DATA_DIR / f"y_test_{version}.csv", index=False)
    print(f"[{version}] Mentve: X_train_{version}.npy stb. → {DATA_DIR}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(data_pipeline, SPLITS_DIR / f"data_pipeline_{version}.joblib")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v0", "vKNN", "v1"], default=None)
    args = parser.parse_args()
    build_features(args.version)
