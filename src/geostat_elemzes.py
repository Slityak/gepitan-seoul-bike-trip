"""Feature engineering + scaling pipeline. Bemenet: train/test parquet.
Kimenet: X_*_{version}.npy + y_*_{version}.csv + feature_names_{version}.json.
vKNN esetén KNN fit + R²."""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

from .config import DATA_DIR, DATA_VERSION, RANDOM_SEED, SPLITS_DIR
from .data_io import load_train_test
from .feature_engineering import build_pipeline


def clean_data_logic(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Cél: 1–180 perc közötti utak, és Distance ≥ 10 m (a Distance-t a feature
    engineering később dobja, csak a degenerált sorok szűréséhez kell)."""
    df = X.copy()
    df["target"] = y
    df = df[(df["target"] >= 1) & (df["target"] <= 180)]
    df = df[df["Distance"] >= 10]
    return df.drop(columns=["target"]), df["target"]


def build_features(
    version: str | None = None,
    use_target_encoding: bool = False,
    log_target: bool = False,
    n_zones: int = 20,
) -> dict:
    """Lefuttatja a feature engineering + scaling pipeline-t a megadott verzión.

    Args:
        version: "v0" | "vKNN" | "v1". None → config DATA_VERSION.
        use_target_encoding: True esetén a start_zone/end_zone/PDweek kategóriákat
            sklearn TargetEncoder-rel kódoljuk (cv=5, smooth='auto').
        log_target: True esetén log1p-vel transzformáljuk a y_train/y_test-et.
            A mentett CSV-ben ekkor a transzformált érték van, és a fájlnév suffixe
            "_log".
        n_zones: KMeans klaszterek száma start/end zónákhoz.

    Returns:
        dict shapes-szel és — ha vKNN — a KNN R² értékkel.
    """
    version = version or DATA_VERSION

    X_train, X_test, y_train, y_test = load_train_test(version=version)

    X_train, y_train = clean_data_logic(X_train, y_train)
    X_test, y_test = clean_data_logic(X_test, y_test)

    if log_target:
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)

    pipe = build_pipeline(
        use_target_encoding=use_target_encoding,
        n_zones=n_zones,
        random_state=RANDOM_SEED,
    )

    print(
        f"[{version}] Pipeline fit/transform "
        f"(target_encoding={use_target_encoding}, log_target={log_target}, n_zones={n_zones})..."
    )
    X_train_final = pipe.fit_transform(X_train, y_train)
    X_test_final = pipe.transform(X_test)

    feature_names = list(pipe.named_steps["pre"].get_feature_names_out())

    result: dict = {
        "version": version,
        "X_train_shape": tuple(X_train_final.shape),
        "X_test_shape": tuple(X_test_final.shape),
        "n_features": len(feature_names),
        "use_target_encoding": use_target_encoding,
        "log_target": log_target,
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

    suffix = f"{version}_log" if log_target else version

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / f"X_train_{suffix}.npy", X_train_final.astype(np.float32))
    np.save(DATA_DIR / f"X_test_{suffix}.npy", X_test_final.astype(np.float32))
    pd.Series(np.asarray(y_train), name="Duration").to_csv(
        DATA_DIR / f"y_train_{suffix}.csv", index=False
    )
    pd.Series(np.asarray(y_test), name="Duration").to_csv(
        DATA_DIR / f"y_test_{suffix}.csv", index=False
    )
    with open(DATA_DIR / f"feature_names_{suffix}.json", "w", encoding="utf-8") as fh:
        json.dump(feature_names, fh, ensure_ascii=False, indent=2)
    print(
        f"[{version}] Mentve: X_train_{suffix}.npy / X_test_{suffix}.npy / "
        f"y_*_{suffix}.csv / feature_names_{suffix}.json → {DATA_DIR}"
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, SPLITS_DIR / f"data_pipeline_{suffix}.joblib")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v0", "vKNN", "v1"], default=None)
    parser.add_argument(
        "--target-encoding", action="store_true",
        help="start_zone/end_zone/PDweek TargetEncoder-rel (alap: OneHot)",
    )
    parser.add_argument(
        "--log-target", action="store_true",
        help="log1p(Duration) a target oszlopra",
    )
    parser.add_argument("--n-zones", type=int, default=20)
    args = parser.parse_args()
    build_features(
        args.version,
        use_target_encoding=args.target_encoding,
        log_target=args.log_target,
        n_zones=args.n_zones,
    )
