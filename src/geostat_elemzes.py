import pandas as pd
import numpy as np
import joblib

from src.data_io import load_train_test
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor

from config import SPLITS_DIR, RANDOM_SEED


# 1. ADATBETÖLTÉS
X_train, X_test, y_train, y_test = load_train_test()


# 2. ADATTISZTÍTÁS
def clean_data_logic(X, y):
    df = X.copy()
    df["target"] = y

    # Reális utazási idők megtartása
    df = df[(df["target"] >= 1) & (df["target"] <= 180)]

    return df.drop(columns=["target"]), df["target"]


X_train, y_train = clean_data_logic(X_train, y_train)
X_test, y_test = clean_data_logic(X_test, y_test)


# 3. FEATURE ENGINEERING - SZIVÁRGÁSMENTES
def feature_engineering_transform(df):
    df = df.copy()

    # Biztonság: ha valahogy benne maradt a Distance, töröljük.
    # Nem használjuk sem közvetlenül, sem route_directness számítására.
    if "Distance" in df.columns:
        df = df.drop(columns=["Distance"])

    # Leadási időpont alapú leakage oszlopok törlése, ha esetleg bent maradtak
    leakage_cols = ["Dmonth", "Dday", "Dhour", "Dmin", "DDweek"]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    # Redundáns időjárási feature törlése
    if "GroundTemp" in df.columns:
        df = df.drop(columns=["GroundTemp"])

    # Koordinátákból számított távolságok - induláskor ismertek
    df["manhattan_dist"] = (
        (df["PLatd"] - df["DLatd"]).abs()
        + (df["PLong"] - df["DLong"]).abs()
    )

    # Haversine nullás utak jelölése
    df["is_same_station"] = (df["Haversine"] < 1e-6).astype(int)

    # Ciklikus idő-kódolás
    df["hour_sin"] = np.sin(2 * np.pi * df["Phour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Phour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["Pmonth"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Pmonth"] / 12)

    # Bináris időbeli jellemzők
    df["is_rush_hour"] = df["Phour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_weekend"] = (df["PDweek"] >= 5).astype(int)

    # Napszak kategória
    df["part_of_day"] = pd.cut(
        df["Phour"],
        bins=[-1, 5, 11, 16, 20, 23],
        labels=["night", "morning", "afternoon", "evening", "late"]
    )

    # One-hot encoding
    df = pd.get_dummies(df, columns=["PDweek", "part_of_day"], prefix=["day", "tod"])

    # Eredeti idő oszlopok közül párat eldobunk, mert a ciklikus verzió jobb
    drop_cols = ["Pmin"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# 4. PIPELINE
# HistGradientBoostinghoz nem kell StandardScaler.
# SelectKBest-et kivesszük, mert kidobhat hasznos nemlineáris feature-öket.
data_pipeline = Pipeline([
    ("feature_eng", FunctionTransformer(feature_engineering_transform)),
    ("var_threshold", VarianceThreshold(threshold=0.0))
])


print("Pipeline futtatása szivárgásmentes feature-ökkel...")
X_train_final = data_pipeline.fit_transform(X_train, y_train)
X_test_final = data_pipeline.transform(X_test)


# 5. ERŐSEBB MODELL LOG-TARGETTEL
# A Duration jobbra ferde, ezért log1p target transzformáció gyakran javít.
base_model = HistGradientBoostingRegressor(
    loss="squared_error",
    max_iter=700,
    learning_rate=0.04,
    max_leaf_nodes=63,
    min_samples_leaf=30,
    l2_regularization=0.05,
    random_state=RANDOM_SEED,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30
)

model = TransformedTargetRegressor(
    regressor=base_model,
    func=np.log1p,
    inverse_func=np.expm1
)

print("Modell tanítása: HistGradientBoostingRegressor log-target transzformációval...")
model.fit(X_train_final, y_train)


# 6. KIÉRTÉKELÉS
print("Kiértékelés...")
y_pred = model.predict(X_test_final)

# Biztonság: negatív predikció ne legyen
y_pred = np.maximum(y_pred, 1)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Szivárgásmentes modell teljesítmény (500k mintán) ---")
print(f"R2 Score:  {r2:.4f}")
print(f"MAE:       {mae:.4f} perc")


# 7. MENTÉS
print("Pipeline és modell mentése...")

np.save(SPLITS_DIR / "X_train_final_500k.npy", X_train_final)
np.save(SPLITS_DIR / "X_test_final_500k.npy", X_test_final)

y_train.to_csv(SPLITS_DIR / "y_train_final_500k.csv", index=False)
y_test.to_csv(SPLITS_DIR / "y_test_final_500k.csv", index=False)

joblib.dump(data_pipeline, SPLITS_DIR / "data_pipeline_final.joblib")
joblib.dump(model, SPLITS_DIR / "hgb_log_model_final.joblib")

print(f"Siker! Modell és pipeline mentve ide: {SPLITS_DIR}")