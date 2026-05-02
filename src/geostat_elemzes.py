import pandas as pd
import numpy as np
from src.data_io import load_train_test
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from config import SPLITS_DIR

# 1. ADATBETÖLTÉS
X_train, X_test, y_train, y_test = load_train_test()


# --- 1. LÉPÉS: TISZTÍTÁS
def clean_data_logic(X, y):
    df = X.copy()
    df['target'] = y
    df = df[(df['target'] >= 1) & (df['target'] <= 180)]
    df = df[df['Distance'] >= 10]
    return df.drop(columns=['target']), df['target']


X_train, y_train = clean_data_logic(X_train, y_train)
X_test, y_test = clean_data_logic(X_test, y_test)


# --- 2. LÉPÉS: FEATURE ENGINEERING FÜGGVÉNY ---
def feature_engineering_transform(df):
    df = df.copy()

    if 'GroundTemp' in df.columns:
        df = df.drop(columns=['GroundTemp'])

    df['manhattan_dist'] = (df['PLatd'] - df['DLatd']).abs() + (df['PLong'] - df['DLong']).abs()
    df['route_directness'] = df['Distance'] / (df['Haversine'] * 1000 + 1e-6)

    df['hour_sin'] = np.sin(2 * np.pi * df['Phour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Phour'] / 24)

    df['is_rush_hour'] = df['Phour'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)
    df['is_weekend'] = df['PDweek'].apply(lambda x: 1 if x >= 5 else 0)

    df = pd.get_dummies(df, columns=['PDweek'], prefix='day')

    return df


# --- 3. LÉPÉS: A PIPELINE ÖSSZEÁLLÍTÁSA ---
data_pipeline = Pipeline([
    ('feature_eng', FunctionTransformer(feature_engineering_transform)),
    ('scaler', StandardScaler())
])

# Tanítás és transzformáció
print("Pipeline futtatása...")
X_train_final = data_pipeline.fit_transform(X_train)
X_test_final = data_pipeline.transform(X_test)

# 4. MODELLEZÉS (KNN)
knn = KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1)
knn.fit(X_train_final, y_train)

# 5. KIÉRTÉKELÉS
y_pred = knn.predict(X_test_final)
print(f"R2: {r2_score(y_test, y_pred):.4f}")

# MENTÉS

np.save(SPLITS_DIR / "X_train_final.npy", X_train_final) # Itt _final kell _scaled helyett
np.save(SPLITS_DIR / "X_test_final.npy", X_test_final)   # Itt is _final kell
y_train.to_csv(SPLITS_DIR / "y_train_final.csv", index=False)
y_test.to_csv(SPLITS_DIR / "y_test_final.csv", index=False)

# A teljes pipeline mentése[cite: 3, 4]
joblib.dump(data_pipeline, SPLITS_DIR / "data_pipeline_v1.joblib")

