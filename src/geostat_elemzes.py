import pandas as pd
import numpy as np
import joblib
from src.data_io import load_train_test
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.metrics import r2_score, mean_absolute_error
from config import SPLITS_DIR, RANDOM_SEED

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

# --- 2. LÉPÉS: FEATURE ENGINEERING ---
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

# --- 3. LÉPÉS: PIPELINE (A 9,6M soros feldolgozáshoz)[cite: 1, 5] ---
data_pipeline = Pipeline([
    ('feature_eng', FunctionTransformer(feature_engineering_transform)),
    ('var_threshold', VarianceThreshold(threshold=0.01)),
    ('scaler', StandardScaler()),
    ('feature_select', SelectKBest(score_func=f_regression, k=15))
])

print("Pipeline futtatása a teljes adathalmazon (ez eltarthat egy ideig)...")
X_train_final = data_pipeline.fit_transform(X_train, y_train)
X_test_final = data_pipeline.transform(X_test)

# --- 4. MODELLEZÉS (Optimalizált KNN a nagy adathoz)
# A GridSearchCV-t elhagyjuk, a v2 során talált optimális paramétereket használjuk fixen
print("Modell illesztése (KNN - KD-Tree algoritmus)...")
best_knn = KNeighborsRegressor(
    n_neighbors=15,
    weights='distance',
    algorithm='kd_tree', # Gyorsabb keresés nagy adathalmazon
    n_jobs=-1
)
best_knn.fit(X_train_final, y_train)

# --- 5. KIÉRTÉKELÉS ---
print("Kiértékelés futtatása...")
y_pred = best_knn.predict(X_test_final)
print("\n--- Teljes adatos modell teljesítmény ---")
print(f"R2 Score:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE:       {mean_absolute_error(y_test, y_pred):.4f} perc")

# --- 6. MENTÉS[cite: 3, 6] ---
print("Eredmények mentése...")
np.save(SPLITS_DIR / "X_train_final_9M.npy", X_train_final)
np.save(SPLITS_DIR / "X_test_final_9M.npy", X_test_final)
joblib.dump(data_pipeline, SPLITS_DIR / "data_pipeline_9M.joblib")
joblib.dump(best_knn, SPLITS_DIR / "best_knn_model_9M.joblib")

print(f"\nSiker! A 9,6 millió soros projekt elemei mentve ide: {SPLITS_DIR}")