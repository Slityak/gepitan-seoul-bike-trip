"""Modell-agnosztikus kiértékelés és metrika-gyűjtés."""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import METRICS_CSV, RESULTS_DIR


@dataclass
class EvaluationResult:
    """Egy modell kiértékelésének eredménye."""

    model_name: str
    mae: float
    rmse: float
    r2: float
    train_time_sec: float
    predict_time_sec: float
    n_train: int
    n_test: int
    notes: str = ""


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Standard regressziós metrikák kiszámítása."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    model_name: str,
    notes: str = "",
    fit: bool = True,
) -> tuple[EvaluationResult, np.ndarray]:
    """Modell-agnosztikus kiértékelés.

    Bármely sklearn-kompatibilis estimator (fit/predict interfész) támogatott:
    DummyRegressor, LinearRegression, KNeighborsRegressor, DecisionTreeRegressor,
    RandomForestRegressor, GridSearchCV-vel becsomagolt modellek, stb.

    Args:
        model: sklearn-kompatibilis regressor (.fit / .predict)
        X_train, y_train: tanuló adat
        X_test, y_test: teszt adat
        model_name: emberi olvasású név a riportba (pl. "Decision Tree (tuned)")
        notes: opcionális jegyzet (pl. "max_depth=15, min_samples_leaf=20")
        fit: ha False, a modell már be van tanítva (pl. GridSearchCV.best_estimator_)

    Returns:
        (EvaluationResult, predikciók a teszt seten)
    """
    if fit:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
    else:
        train_time = float("nan")

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    metrics = compute_metrics(np.asarray(y_test), y_pred)

    result = EvaluationResult(
        model_name=model_name,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        r2=metrics["r2"],
        train_time_sec=train_time,
        predict_time_sec=predict_time,
        n_train=len(X_train),
        n_test=len(X_test),
        notes=notes,
    )
    return result, y_pred


def append_metrics(result: EvaluationResult, csv_path: Path = METRICS_CSV) -> None:
    """Egy kiértékelés sorát hozzáfűzi a közös metrika-CSV-hez.

    Idempotensen: ha már van ilyen model_name + notes kombináció, felülírja.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([asdict(result)])

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        # Felülírás, ha már van ugyanilyen név + notes kombináció
        mask = (existing["model_name"] == result.model_name) & (
            existing["notes"] == result.notes
        )
        existing = existing[~mask]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(csv_path, index=False)


def load_metrics(csv_path: Path = METRICS_CSV) -> pd.DataFrame:
    """Beolvassa az összegyűjtött metrikákat. Üres DataFrame-et ad, ha még nincs."""
    if not csv_path.exists():
        return pd.DataFrame(
            columns=[
                "model_name",
                "mae",
                "rmse",
                "r2",
                "train_time_sec",
                "predict_time_sec",
                "n_train",
                "n_test",
                "notes",
            ]
        )
    return pd.read_csv(csv_path)
