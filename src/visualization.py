"""Közös vizualizációs függvények a riporthoz."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR


def _save_or_show(fig: plt.Figure, save_path: Path | str | None) -> None:
    """Helper: ha van path, ment, különben megjelenít."""
    if save_path is not None:
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = FIGURES_DIR / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path | str | None = None,
    sample_size: int = 5000,
) -> plt.Figure:
    """Predicted vs actual scatter plot az ideális y=x vonallal.

    Nagy adatnál (>sample_size) random subsample-t használ a megjeleníthetőség miatt.
    """
    if len(y_true) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_true), size=sample_size, replace=False)
        y_true_plot = np.asarray(y_true)[idx]
        y_pred_plot = np.asarray(y_pred)[idx]
    else:
        y_true_plot = np.asarray(y_true)
        y_pred_plot = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=8, edgecolors="none")

    lo = min(y_true_plot.min(), y_pred_plot.min())
    hi = max(y_true_plot.max(), y_pred_plot.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Ideális (y = x)")

    ax.set_xlabel("Valós időtartam (perc)")
    ax.set_ylabel("Predikált időtartam (perc)")
    ax.set_title(f"Predikció vs valós érték — {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    _save_or_show(fig, save_path)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path | str | None = None,
    sample_size: int = 5000,
) -> plt.Figure:
    """Reziduum-plot: a hibák eloszlása a predikált érték függvényében."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    if len(residuals) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(residuals), size=sample_size, replace=False)
        y_pred_plot = np.asarray(y_pred)[idx]
        residuals_plot = residuals[idx]
    else:
        y_pred_plot = np.asarray(y_pred)
        residuals_plot = residuals

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reziduum scatter
    axes[0].scatter(y_pred_plot, residuals_plot, alpha=0.3, s=8, edgecolors="none")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predikált időtartam (perc)")
    axes[0].set_ylabel("Reziduum (valós - predikált)")
    axes[0].set_title("Reziduumok eloszlása")
    axes[0].grid(alpha=0.3)

    # Reziduum hisztogram
    axes[1].hist(residuals, bins=60, edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Reziduum (perc)")
    axes[1].set_ylabel("Gyakoriság")
    axes[1].set_title("Reziduumok hisztogramja")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Reziduum-elemzés — {model_name}", fontsize=13)
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    model_name: str,
    top_n: int = 20,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Feature importance bar plot (pl. döntési fa vagy Random Forest)."""
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(importance_df))))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top {top_n} feature — {model_name}")
    ax.grid(alpha=0.3, axis="x")

    _save_or_show(fig, save_path)
    return fig


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    metric: str = "mae",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Modellek összehasonlítása egy adott metrika szerint.

    A metrics_df-et a load_metrics() adja vissza.
    """
    sorted_df = metrics_df.sort_values(metric, ascending=(metric != "r2"))

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(sorted_df))))
    ax.barh(sorted_df["model_name"], sorted_df[metric])
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Modellek összehasonlítása — {metric.upper()}")
    ax.grid(alpha=0.3, axis="x")

    # Értékek a barokra
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        ax.text(row[metric], i, f"  {row[metric]:.3f}", va="center", fontsize=9)

    _save_or_show(fig, save_path)
    return fig
