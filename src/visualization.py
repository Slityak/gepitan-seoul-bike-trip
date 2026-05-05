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


def plot_target_distribution(
    y: np.ndarray | pd.Series,
    save_path: Path | str | None = None,
    bins: int = 60,
    title: str = "Target eloszlás (y_train)",
) -> plt.Figure:
    """A target hisztogramja, median + mean függőleges vonallal."""
    y_arr = np.asarray(y)
    median = float(np.median(y_arr))
    mean = float(np.mean(y_arr))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(y_arr, bins=bins, edgecolor="black", alpha=0.7)
    ax.axvline(median, color="red", linestyle="--", linewidth=1.5,
               label=f"median = {median:.0f}")
    ax.axvline(mean, color="orange", linestyle="--", linewidth=1.5,
               label=f"mean = {mean:.1f}")
    ax.set_xlabel("Időtartam (perc)")
    ax.set_ylabel("Gyakoriság")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    _save_or_show(fig, save_path)
    return fig


def plot_validation_curve(
    param_range_labels: list[str],
    train_mae: np.ndarray,
    test_mae: np.ndarray,
    *,
    param_name: str = "",
    model_name: str = "",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Validation curve precomputed MAE arrayokból.

    Args:
        param_range_labels: a hiperparaméter értékek string-címkéi (x tengely)
        train_mae: shape (n_params, n_folds) — pozitív MAE
        test_mae: shape (n_params, n_folds) — pozitív MAE
    """
    train_mae = np.asarray(train_mae)
    test_mae = np.asarray(test_mae)
    x = np.arange(len(param_range_labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, train_mae.mean(axis=1), "o-", color="C0", label="Train MAE")
    ax.fill_between(x, train_mae.mean(axis=1) - train_mae.std(axis=1),
                    train_mae.mean(axis=1) + train_mae.std(axis=1), alpha=0.2, color="C0")
    ax.plot(x, test_mae.mean(axis=1), "o-", color="C1", label="CV MAE")
    ax.fill_between(x, test_mae.mean(axis=1) - test_mae.std(axis=1),
                    test_mae.mean(axis=1) + test_mae.std(axis=1), alpha=0.2, color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(param_range_labels)
    ax.set_xlabel(param_name)
    ax.set_ylabel("MAE (perc)")
    title = f"Validation curve — {model_name}" if model_name else "Validation curve"
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    _save_or_show(fig, save_path)
    return fig


def plot_learning_curve(
    sizes: np.ndarray,
    train_mae: np.ndarray,
    test_mae: np.ndarray,
    *,
    model_name: str = "",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Learning curve precomputed MAE arrayokból.

    Args:
        sizes: a tanítóhalmaz méretei (x tengely), shape (n_sizes,)
        train_mae: shape (n_sizes, n_folds) — pozitív MAE
        test_mae: shape (n_sizes, n_folds) — pozitív MAE
    """
    sizes = np.asarray(sizes)
    train_mae = np.asarray(train_mae)
    test_mae = np.asarray(test_mae)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, train_mae.mean(axis=1), "o-", color="C0", label="Train MAE")
    ax.fill_between(sizes, train_mae.mean(axis=1) - train_mae.std(axis=1),
                    train_mae.mean(axis=1) + train_mae.std(axis=1), alpha=0.2, color="C0")
    ax.plot(sizes, test_mae.mean(axis=1), "o-", color="C1", label="CV MAE")
    ax.fill_between(sizes, test_mae.mean(axis=1) - test_mae.std(axis=1),
                    test_mae.mean(axis=1) + test_mae.std(axis=1), alpha=0.2, color="C1")
    ax.set_xlabel("Train size")
    ax.set_ylabel("MAE (perc)")
    title = f"Learning curve — {model_name}" if model_name else "Learning curve"
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    _save_or_show(fig, save_path)
    return fig


def plot_error_by_feature_quantile(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    feature_values: np.ndarray | pd.Series,
    feature_name: str,
    *,
    n_bins: int = 10,
    model_name: str = "",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Átlagos abszolút hiba (MAE) az adott feature kvantilis-binjei szerint.
    Háttérben a binméret oszlopdiagrammal.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    feature_values = np.asarray(feature_values)
    abs_err = np.abs(y_true - y_pred)

    bin_idx = pd.qcut(feature_values, q=n_bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"bin": bin_idx, "abs_err": abs_err})
    grouped = df.groupby("bin")["abs_err"].agg(["mean", "count"]).reset_index()

    bin_labels = [f"{int(b) + 1}/{n_bins}" for b in grouped["bin"]]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(bin_labels, grouped["count"], color="C0", alpha=0.25, edgecolor="C0",
            label="Bin méret")
    ax1.set_xlabel(f"{feature_name} kvantilis")
    ax1.set_ylabel("Bin méret", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(bin_labels, grouped["mean"], "o-", color="C1", linewidth=2, label="MAE")
    ax2.set_ylabel("MAE (perc)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    title = f"Hiba {feature_name} kvantilis szerint"
    if model_name:
        title += f" — {model_name}"
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    fig.tight_layout()

    _save_or_show(fig, save_path)
    return fig


def plot_decision_tree_top(
    tree_model: object,
    feature_names: list[str],
    *,
    max_depth: int = 3,
    save_path: Path | str | None = None,
    model_name: str = "",
) -> plt.Figure:
    """A betanult döntési fa felső szintjeit rajzolja ki (a teljes fa túl nagy)."""
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        max_depth=max_depth,
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=9,
        impurity=False,
    )
    title = f"Decision tree (felső {max_depth} szint)"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title)

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
