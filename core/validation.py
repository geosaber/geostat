# -*- coding: utf-8 -*-
"""
Cross-Validation Module
=========================
Leave-One-Out cross-validation, validation metrics, and diagnostic plots.
Provides auditable proof of model quality for Qualified Person reports.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple, Optional

from .variography import VariogramModel


def leave_one_out_cv(
    coords: np.ndarray,
    values: np.ndarray,
    model: VariogramModel,
    method: str = "ordinary",
    **kriging_kwargs,
) -> Dict[str, Any]:
    """Perform Leave-One-Out Cross-Validation.

    Removes each sample one at a time, performs kriging with the
    remaining samples, and compares predicted vs observed values.

    Args:
        coords: (N, 2) array of sample XY coordinates.
        values: (N,) array of sample Z values.
        model: Fitted VariogramModel.
        method: Kriging method ('ordinary', 'universal', 'indicator').
        **kriging_kwargs: Additional arguments for specific methods.

    Returns:
        Dictionary with:
        - observed: array of observed values
        - predicted: array of predicted values
        - errors: array of (predicted - observed) errors
        - variance: array of kriging variance (if available)
        - metrics: dict of validation metrics
    """
    n = len(values)
    observed = np.zeros(n)
    predicted = np.zeros(n)
    variance = np.zeros(n)

    # Lazy import to avoid requiring pykrige at module load time
    from .kriging_engine import run_kriging

    for i in range(n):
        # Remove the i-th sample
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        train_coords = coords[mask]
        train_values = values[mask]

        # Predict at the removed point
        grid_x = np.array([coords[i, 0]])
        grid_y = np.array([coords[i, 1]])

        try:
            z_pred, ss_pred = run_kriging(
                method=method,
                coords=train_coords,
                values=train_values,
                grid_x=grid_x,
                grid_y=grid_y,
                model=model,
                **kriging_kwargs,
            )
            observed[i] = values[i]
            predicted[i] = z_pred.flatten()[0]
            variance[i] = ss_pred.flatten()[0]
        except Exception:
            observed[i] = values[i]
            predicted[i] = np.nan
            variance[i] = np.nan

    errors = predicted - observed

    # Compute metrics
    metrics = compute_metrics(observed, predicted, variance)

    return {
        "observed": observed,
        "predicted": predicted,
        "errors": errors,
        "variance": variance,
        "metrics": metrics,
    }


def compute_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    variance: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute cross-validation metrics.

    Args:
        observed: Array of observed values.
        predicted: Array of predicted values.
        variance: Array of kriging variance (optional).

    Returns:
        Dictionary of metrics:
        - ME: Mean Error (bias indicator, should be ≈ 0)
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Square Error
        - R2: Coefficient of Determination
        - MSDR: Mean Squared Deviation Ratio (uses kriging variance)
    """
    valid = ~np.isnan(predicted) & ~np.isnan(observed)
    obs = observed[valid]
    pred = predicted[valid]

    errors = pred - obs
    n = len(errors)

    if n == 0:
        return {"ME": np.nan, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MSDR": np.nan}

    me = float(np.mean(errors))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Mean Squared Deviation Ratio
    msdr = np.nan
    if variance is not None:
        var_valid = variance[valid]
        nonzero_var = var_valid > 0
        if np.sum(nonzero_var) > 0:
            msdr = float(np.mean((errors[nonzero_var] ** 2) / var_valid[nonzero_var]))

    return {
        "ME": me,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MSDR": msdr,
    }


def generate_validation_plots(
    observed: np.ndarray,
    predicted: np.ndarray,
    errors: np.ndarray,
    coords: np.ndarray,
) -> Dict[str, Any]:
    """Generate validation diagnostic plots using Matplotlib.

    Returns figure objects that can be embedded in reports or displayed.

    Args:
        observed: Array of observed values.
        predicted: Array of predicted values.
        errors: Array of prediction errors.
        coords: (N, 2) array of sample coordinates.

    Returns:
        Dictionary of matplotlib Figure objects:
        - scatter_fig: Observed vs Predicted
        - histogram_fig: Error histogram
        - spatial_fig: Spatial distribution of errors
    """
    import matplotlib.pyplot as plt

    valid = ~np.isnan(predicted) & ~np.isnan(observed)
    obs = observed[valid]
    pred = predicted[valid]
    err = errors[valid]
    crd = coords[valid]

    # 1. Scatter: Observed vs Predicted
    scatter_fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.scatter(obs, pred, c="#2c3e50", s=30, alpha=0.7, edgecolors="#ecf0f1", linewidths=0.5)
    lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
    ax1.plot(lims, lims, "--", color="#e74c3c", linewidth=1.5, label="1:1 Line")
    ax1.set_xlabel("Observed")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Cross-Validation: Observed vs Predicted")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    scatter_fig.tight_layout()

    # 2. Histogram of errors
    histogram_fig, ax2 = plt.subplots(figsize=(6, 5))
    ax2.hist(err, bins=25, color="#3498db", edgecolor="#2c3e50", alpha=0.8)
    ax2.axvline(0, color="#e74c3c", linewidth=1.5, linestyle="--", label="Zero Error")
    ax2.axvline(np.mean(err), color="#f39c12", linewidth=1.5, linestyle="-", label=f"ME = {np.mean(err):.4f}")
    ax2.set_xlabel("Error (Predicted - Observed)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Cross-Validation Errors")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    histogram_fig.tight_layout()

    # 3. Spatial distribution of errors
    spatial_fig, ax3 = plt.subplots(figsize=(6, 5))
    sc = ax3.scatter(
        crd[:, 0], crd[:, 1],
        c=err, cmap="RdBu_r", s=40,
        edgecolors="#2c3e50", linewidths=0.5,
        vmin=-np.abs(err).max(), vmax=np.abs(err).max(),
    )
    plt.colorbar(sc, ax=ax3, label="Error")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_title("Spatial Distribution of Errors")
    ax3.set_aspect("equal")
    spatial_fig.tight_layout()

    return {
        "scatter_fig": scatter_fig,
        "histogram_fig": histogram_fig,
        "spatial_fig": spatial_fig,
    }
