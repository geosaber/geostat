# -*- coding: utf-8 -*-
"""
Kriging Engine
===============
Kriging interpolation methods using PyKrige as backend.

Supported methods:
- Ordinary Kriging (OK)
- Universal Kriging (UK) with polynomial drift
- External Drift Kriging (EDK)
- Indicator Kriging (IK)

CoKriging is deferred to phase 2.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any

from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from .variography import VariogramModel


# ======================================================================
# Kriging Parameter Mapping
# ======================================================================

def _pykrige_model_name(model_type: str) -> str:
    """Map internal model names to PyKrige names."""
    mapping = {
        "spherical": "spherical",
        "exponential": "exponential",
        "gaussian": "gaussian",
        "linear": "linear",
    }
    return mapping.get(model_type, "spherical")


def _build_variogram_parameters(model: VariogramModel) -> Dict[str, Any]:
    """Build PyKrige variogram parameters from VariogramModel.

    PyKrige expects variogram_parameters as dict:
        {'sill': s, 'range': r, 'nugget': n}
    or a list [sill-nugget, range, nugget] depending on version.
    """
    return {
        "sill": model.sill,
        "range": model.range_,
        "nugget": model.nugget,
    }


# ======================================================================
# Ordinary Kriging
# ======================================================================

def ordinary_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    model: VariogramModel,
    anisotropy_scaling: float = 1.0,
    anisotropy_angle: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Ordinary Kriging interpolation.

    Args:
        coords: (N, 2) array of sample XY coordinates.
        values: (N,) array of sample Z values.
        grid_x: 1D array of X grid coordinates.
        grid_y: 1D array of Y grid coordinates.
        model: Fitted VariogramModel.
        anisotropy_scaling: Scaling ratio for anisotropy (minor/major).
        anisotropy_angle: Angle of anisotropy in degrees (clockwise from N).

    Returns:
        Tuple of (kriged_values, kriging_variance) as 2D arrays.
    """
    ok = OrdinaryKriging(
        x=coords[:, 0],
        y=coords[:, 1],
        z=values,
        variogram_model=_pykrige_model_name(model.model_type),
        variogram_parameters=_build_variogram_parameters(model),
        anisotropy_scaling=anisotropy_scaling if model.is_anisotropic else 1.0,
        anisotropy_angle=anisotropy_angle if model.is_anisotropic else 0.0,
        verbose=False,
        enable_plotting=False,
    )

    z_pred, ss_pred = ok.execute("grid", grid_x, grid_y)
    return np.asarray(z_pred), np.asarray(ss_pred)


# ======================================================================
# Universal Kriging
# ======================================================================

def universal_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    model: VariogramModel,
    drift_terms: str = "regional_linear",
    anisotropy_scaling: float = 1.0,
    anisotropy_angle: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Universal Kriging with polynomial drift.

    Args:
        coords: (N, 2) array of sample XY coordinates.
        values: (N,) array of sample Z values.
        grid_x: 1D array of X grid coordinates.
        grid_y: 1D array of Y grid coordinates.
        model: Fitted VariogramModel.
        drift_terms: Drift specification for PyKrige:
            'regional_linear', 'specified', or list of custom terms.
        anisotropy_scaling: Scaling ratio for anisotropy.
        anisotropy_angle: Angle of anisotropy in degrees.

    Returns:
        Tuple of (kriged_values, kriging_variance) as 2D arrays.
    """
    uk = UniversalKriging(
        x=coords[:, 0],
        y=coords[:, 1],
        z=values,
        variogram_model=_pykrige_model_name(model.model_type),
        variogram_parameters=_build_variogram_parameters(model),
        drift_terms=[drift_terms],
        anisotropy_scaling=anisotropy_scaling if model.is_anisotropic else 1.0,
        anisotropy_angle=anisotropy_angle if model.is_anisotropic else 0.0,
        verbose=False,
        enable_plotting=False,
    )

    z_pred, ss_pred = uk.execute("grid", grid_x, grid_y)
    return np.asarray(z_pred), np.asarray(ss_pred)


# ======================================================================
# External Drift Kriging
# ======================================================================

def external_drift_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    model: VariogramModel,
    drift_values_at_samples: np.ndarray,
    drift_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform External Drift Kriging (KED).

    Uses an external variable (e.g., elevation raster) as drift function.

    Args:
        coords: (N, 2) array of sample XY coordinates.
        values: (N,) array of sample Z values.
        grid_x: 1D array of X grid coordinates.
        grid_y: 1D array of Y grid coordinates.
        model: Fitted VariogramModel.
        drift_values_at_samples: (N,) array of drift variable at sample locations.
        drift_grid: (ny, nx) array of drift variable on the prediction grid.

    Returns:
        Tuple of (kriged_values, kriging_variance) as 2D arrays.
    """
    uk = UniversalKriging(
        x=coords[:, 0],
        y=coords[:, 1],
        z=values,
        variogram_model=_pykrige_model_name(model.model_type),
        variogram_parameters=_build_variogram_parameters(model),
        drift_terms=["specified"],
        specified_drift=[drift_values_at_samples],
        verbose=False,
        enable_plotting=False,
    )

    # For EDK, we need to pass the drift grid as a specified drift
    z_pred, ss_pred = uk.execute(
        "grid", grid_x, grid_y,
        specified_drift_arrays=[drift_grid],
    )
    return np.asarray(z_pred), np.asarray(ss_pred)


# ======================================================================
# Indicator Kriging
# ======================================================================

def indicator_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    model: VariogramModel,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Indicator Kriging.

    Transforms data into binary indicators (above/below cutoff),
    then applies Ordinary Kriging on the indicator variable.

    Args:
        coords: (N, 2) array of sample XY coordinates.
        values: (N,) array of sample Z values.
        grid_x: 1D array of X grid coordinates.
        grid_y: 1D array of Y grid coordinates.
        model: Fitted VariogramModel.
        cutoff: Threshold value for indicator transformation.

    Returns:
        Tuple of (probability_grid, kriging_variance) as 2D arrays.
        probability_grid contains P(Z > cutoff) at each grid cell.
    """
    # Transform to indicator: 1 if value >= cutoff, 0 otherwise
    indicators = (values >= cutoff).astype(float)

    # Refit variogram on indicator data for better results
    # Use the model structure but OK to use passed model
    ok = OrdinaryKriging(
        x=coords[:, 0],
        y=coords[:, 1],
        z=indicators,
        variogram_model=_pykrige_model_name(model.model_type),
        variogram_parameters=_build_variogram_parameters(model),
        verbose=False,
        enable_plotting=False,
    )

    z_pred, ss_pred = ok.execute("grid", grid_x, grid_y)

    # Clip probabilities to [0, 1]
    z_pred = np.clip(z_pred, 0.0, 1.0)

    return np.asarray(z_pred), np.asarray(ss_pred)


# ======================================================================
# Dispatcher
# ======================================================================

def run_kriging(
    method: str,
    coords: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    model: VariogramModel,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to the appropriate kriging method.

    Args:
        method: One of 'ordinary', 'universal', 'external_drift', 'indicator'.
        coords: (N, 2) array of sample XY coordinates.
        values: (N,) array of sample Z values.
        grid_x: 1D array of X grid coordinates.
        grid_y: 1D array of Y grid coordinates.
        model: Fitted VariogramModel.
        **kwargs: Additional arguments for specific methods.

    Returns:
        Tuple of (kriged_values, kriging_variance) as 2D arrays.
    """
    aniso_scaling = model.aniso_ratio if model.is_anisotropic else 1.0
    aniso_angle = model.direction if model.is_anisotropic else 0.0

    if method == "ordinary":
        return ordinary_kriging(
            coords, values, grid_x, grid_y, model,
            anisotropy_scaling=aniso_scaling,
            anisotropy_angle=aniso_angle,
        )
    elif method == "universal":
        drift = kwargs.get("drift_terms", "regional_linear")
        return universal_kriging(
            coords, values, grid_x, grid_y, model,
            drift_terms=drift,
            anisotropy_scaling=aniso_scaling,
            anisotropy_angle=aniso_angle,
        )
    elif method == "external_drift":
        return external_drift_kriging(
            coords, values, grid_x, grid_y, model,
            drift_values_at_samples=kwargs["drift_values_at_samples"],
            drift_grid=kwargs["drift_grid"],
        )
    elif method == "indicator":
        return indicator_kriging(
            coords, values, grid_x, grid_y, model,
            cutoff=kwargs.get("cutoff", np.median(values)),
        )
    else:
        raise ValueError(f"Unknown kriging method: {method}")
