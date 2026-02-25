# -*- coding: utf-8 -*-
"""
Variography Module
===================
Experimental variogram computation, theoretical model fitting,
anisotropy detection, and variogram map generation.

Supports Spherical, Exponential, Gaussian, and Linear models.
Handles omnidirectional and directional variograms.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform


# ======================================================================
# Theoretical variogram models
# ======================================================================

def spherical_model(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """Spherical variogram model.

    γ(h) = nugget + (sill - nugget) * [1.5*(h/a) - 0.5*(h/a)³]  for h <= a
    γ(h) = sill                                                    for h > a
    """
    h = np.asarray(h, dtype=float)
    result = np.full_like(h, sill)
    mask = h <= range_
    hr = h[mask] / range_
    result[mask] = nugget + (sill - nugget) * (1.5 * hr - 0.5 * hr ** 3)
    result[h == 0] = 0.0
    return result


def exponential_model(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """Exponential variogram model.

    γ(h) = nugget + (sill - nugget) * [1 - exp(-3h/a)]
    """
    h = np.asarray(h, dtype=float)
    result = nugget + (sill - nugget) * (1.0 - np.exp(-3.0 * h / range_))
    result[h == 0] = 0.0
    return result


def gaussian_model(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """Gaussian variogram model.

    γ(h) = nugget + (sill - nugget) * [1 - exp(-3(h/a)²)]
    """
    h = np.asarray(h, dtype=float)
    result = nugget + (sill - nugget) * (1.0 - np.exp(-3.0 * (h / range_) ** 2))
    result[h == 0] = 0.0
    return result


def linear_model(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    """Linear variogram model (bounded).

    γ(h) = nugget + (sill - nugget) * (h / a)  for h <= a
    γ(h) = sill                                 for h > a
    """
    h = np.asarray(h, dtype=float)
    result = np.full_like(h, sill)
    mask = h <= range_
    result[mask] = nugget + (sill - nugget) * (h[mask] / range_)
    result[h == 0] = 0.0
    return result


MODEL_FUNCTIONS = {
    "spherical": spherical_model,
    "exponential": exponential_model,
    "gaussian": gaussian_model,
    "linear": linear_model,
}


# ======================================================================
# VariogramModel dataclass
# ======================================================================

@dataclass
class VariogramModel:
    """Stores fitted variogram model parameters.

    Serializable to/from JSON for inter-module communication.
    """
    model_type: str = "spherical"
    nugget: float = 0.0
    sill: float = 1.0
    range_: float = 1.0
    direction: float = 0.0          # Azimuth in degrees (0=N, 90=E)
    aniso_ratio: float = 1.0        # ratio minor/major axis (1 = isotropic)
    is_anisotropic: bool = False

    # Experimental data (stored for plotting / reporting)
    lags: List[float] = field(default_factory=list)
    semivariance: List[float] = field(default_factory=list)
    pair_counts: List[int] = field(default_factory=list)
    fit_rmse: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        """Serialize model to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, filepath: str) -> "VariogramModel":
        """Deserialize model from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def evaluate(self, h: np.ndarray) -> np.ndarray:
        """Evaluate the theoretical model at distances h."""
        func = MODEL_FUNCTIONS[self.model_type]
        return func(h, self.nugget, self.sill, self.range_)


# ======================================================================
# Experimental variogram computation
# ======================================================================

def compute_experimental_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    lag_distance: Optional[float] = None,
    max_distance: Optional[float] = None,
    n_lags: int = 15,
    direction: Optional[float] = None,
    tolerance: float = 22.5,
    bandwidth: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute experimental (semi)variogram.

    Args:
        coords: (N, 2) array of XY coordinates.
        values: (N,) array of Z values.
        lag_distance: Bin width. If None, auto-estimated.
        max_distance: Maximum lag distance. If None, auto-estimated.
        n_lags: Number of lag bins (used if lag_distance is None).
        direction: Azimuth in degrees for directional variogram (None = omnidirectional).
        tolerance: Angular tolerance in degrees for directional variogram.
        bandwidth: Maximum perpendicular distance for directional variogram.

    Returns:
        Tuple of (lag_centers, semivariances, pair_counts) arrays.
    """
    n = coords.shape[0]

    # Compute pairwise distances and squared differences
    dist_matrix = squareform(pdist(coords))
    diff_matrix = np.subtract.outer(values, values)
    sq_diff = 0.5 * diff_matrix ** 2

    # Auto-estimate parameters
    if max_distance is None:
        max_distance = np.max(dist_matrix) / 2.0

    if lag_distance is None:
        lag_distance = max_distance / n_lags

    # Directional filtering mask
    if direction is not None:
        angle_mask = _directional_mask(coords, direction, tolerance, bandwidth)
    else:
        angle_mask = np.ones((n, n), dtype=bool)

    # Upper triangle only (avoid double-counting)
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_mask = upper_tri & angle_mask

    # Bin edges
    bins = np.arange(0, max_distance + lag_distance, lag_distance)
    n_bins = len(bins) - 1

    lag_centers = np.zeros(n_bins)
    semivariances = np.zeros(n_bins)
    pair_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        bin_mask = valid_mask & (dist_matrix >= bins[i]) & (dist_matrix < bins[i + 1])
        count = np.sum(bin_mask)
        if count > 0:
            lag_centers[i] = np.mean(dist_matrix[bin_mask])
            semivariances[i] = np.mean(sq_diff[bin_mask])
            pair_counts[i] = count
        else:
            lag_centers[i] = (bins[i] + bins[i + 1]) / 2.0
            semivariances[i] = np.nan
            pair_counts[i] = 0

    # Remove empty bins
    valid = pair_counts > 0
    return lag_centers[valid], semivariances[valid], pair_counts[valid]


def _directional_mask(
    coords: np.ndarray,
    direction: float,
    tolerance: float,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """Create directional filter mask for variogram computation.

    Args:
        coords: (N, 2) array of coordinates.
        direction: Azimuth in degrees (0=N, 90=E, clockwise from north).
        tolerance: Angular tolerance in degrees.
        bandwidth: Max perpendicular distance (None = no bandwidth filter).

    Returns:
        (N, N) boolean mask.
    """
    n = coords.shape[0]
    dx = coords[:, 0].reshape(1, -1) - coords[:, 0].reshape(-1, 1)
    dy = coords[:, 1].reshape(1, -1) - coords[:, 1].reshape(-1, 1)

    # Convert azimuth to mathematical angle
    # Azimuth: 0=N, 90=E → Math angle: 90-azimuth
    angles = np.degrees(np.arctan2(dx, dy)) % 360.0
    dir_mod = direction % 360.0

    # Angular difference (considering opposite direction for variogram symmetry)
    diff1 = np.abs(angles - dir_mod)
    diff2 = np.abs(angles - (dir_mod + 180.0) % 360.0)
    ang_diff = np.minimum(diff1, diff2)
    ang_diff = np.minimum(ang_diff, 360.0 - ang_diff)

    mask = ang_diff <= tolerance

    if bandwidth is not None:
        dir_rad = np.radians(90.0 - direction)
        perp_dist = np.abs(dx * np.sin(dir_rad) - dy * np.cos(dir_rad))
        mask = mask & (perp_dist <= bandwidth)

    return mask


# ======================================================================
# Variogram Map (for anisotropy detection)
# ======================================================================

def compute_variogram_map(
    coords: np.ndarray,
    values: np.ndarray,
    grid_size: int = 50,
    max_distance: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2D variogram map for anisotropy detection.

    The map shows semivariance as a function of lag vector (dx, dy)
    in a 2D grid.

    Args:
        coords: (N, 2) array of XY coordinates.
        values: (N,) array of Z values.
        grid_size: Number of cells per dimension (grid_size × grid_size).
        max_distance: Maximum lag distance to consider.

    Returns:
        Tuple of (variogram_map, x_edges, y_edges).
        variogram_map: (grid_size, grid_size) array of semivariances.
    """
    n = coords.shape[0]

    if max_distance is None:
        max_distance = np.max(pdist(coords)) / 2.0

    # Create grid edges
    edges = np.linspace(-max_distance, max_distance, grid_size + 1)

    vario_map = np.full((grid_size, grid_size), np.nan)
    count_map = np.zeros((grid_size, grid_size), dtype=int)
    sum_map = np.zeros((grid_size, grid_size))

    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[j, 0] - coords[i, 0]
            dy = coords[j, 1] - coords[i, 1]

            if abs(dx) > max_distance or abs(dy) > max_distance:
                continue

            sq_diff = 0.5 * (values[j] - values[i]) ** 2

            # Find bin for (dx, dy)
            ix = np.searchsorted(edges, dx) - 1
            iy = np.searchsorted(edges, dy) - 1
            if 0 <= ix < grid_size and 0 <= iy < grid_size:
                sum_map[iy, ix] += sq_diff
                count_map[iy, ix] += 1

            # Symmetric: also add (-dx, -dy)
            ix2 = np.searchsorted(edges, -dx) - 1
            iy2 = np.searchsorted(edges, -dy) - 1
            if 0 <= ix2 < grid_size and 0 <= iy2 < grid_size:
                sum_map[iy2, ix2] += sq_diff
                count_map[iy2, ix2] += 1

    valid = count_map > 0
    vario_map[valid] = sum_map[valid] / count_map[valid]

    return vario_map, edges, edges


# ======================================================================
# Anisotropy Detection
# ======================================================================

def detect_anisotropy(
    coords: np.ndarray,
    values: np.ndarray,
    directions: Optional[List[float]] = None,
    lag_distance: Optional[float] = None,
    max_distance: Optional[float] = None,
    n_lags: int = 15,
    ratio_threshold: float = 0.7,
) -> Dict[str, Any]:
    """Detect anisotropy by computing directional variograms.

    Estimates the major direction, range ratio, and classifies
    isotropic vs anisotropic behavior.

    Args:
        coords: (N, 2) array of XY coordinates.
        values: (N,) array of Z values.
        directions: List of azimuths to test (default: [0, 45, 90, 135]).
        lag_distance: Bin width.
        max_distance: Maximum lag distance.
        n_lags: Number of lag bins.
        ratio_threshold: Below this major/minor ratio → anisotropic.

    Returns:
        Dictionary with anisotropy results:
        - is_anisotropic: bool
        - major_direction: float (azimuth of max continuity)
        - minor_direction: float (perpendicular)
        - aniso_ratio: float (range_minor / range_major)
        - directional_variograms: dict of direction → (lags, semivariance)
        - ellipse_params: dict with center, major_axis, minor_axis, angle
    """
    if directions is None:
        directions = [0.0, 45.0, 90.0, 135.0]

    dir_ranges: Dict[float, float] = {}
    dir_variograms: Dict[float, Dict[str, Any]] = {}

    for azimuth in directions:
        lags, semiv, counts = compute_experimental_variogram(
            coords, values,
            lag_distance=lag_distance,
            max_distance=max_distance,
            n_lags=n_lags,
            direction=azimuth,
            tolerance=22.5,
        )

        if len(lags) < 3:
            continue

        # Estimate effective range: distance at which semivariance
        # reaches ~95% of the overall sill
        overall_sill = float(np.nanmax(semiv))
        threshold = 0.95 * overall_sill

        above = np.where(semiv >= threshold)[0]
        if len(above) > 0:
            effective_range = float(lags[above[0]])
        else:
            effective_range = float(lags[-1])

        dir_ranges[azimuth] = effective_range
        dir_variograms[azimuth] = {
            "lags": lags.tolist(),
            "semivariance": semiv.tolist(),
            "counts": counts.tolist(),
            "effective_range": effective_range,
        }

    if len(dir_ranges) < 2:
        return {
            "is_anisotropic": False,
            "major_direction": 0.0,
            "minor_direction": 90.0,
            "aniso_ratio": 1.0,
            "directional_variograms": dir_variograms,
            "ellipse_params": None,
        }

    # Direction with maximum range = major axis (max continuity)
    major_dir = max(dir_ranges, key=dir_ranges.get)
    major_range = dir_ranges[major_dir]

    # Direction with minimum range = minor axis
    minor_dir = min(dir_ranges, key=dir_ranges.get)
    minor_range = dir_ranges[minor_dir]

    aniso_ratio = minor_range / major_range if major_range > 0 else 1.0
    is_aniso = aniso_ratio < ratio_threshold

    # Ellipse parameters for overlay visualization
    ellipse_params = {
        "center_x": 0.0,
        "center_y": 0.0,
        "major_axis": major_range,
        "minor_axis": minor_range,
        "angle": major_dir,  # azimuth of major axis
    }

    return {
        "is_anisotropic": is_aniso,
        "major_direction": float(major_dir),
        "minor_direction": float(minor_dir),
        "aniso_ratio": float(aniso_ratio),
        "directional_variograms": dir_variograms,
        "ellipse_params": ellipse_params,
    }


# ======================================================================
# Parameter auto-estimation
# ======================================================================

def auto_estimate_parameters(
    coords: np.ndarray,
    values: np.ndarray,
) -> Dict[str, float]:
    """Auto-estimate initial variogram parameters from data.

    Heuristics:
    - lag_distance ≈ median nearest-neighbor distance
    - max_distance ≈ max pairwise distance / 2
    - nugget ≈ 0 (or small fraction of variance)
    - sill ≈ sample variance
    - range ≈ max_distance / 3

    Args:
        coords: (N, 2) array of XY coordinates.
        values: (N,) array of Z values.

    Returns:
        Dictionary of estimated parameters.
    """
    dist_full = squareform(pdist(coords))
    np.fill_diagonal(dist_full, np.inf)
    nn_distances = np.min(dist_full, axis=1)

    lag_distance = float(np.median(nn_distances))
    max_dist = float(np.max(pdist(coords)))
    max_distance = max_dist / 2.0

    variance = float(np.var(values, ddof=1))

    return {
        "lag_distance": lag_distance,
        "max_distance": max_distance,
        "nugget": 0.0,
        "sill": variance,
        "range": max_distance / 3.0,
        "n_lags": max(5, min(20, int(max_distance / lag_distance))),
    }


# ======================================================================
# Theoretical model fitting
# ======================================================================

def fit_theoretical_model(
    lags: np.ndarray,
    semivariance: np.ndarray,
    pair_counts: np.ndarray,
    model_type: Optional[str] = None,
    initial_nugget: Optional[float] = None,
    initial_sill: Optional[float] = None,
    initial_range: Optional[float] = None,
) -> VariogramModel:
    """Fit a theoretical variogram model to experimental data.

    If model_type is None, tries all models and picks the best fit (lowest RMSE).

    Args:
        lags: Array of lag distances.
        semivariance: Array of experimental semivariances.
        pair_counts: Array of pair counts per lag.
        model_type: Model to fit. None → auto-select best.
        initial_nugget: Initial guess for nugget (None → 0).
        initial_sill: Initial guess for sill (None → max semivariance).
        initial_range: Initial guess for range (None → max lag / 2).

    Returns:
        Fitted VariogramModel instance.
    """
    lags = np.asarray(lags, dtype=float)
    semivariance = np.asarray(semivariance, dtype=float)
    pair_counts = np.asarray(pair_counts, dtype=int)

    # Defaults
    if initial_sill is None:
        initial_sill = float(np.max(semivariance))
    if initial_nugget is None:
        initial_nugget = 0.0
    if initial_range is None:
        initial_range = float(lags[-1]) / 2.0

    # Weights: more pairs = more reliable
    weights = np.sqrt(pair_counts.astype(float))
    weights = weights / np.max(weights)

    models_to_try = [model_type] if model_type else list(MODEL_FUNCTIONS.keys())

    best_model: Optional[VariogramModel] = None
    best_rmse = np.inf

    for mtype in models_to_try:
        model_func = MODEL_FUNCTIONS[mtype]

        try:
            popt, _ = curve_fit(
                model_func,
                lags,
                semivariance,
                p0=[initial_nugget, initial_sill, initial_range],
                bounds=(
                    [0.0, 0.0, 1e-6],
                    [initial_sill * 2, initial_sill * 3, float(lags[-1]) * 3],
                ),
                sigma=1.0 / (weights + 1e-10),
                maxfev=5000,
            )

            fitted_nugget, fitted_sill, fitted_range = popt
            predicted = model_func(lags, *popt)
            rmse = float(np.sqrt(np.mean((semivariance - predicted) ** 2)))

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = VariogramModel(
                    model_type=mtype,
                    nugget=float(fitted_nugget),
                    sill=float(fitted_sill),
                    range_=float(fitted_range),
                    lags=lags.tolist(),
                    semivariance=semivariance.tolist(),
                    pair_counts=pair_counts.tolist(),
                    fit_rmse=rmse,
                )

        except (RuntimeError, ValueError):
            # curve_fit failed for this model → skip
            continue

    if best_model is None:
        # Fallback: return a default model with initial guesses
        best_model = VariogramModel(
            model_type="spherical",
            nugget=initial_nugget,
            sill=initial_sill,
            range_=initial_range,
            lags=lags.tolist(),
            semivariance=semivariance.tolist(),
            pair_counts=pair_counts.tolist(),
            fit_rmse=0.0,
        )

    return best_model
