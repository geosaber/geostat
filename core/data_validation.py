# -*- coding: utf-8 -*-
"""
Data Validation Module
=======================
Input data QA/QC: duplicate detection, null checks, outlier flagging,
CRS validation. Ensures data quality before geostatistical analysis.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Dict, Any


def check_duplicates(
    coords: np.ndarray,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int]:
    """Detect duplicate coordinates.

    Args:
        coords: (N, 2) or (N, 3) array of XY(Z) coordinates.
        tolerance: Distance below which two points are considered duplicates.

    Returns:
        Tuple of (boolean mask for duplicates, count of duplicates).
    """
    n = coords.shape[0]
    is_dup = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dup[i]:
            continue
        diffs = np.linalg.norm(coords[i + 1 :] - coords[i], axis=1)
        dup_indices = np.where(diffs < tolerance)[0] + i + 1
        is_dup[dup_indices] = True

    return is_dup, int(np.sum(is_dup))


def check_null_values(values: np.ndarray) -> Tuple[np.ndarray, int]:
    """Identify null/NaN values.

    Args:
        values: 1D array of Z values.

    Returns:
        Tuple of (boolean mask for nulls, count of nulls).
    """
    mask = np.isnan(values) | np.isinf(values)
    return mask, int(np.sum(mask))


def detect_outliers(
    values: np.ndarray,
    method: str = "iqr",
    factor: float = 1.5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Detect statistical outliers.

    Args:
        values: 1D array of Z values (NaN-free).
        method: Detection method ('iqr' for Inter-Quartile Range).
        factor: IQR multiplier (1.5 = mild, 3.0 = extreme).

    Returns:
        Tuple of (boolean mask for outliers, stats dict with Q1/Q3/bounds).
    """
    clean = values[~np.isnan(values)]
    q1 = float(np.percentile(clean, 25))
    q3 = float(np.percentile(clean, 75))
    iqr = q3 - q1

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    outlier_mask = (values < lower) | (values > upper)

    stats = {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_outliers": int(np.sum(outlier_mask)),
    }
    return outlier_mask, stats


def compute_descriptive_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute descriptive statistics for the Z variable.

    Args:
        values: 1D array of Z values (NaN-free).

    Returns:
        Dictionary of descriptive statistics.
    """
    clean = values[~np.isnan(values)]
    return {
        "count": len(clean),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "min": float(np.min(clean)),
        "Q1": float(np.percentile(clean, 25)),
        "median": float(np.median(clean)),
        "Q3": float(np.percentile(clean, 75)),
        "max": float(np.max(clean)),
        "skewness": float(_skewness(clean)),
        "kurtosis": float(_kurtosis(clean)),
        "cv": float(np.std(clean, ddof=1) / np.mean(clean)) if np.mean(clean) != 0 else 0.0,
    }


def validate_projected_crs(crs_wkt: str) -> bool:
    """Check if a CRS is projected (not geographic).

    Args:
        crs_wkt: CRS in WKT format.

    Returns:
        True if projected, False if geographic or invalid.
    """
    # Simple heuristic: projected CRS contain 'PROJCS' or 'PROJCRS'
    upper = crs_wkt.upper()
    return "PROJCS" in upper or "PROJCRS" in upper


def validate_input_data(
    coords: np.ndarray,
    values: np.ndarray,
    crs_wkt: str = "",
) -> Dict[str, Any]:
    """Run full validation pipeline on input data.

    Args:
        coords: (N, 2) array of XY coordinates.
        values: (N,) array of Z values.
        crs_wkt: CRS in WKT format (optional).

    Returns:
        Dictionary with validation results and cleaned data.
    """
    results: Dict[str, Any] = {"warnings": [], "errors": []}

    # Null values
    null_mask, n_nulls = check_null_values(values)
    if n_nulls > 0:
        results["warnings"].append(f"{n_nulls} null/NaN values found and will be removed.")

    # Remove nulls
    valid_mask = ~null_mask
    clean_coords = coords[valid_mask]
    clean_values = values[valid_mask]

    # Duplicates
    dup_mask, n_dups = check_duplicates(clean_coords)
    if n_dups > 0:
        results["warnings"].append(
            f"{n_dups} duplicate coordinates detected. "
            f"Consider averaging or removing duplicates."
        )

    # Outliers
    outlier_mask, outlier_stats = detect_outliers(clean_values)
    if outlier_stats["n_outliers"] > 0:
        results["warnings"].append(
            f"{outlier_stats['n_outliers']} statistical outliers detected "
            f"(IQR method, bounds: [{outlier_stats['lower_bound']:.4f}, "
            f"{outlier_stats['upper_bound']:.4f}])."
        )

    # CRS check
    if crs_wkt:
        if not validate_projected_crs(crs_wkt):
            results["warnings"].append(
                "CRS appears to be geographic (lat/lon). "
                "Geostatistical distances require a projected CRS. "
                "Consider reprojecting your data."
            )

    # Descriptive stats
    results["stats"] = compute_descriptive_stats(clean_values)
    results["outlier_stats"] = outlier_stats
    results["n_valid"] = len(clean_values)
    results["n_removed"] = n_nulls
    results["clean_coords"] = clean_coords
    results["clean_values"] = clean_values
    results["duplicate_mask"] = dup_mask
    results["outlier_mask"] = outlier_mask

    if len(clean_values) < 10:
        results["errors"].append(
            f"Only {len(clean_values)} valid samples. "
            f"A minimum of 10 is required for variogram estimation."
        )

    return results


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _skewness(x: np.ndarray) -> float:
    """Fisher-Pearson skewness coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher definition)."""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    k = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(((x - m) / s) ** 4)
    correction = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return float(k - correction)
