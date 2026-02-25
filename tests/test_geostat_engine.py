# -*- coding: utf-8 -*-
"""
Unit Tests for the GeoStats Plugin
=====================================
Tests mathematical precision of the geostatistical engine
using synthetic datasets with known properties.
"""

import os
import sys
import json
import tempfile

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.variography import (
    spherical_model,
    exponential_model,
    gaussian_model,
    linear_model,
    compute_experimental_variogram,
    compute_variogram_map,
    detect_anisotropy,
    auto_estimate_parameters,
    fit_theoretical_model,
    VariogramModel,
)
from core.data_validation import (
    check_duplicates,
    check_null_values,
    detect_outliers,
    compute_descriptive_stats,
    validate_input_data,
)
from core.validation import compute_metrics


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def synthetic_data():
    """Generate synthetic 2D point data with known spatial correlation."""
    np.random.seed(42)
    n = 100
    coords = np.random.uniform(0, 1000, (n, 2))
    # Create spatially correlated values using distance-based model
    values = np.zeros(n)
    for i in range(n):
        values[i] = (
            0.5 * coords[i, 0] / 1000
            + 0.3 * coords[i, 1] / 1000
            + np.random.normal(0, 0.1)
        )
    return coords, values


@pytest.fixture
def known_variogram_data():
    """Generate data with known variogram parameters for fitting tests."""
    np.random.seed(123)
    lags = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500], dtype=float)
    # Spherical model with nugget=0.1, sill=1.0, range=300
    true_semiv = spherical_model(lags, 0.1, 1.0, 300.0)
    # Add small noise
    noise = np.random.normal(0, 0.02, len(lags))
    semiv = true_semiv + noise
    counts = np.full(len(lags), 50, dtype=int)
    return lags, semiv, counts


# ======================================================================
# Test: Theoretical Models
# ======================================================================

class TestTheoreticalModels:
    """Test theoretical variogram model functions."""

    def test_spherical_at_zero(self):
        """Spherical model should return 0 at h=0."""
        assert spherical_model(np.array([0.0]), 0.1, 1.0, 100.0)[0] == 0.0

    def test_spherical_at_sill(self):
        """Spherical model should return sill for h > range."""
        result = spherical_model(np.array([200.0]), 0.0, 1.0, 100.0)
        assert result[0] == pytest.approx(1.0)

    def test_exponential_at_zero(self):
        assert exponential_model(np.array([0.0]), 0.1, 1.0, 100.0)[0] == 0.0

    def test_exponential_approaches_sill(self):
        """Exponential should approach sill for large h."""
        result = exponential_model(np.array([1000.0]), 0.0, 1.0, 100.0)
        assert result[0] == pytest.approx(1.0, abs=0.01)

    def test_gaussian_at_zero(self):
        assert gaussian_model(np.array([0.0]), 0.1, 1.0, 100.0)[0] == 0.0

    def test_gaussian_approaches_sill(self):
        result = gaussian_model(np.array([500.0]), 0.0, 1.0, 100.0)
        assert result[0] == pytest.approx(1.0, abs=0.01)

    def test_linear_at_zero(self):
        assert linear_model(np.array([0.0]), 0.1, 1.0, 100.0)[0] == 0.0

    def test_linear_at_range(self):
        result = linear_model(np.array([100.0]), 0.0, 1.0, 100.0)
        assert result[0] == pytest.approx(1.0)

    def test_models_monotonic(self):
        """All models should be monotonically non-decreasing."""
        h = np.linspace(0.1, 500, 50)
        for model_func in [spherical_model, exponential_model, gaussian_model, linear_model]:
            gamma = model_func(h, 0.0, 1.0, 200.0)
            diffs = np.diff(gamma)
            assert np.all(diffs >= -1e-10), f"{model_func.__name__} is not monotonic"


# ======================================================================
# Test: Experimental Variogram
# ======================================================================

class TestExperimentalVariogram:
    """Test experimental variogram computation."""

    def test_returns_arrays(self, synthetic_data):
        coords, values = synthetic_data
        lags, semiv, counts = compute_experimental_variogram(coords, values)
        assert len(lags) > 0
        assert len(semiv) == len(lags)
        assert len(counts) == len(lags)

    def test_positive_semivariance(self, synthetic_data):
        coords, values = synthetic_data
        lags, semiv, counts = compute_experimental_variogram(coords, values)
        assert np.all(semiv >= 0)

    def test_positive_counts(self, synthetic_data):
        coords, values = synthetic_data
        lags, semiv, counts = compute_experimental_variogram(coords, values)
        assert np.all(counts > 0)

    def test_custom_lag_distance(self, synthetic_data):
        coords, values = synthetic_data
        lags1, _, _ = compute_experimental_variogram(coords, values, lag_distance=50)
        lags2, _, _ = compute_experimental_variogram(coords, values, lag_distance=100)
        # Fewer bins with larger lag
        assert len(lags2) <= len(lags1)

    def test_directional_variogram(self, synthetic_data):
        coords, values = synthetic_data
        lags, semiv, counts = compute_experimental_variogram(
            coords, values, direction=0.0, tolerance=22.5,
        )
        assert len(lags) > 0


# ======================================================================
# Test: Model Fitting
# ======================================================================

class TestModelFitting:
    """Test variogram model fitting."""

    def test_auto_fit(self, known_variogram_data):
        lags, semiv, counts = known_variogram_data
        model = fit_theoretical_model(lags, semiv, counts)
        assert model is not None
        assert model.model_type in ["spherical", "exponential", "gaussian", "linear"]
        assert model.nugget >= 0
        assert model.sill > 0
        assert model.range_ > 0

    def test_specific_model_fit(self, known_variogram_data):
        lags, semiv, counts = known_variogram_data
        model = fit_theoretical_model(lags, semiv, counts, model_type="spherical")
        assert model.model_type == "spherical"
        # Should be close to true parameters
        assert model.nugget == pytest.approx(0.1, abs=0.15)
        assert model.sill == pytest.approx(1.0, abs=0.3)
        assert model.range_ == pytest.approx(300.0, abs=100.0)

    def test_fit_rmse(self, known_variogram_data):
        lags, semiv, counts = known_variogram_data
        model = fit_theoretical_model(lags, semiv, counts)
        assert model.fit_rmse >= 0
        assert model.fit_rmse < 0.5  # Should fit well


# ======================================================================
# Test: VariogramModel Serialization
# ======================================================================

class TestVariogramModelSerialization:
    """Test serialization to/from JSON."""

    def test_to_json(self, tmp_path):
        model = VariogramModel(
            model_type="spherical",
            nugget=0.1, sill=1.0, range_=300.0,
        )
        filepath = str(tmp_path / "model.json")
        model.to_json(filepath)
        assert os.path.exists(filepath)

        with open(filepath) as f:
            data = json.load(f)
        assert data["model_type"] == "spherical"

    def test_roundtrip(self, tmp_path):
        original = VariogramModel(
            model_type="exponential",
            nugget=0.2, sill=1.5, range_=250.0,
            direction=45.0, aniso_ratio=0.7,
        )
        filepath = str(tmp_path / "model.json")
        original.to_json(filepath)
        loaded = VariogramModel.from_json(filepath)

        assert loaded.model_type == original.model_type
        assert loaded.nugget == original.nugget
        assert loaded.sill == original.sill
        assert loaded.range_ == original.range_
        assert loaded.direction == original.direction
        assert loaded.aniso_ratio == original.aniso_ratio


# ======================================================================
# Test: Data Validation
# ======================================================================

class TestDataValidation:
    """Test data QA/QC functions."""

    def test_check_duplicates(self):
        coords = np.array([[0, 0], [0, 0], [1, 1], [2, 2]])
        mask, count = check_duplicates(coords)
        assert count == 1
        assert mask[1] == True  # Second point is duplicate

    def test_check_no_duplicates(self):
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        _, count = check_duplicates(coords)
        assert count == 0

    def test_check_null_values(self):
        values = np.array([1.0, np.nan, 3.0, np.inf])
        mask, count = check_null_values(values)
        assert count == 2

    def test_detect_outliers(self):
        values = np.concatenate([
            np.random.normal(10, 1, 100),
            np.array([100.0, -50.0]),  # Extreme outliers
        ])
        mask, stats = detect_outliers(values)
        assert stats["n_outliers"] >= 2

    def test_descriptive_stats(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_descriptive_stats(values)
        assert stats["count"] == 5
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)

    def test_validate_input_pipeline(self, synthetic_data):
        coords, values = synthetic_data
        result = validate_input_data(coords, values)
        assert result["n_valid"] > 0
        assert "stats" in result


# ======================================================================
# Test: Validation Metrics
# ======================================================================

class TestValidationMetrics:
    """Test cross-validation metric calculations."""

    def test_perfect_prediction(self):
        observed = np.array([1, 2, 3, 4, 5], dtype=float)
        predicted = np.array([1, 2, 3, 4, 5], dtype=float)
        metrics = compute_metrics(observed, predicted)
        assert metrics["ME"] == pytest.approx(0.0)
        assert metrics["MAE"] == pytest.approx(0.0)
        assert metrics["RMSE"] == pytest.approx(0.0)
        assert metrics["R2"] == pytest.approx(1.0)

    def test_biased_prediction(self):
        observed = np.array([1, 2, 3, 4, 5], dtype=float)
        predicted = np.array([2, 3, 4, 5, 6], dtype=float)  # Bias of +1
        metrics = compute_metrics(observed, predicted)
        assert metrics["ME"] == pytest.approx(1.0)

    def test_r2_range(self, synthetic_data):
        coords, values = synthetic_data
        predicted = values + np.random.normal(0, 0.1, len(values))
        metrics = compute_metrics(values, predicted)
        assert -1.0 <= metrics["R2"] <= 1.0


# ======================================================================
# Test: Auto-estimation
# ======================================================================

class TestAutoEstimation:
    """Test auto-estimation of variogram parameters."""

    def test_auto_estimate_params(self, synthetic_data):
        coords, values = synthetic_data
        params = auto_estimate_parameters(coords, values)
        assert params["lag_distance"] > 0
        assert params["max_distance"] > 0
        assert params["sill"] > 0
        assert params["n_lags"] >= 5


# ======================================================================
# Test: Grid Manager (pure functions only)
# ======================================================================

class TestGridManager:
    """Test grid generation functions."""

    def test_generate_grid(self):
        pytest.importorskip("osgeo", reason="GDAL only available in QGIS Python")
        from core.grid_manager import generate_grid
        gx, gy, nc, nr = generate_grid(0, 100, 0, 100, 10)
        assert nc == 10
        assert nr == 10
        assert gx[0] == pytest.approx(5.0)
        assert gy[0] == pytest.approx(5.0)

    def test_grid_dimensions(self):
        pytest.importorskip("osgeo", reason="GDAL only available in QGIS Python")
        from core.grid_manager import generate_grid
        gx, gy, nc, nr = generate_grid(0, 200, 0, 100, 10)
        assert nc == 20
        assert nr == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
