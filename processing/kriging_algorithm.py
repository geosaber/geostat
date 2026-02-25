# -*- coding: utf-8 -*-
"""
Kriging Processing Algorithm
===============================
QgsProcessingAlgorithm for kriging interpolation.
Loads a fitted variogram model from JSON, performs kriging on a regular grid,
applies a boundary mask, and outputs GeoTIFF rasters.
"""

from __future__ import annotations

import os
import tempfile

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterFile,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFileDestination,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsMessageLog,
    Qgis,
)

import numpy as np

from ..core.variography import VariogramModel
from ..core.kriging_engine import run_kriging
from ..core.grid_manager import (
    generate_grid,
    compute_convex_hull,
    create_boundary_mask,
    apply_mask,
    create_output_raster,
    boundary_from_layer_wkt,
)
from ..core.data_validation import validate_input_data


class KrigingAlgorithm(QgsProcessingAlgorithm):
    """Processing algorithm for Kriging interpolation."""

    INPUT_LAYER = "INPUT_LAYER"
    Z_FIELD = "Z_FIELD"
    VARIOGRAM_FILE = "VARIOGRAM_FILE"
    METHOD = "METHOD"
    CELL_SIZE = "CELL_SIZE"
    BOUNDARY_LAYER = "BOUNDARY_LAYER"
    USE_CONVEX_HULL = "USE_CONVEX_HULL"
    CUTOFF = "CUTOFF"
    DRIFT_RASTER = "DRIFT_RASTER"
    DRIFT_TERMS = "DRIFT_TERMS"
    RUN_VALIDATION = "RUN_VALIDATION"
    OUTPUT_RASTER = "OUTPUT_RASTER"
    OUTPUT_VARIANCE = "OUTPUT_VARIANCE"
    OUTPUT_REPORT = "OUTPUT_REPORT"

    METHODS = ["ordinary", "universal", "external_drift", "indicator"]
    METHOD_LABELS = [
        "Ordinary Kriging (OK)",
        "Universal Kriging (UK)",
        "External Drift Kriging (EDK)",
        "Indicator Kriging (IK)",
    ]
    DRIFT_OPTIONS = ["regional_linear", "regional_quadratic"]

    def name(self) -> str:
        return "kriging_interpolation"

    def displayName(self) -> str:
        return "Kriging Interpolation"

    def group(self) -> str:
        return "Interpolation"

    def groupId(self) -> str:
        return "interpolation"

    def shortHelpString(self) -> str:
        return (
            "Kriging interpolation using a fitted variogram model.\n\n"
            "Methods:\n"
            "- Ordinary Kriging (OK): Standard kriging, assumes constant unknown mean.\n"
            "- Universal Kriging (UK): Kriging with polynomial drift (trend).\n"
            "- External Drift Kriging (EDK): Uses an auxiliary raster as drift function.\n"
            "- Indicator Kriging (IK): Estimates probability above a cutoff threshold.\n\n"
            "The variogram model must be provided as a JSON file (output of Variogram Modeling).\n"
            "A boundary polygon can be used to mask the output raster."
        )

    def createInstance(self):
        return KrigingAlgorithm()

    def initAlgorithm(self, config=None):
        """Define algorithm inputs and outputs."""
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_LAYER,
                "Input Point Layer",
                [QgsProcessing.TypeVectorPoint],
            )
        )

        self.addParameter(
            QgsProcessingParameterField(
                self.Z_FIELD,
                "Z Field (Variable to Interpolate)",
                parentLayerParameterName=self.INPUT_LAYER,
                type=QgsProcessingParameterField.DataType.Numeric,
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.VARIOGRAM_FILE,
                "Variogram Model (JSON)",
                extension="json",
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                "Kriging Method",
                options=self.METHOD_LABELS,
                defaultValue=0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.CELL_SIZE,
                "Cell Size (Grid Resolution)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=100.0,
                minValue=0.01,
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.BOUNDARY_LAYER,
                "Boundary Polygon (optional)",
                [QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_CONVEX_HULL,
                "Use Convex Hull as Boundary (if no polygon provided)",
                defaultValue=True,
            )
        )

        # Indicator Kriging: cutoff
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CUTOFF,
                "Indicator Kriging: Cutoff Value",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.0,
                optional=True,
            )
        )

        # External Drift: raster
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.DRIFT_RASTER,
                "External Drift Raster (for EDK)",
                optional=True,
            )
        )

        # Universal Kriging: drift terms
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DRIFT_TERMS,
                "Drift Type (for UK)",
                options=["Regional Linear", "Regional Quadratic"],
                defaultValue=0,
                optional=True,
            )
        )

        # Cross-validation option
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.RUN_VALIDATION,
                "Run Leave-One-Out Cross-Validation",
                defaultValue=False,
            )
        )

        # Outputs
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_RASTER,
                "Output Kriging Raster",
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_VARIANCE,
                "Output Variance Raster",
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_REPORT,
                "Output PDF Report (optional)",
                fileFilter="PDF files (*.pdf)",
                optional=True,
            )
        )

    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Execute the kriging algorithm."""
        layer = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)
        z_field = self.parameterAsString(parameters, self.Z_FIELD, context)
        vario_path = self.parameterAsString(parameters, self.VARIOGRAM_FILE, context)
        method_idx = self.parameterAsEnum(parameters, self.METHOD, context)
        cell_size = self.parameterAsDouble(parameters, self.CELL_SIZE, context)
        boundary_layer = self.parameterAsVectorLayer(parameters, self.BOUNDARY_LAYER, context)
        use_hull = self.parameterAsBool(parameters, self.USE_CONVEX_HULL, context)
        cutoff = self.parameterAsDouble(parameters, self.CUTOFF, context)
        drift_raster = self.parameterAsRasterLayer(parameters, self.DRIFT_RASTER, context)
        drift_idx = self.parameterAsEnum(parameters, self.DRIFT_TERMS, context)
        run_cv = self.parameterAsBool(parameters, self.RUN_VALIDATION, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)
        variance_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_VARIANCE, context)
        report_path = self.parameterAsString(parameters, self.OUTPUT_REPORT, context)

        method = self.METHODS[method_idx]
        feedback.pushInfo(f"Kriging method: {self.METHOD_LABELS[method_idx]}")

        # ── Load variogram model ──
        feedback.pushInfo(f"Loading variogram model from: {vario_path}")
        model = VariogramModel.from_json(vario_path)
        feedback.pushInfo(
            f"Model: {model.model_type}, N={model.nugget:.4f}, "
            f"S={model.sill:.4f}, R={model.range_:.2f}"
        )

        # ── Extract data ──
        coords_list, values_list = [], []
        for feature in layer.getFeatures():
            geom = feature.geometry()
            if geom.isNull():
                continue
            point = geom.asPoint()
            val = feature[z_field]
            if val is None:
                continue
            try:
                coords_list.append([point.x(), point.y()])
                values_list.append(float(val))
            except (ValueError, TypeError):
                continue

        coords = np.array(coords_list)
        values = np.array(values_list)

        # ── Validate ──
        validation = validate_input_data(coords, values)
        for w in validation["warnings"]:
            feedback.pushWarning(w)
        for e in validation["errors"]:
            feedback.reportError(e, fatalError=True)
            return {}

        clean_coords = validation["clean_coords"]
        clean_values = validation["clean_values"]
        feedback.pushInfo(f"Valid samples: {len(clean_values)}")

        # ── Bounding box and grid ──
        extent = layer.extent()
        x_min, x_max = extent.xMinimum(), extent.xMaximum()
        y_min, y_max = extent.yMinimum(), extent.yMaximum()

        # Add buffer of 1 cell around extent
        x_min -= cell_size
        x_max += cell_size
        y_min -= cell_size
        y_max += cell_size

        grid_x, grid_y, n_cols, n_rows = generate_grid(
            x_min, x_max, y_min, y_max, cell_size,
        )
        feedback.pushInfo(f"Grid: {n_cols} × {n_rows} cells (cell size: {cell_size})")

        if feedback.isCanceled():
            return {}

        # ── Prepare kriging kwargs ──
        kriging_kwargs = {}
        if method == "indicator":
            kriging_kwargs["cutoff"] = cutoff
            feedback.pushInfo(f"Indicator Kriging cutoff: {cutoff}")
        elif method == "universal":
            drift = self.DRIFT_OPTIONS[drift_idx]
            kriging_kwargs["drift_terms"] = drift
            feedback.pushInfo(f"UK drift type: {drift}")
        elif method == "external_drift":
            if drift_raster is None:
                feedback.reportError(
                    "External Drift Kriging requires a drift raster.", fatalError=True,
                )
                return {}
            # Extract drift values at sample locations and on grid
            kriging_kwargs.update(
                self._extract_drift_data(drift_raster, clean_coords, grid_x, grid_y, feedback)
            )

        # ── Run Kriging ──
        feedback.pushInfo("Running kriging interpolation...")
        feedback.setProgress(30)

        z_pred, ss_pred = run_kriging(
            method=method,
            coords=clean_coords,
            values=clean_values,
            grid_x=grid_x,
            grid_y=grid_y,
            model=model,
            **kriging_kwargs,
        )

        feedback.setProgress(70)
        feedback.pushInfo("Kriging completed.")

        # ── Apply boundary mask ──
        boundary_geom = None
        if boundary_layer is not None:
            feedback.pushInfo("Applying polygon boundary mask...")
            for feature in boundary_layer.getFeatures():
                wkt = feature.geometry().asWkt()
                boundary_geom = boundary_from_layer_wkt(wkt)
                break  # Use first polygon
        elif use_hull:
            feedback.pushInfo("Applying convex hull boundary mask...")
            boundary_geom = compute_convex_hull(clean_coords)

        nodata = -9999.0
        if boundary_geom is not None:
            mask = create_boundary_mask(grid_x, grid_y, boundary_geom)
            z_pred = apply_mask(z_pred, mask, nodata=nodata)
            ss_pred = apply_mask(ss_pred, mask, nodata=nodata)
            feedback.pushInfo("Boundary mask applied.")

        feedback.setProgress(85)

        # ── Write output rasters ──
        crs_wkt = layer.crs().toWkt() if layer.crs().isValid() else ""

        create_output_raster(
            output_path, z_pred, x_min, y_max, cell_size, crs_wkt, nodata,
        )
        feedback.pushInfo(f"Kriging raster saved: {output_path}")

        results = {self.OUTPUT_RASTER: output_path}

        if variance_path:
            create_output_raster(
                variance_path, ss_pred, x_min, y_max, cell_size, crs_wkt, nodata,
            )
            feedback.pushInfo(f"Variance raster saved: {variance_path}")
            results[self.OUTPUT_VARIANCE] = variance_path

        # ── Cross-validation ──
        cv_results = None
        if run_cv:
            cv_results = self._run_cross_validation(
                clean_coords, clean_values, model, method, kriging_kwargs, feedback,
            )

        # ── PDF Report ──
        if report_path:
            self._generate_report(
                report_path, layer, z_field, validation,
                model, method, cell_size, n_cols, n_rows,
                boundary_layer, use_hull, cutoff, drift_idx,
                output_path, cv_results, feedback,
            )
            results[self.OUTPUT_REPORT] = report_path

        feedback.setProgress(100)

        QgsMessageLog.logMessage(
            f"Kriging ({method}) completed: {n_cols}x{n_rows} grid",
            "GeoStats", Qgis.MessageLevel.Info,
        )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_drift_data(
        self,
        drift_raster,
        coords: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Extract drift values from a raster at sample and grid locations."""
        from osgeo import gdal
        ds = gdal.Open(drift_raster.source())
        band = ds.GetRasterBand(1)
        gt = ds.GetGeoTransform()

        def _sample_raster(x, y):
            col = int((x - gt[0]) / gt[1])
            row = int((y - gt[3]) / gt[5])
            col = max(0, min(col, ds.RasterXSize - 1))
            row = max(0, min(row, ds.RasterYSize - 1))
            return float(band.ReadAsArray(col, row, 1, 1)[0, 0])

        # At sample locations
        drift_at_samples = np.array([_sample_raster(x, y) for x, y in coords])

        # On grid
        ny, nx = len(grid_y), len(grid_x)
        drift_grid = np.zeros((ny, nx))
        for j, y in enumerate(grid_y):
            for i, x in enumerate(grid_x):
                drift_grid[j, i] = _sample_raster(x, y)

        ds = None
        feedback.pushInfo(f"Drift data extracted: {len(drift_at_samples)} samples + {ny}×{nx} grid")

        return {
            "drift_values_at_samples": drift_at_samples,
            "drift_grid": drift_grid,
        }

    def _run_cross_validation(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        model: VariogramModel,
        method: str,
        kriging_kwargs: dict,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Run LOO cross-validation and report metrics."""
        from ..core.validation import leave_one_out_cv

        feedback.pushInfo("Running Leave-One-Out Cross-Validation...")

        # Don't pass drift data for CV (would need per-sample extraction)
        cv_kwargs = {k: v for k, v in kriging_kwargs.items()
                     if k not in ("drift_values_at_samples", "drift_grid")}

        cv_results = leave_one_out_cv(
            coords, values, model, method=method, **cv_kwargs,
        )

        metrics = cv_results["metrics"]
        feedback.pushInfo("=== Cross-Validation Results ===")
        feedback.pushInfo(f"  ME  (Mean Error):          {metrics['ME']:.6f}")
        feedback.pushInfo(f"  MAE (Mean Absolute Error): {metrics['MAE']:.6f}")
        feedback.pushInfo(f"  RMSE:                      {metrics['RMSE']:.6f}")
        feedback.pushInfo(f"  R2:                        {metrics['R2']:.6f}")
        if not np.isnan(metrics['MSDR']):
            feedback.pushInfo(f"  MSDR:                      {metrics['MSDR']:.6f}")
        feedback.pushInfo("================================")

        return cv_results

    def _generate_report(
        self,
        report_path: str,
        layer,
        z_field: str,
        validation: dict,
        model: VariogramModel,
        method: str,
        cell_size: float,
        n_cols: int,
        n_rows: int,
        boundary_layer,
        use_hull: bool,
        cutoff: float,
        drift_idx: int,
        output_raster_path: str,
        cv_results: dict,
        feedback: QgsProcessingFeedback,
    ):
        """Generate comprehensive PDF report."""
        from ..reports.report_generator import generate_report

        feedback.pushInfo("Generating PDF report...")

        # Input info
        input_info = {
            "layer_name": layer.name(),
            "field_name": z_field,
            "n_samples": validation["n_valid"],
            "crs": layer.crs().authid() if layer.crs().isValid() else "N/A",
            "stats": validation.get("stats", {}),
            "warnings": validation.get("warnings", []),
        }

        # Variogram info
        variogram_info = {
            "model_type": model.model_type,
            "nugget": model.nugget,
            "sill": model.sill,
            "range": model.range_,
            "direction": model.direction,
            "aniso_ratio": model.aniso_ratio,
            "fit_rmse": model.fit_rmse,
            "lags": model.lags,
            "semivariance": model.semivariance,
        }

        # Kriging info
        boundary_type = "None"
        if boundary_layer is not None:
            boundary_type = f"Polygon: {boundary_layer.name()}"
        elif use_hull:
            boundary_type = "Convex Hull"

        kriging_info = {
            "method": self.METHOD_LABELS[self.METHODS.index(method)],
            "cell_size": cell_size,
            "grid_size": f"{n_cols} x {n_rows}",
            "boundary_type": boundary_type,
            "output_path": output_raster_path,
            "cutoff": cutoff if method == "indicator" else None,
            "drift_terms": self.DRIFT_OPTIONS[drift_idx] if method == "universal" else None,
        }

        # Validation info
        validation_info = None
        if cv_results is not None:
            validation_info = {
                "metrics": cv_results["metrics"],
                "observed": cv_results["observed"].tolist(),
                "predicted": cv_results["predicted"].tolist(),
                "errors": cv_results["errors"].tolist(),
            }

        generate_report(
            output_path=report_path,
            project_name=f"Kriging: {layer.name()} - {z_field}",
            input_info=input_info,
            variogram_info=variogram_info,
            kriging_info=kriging_info,
            validation_info=validation_info,
        )

        feedback.pushInfo(f"PDF report saved: {report_path}")
