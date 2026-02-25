# -*- coding: utf-8 -*-
"""
Variogram Processing Algorithm
================================
QgsProcessingAlgorithm wrapper for the variography module.
Extracts data from QGIS vector layer, opens the interactive dialog,
and outputs a fitted variogram model as JSON.
"""

from __future__ import annotations

import os
import tempfile

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsMessageLog,
    QgsWkbTypes,
    Qgis,
)

import numpy as np

from ..core.variography import (
    VariogramModel,
    auto_estimate_parameters,
    compute_experimental_variogram,
    fit_theoretical_model,
)
from ..core.data_validation import validate_input_data


class VariogramAlgorithm(QgsProcessingAlgorithm):
    """Processing algorithm for interactive variogram modeling."""

    INPUT_LAYER = "INPUT_LAYER"
    Z_FIELD = "Z_FIELD"
    OUTPUT_MODEL = "OUTPUT_MODEL"

    def name(self) -> str:
        return "variogram_modeling"

    def displayName(self) -> str:
        return "Variogram Modeling"

    def group(self) -> str:
        return "Variography"

    def groupId(self) -> str:
        return "variography"

    def shortHelpString(self) -> str:
        return (
            "Interactive variogram modeling tool.\n\n"
            "Computes the experimental variogram, detects anisotropy, "
            "and opens an interactive dialog for model fitting.\n\n"
            "The fitted variogram model is saved as a JSON file "
            "for use by the Kriging algorithm."
        )

    def createInstance(self):
        return VariogramAlgorithm()

    def flags(self):
        """Force execution on the main GUI thread (required for dialog.exec())."""
        return super().flags() | QgsProcessingAlgorithm.Flag.FlagNoThreading

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
                "Z Field (Variable to Model)",
                parentLayerParameterName=self.INPUT_LAYER,
                type=QgsProcessingParameterField.DataType.Numeric,
            )
        )

        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_MODEL,
                "Output Variogram Model (JSON)",
                fileFilter="JSON files (*.json)",
                defaultValue=os.path.join(tempfile.gettempdir(), "variogram_model.json"),
            )
        )

    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Execute the algorithm."""
        layer = self.parameterAsVectorLayer(parameters, self.INPUT_LAYER, context)
        z_field = self.parameterAsString(parameters, self.Z_FIELD, context)
        output_path = self.parameterAsString(parameters, self.OUTPUT_MODEL, context)

        feedback.pushInfo(f"Input layer: {layer.name()} ({layer.featureCount()} features)")
        feedback.pushInfo(f"Z field: {z_field}")

        # ── Extract coordinates and values ──
        coords_list = []
        values_list = []

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
        feedback.pushInfo(f"Extracted {len(values)} valid samples")

        # ── Validate data ──
        validation = validate_input_data(
            coords, values,
            crs_wkt=layer.crs().toWkt() if layer.crs().isValid() else "",
        )

        for w in validation["warnings"]:
            feedback.pushWarning(w)
        for e in validation["errors"]:
            feedback.reportError(e, fatalError=True)
            return {}

        clean_coords = validation["clean_coords"]
        clean_values = validation["clean_values"]

        feedback.pushInfo(f"After validation: {validation['n_valid']} samples")

        # ── Interactive dialog ──
        # Note: In Processing context, we try to open the interactive dialog.
        # If running headless or in batch mode, we fall back to auto-fit.

        fitted_model = None
        try:
            from qgis.PyQt.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                from ..ui.variogram_dialog import VariogramDialog
                dialog = VariogramDialog(clean_coords, clean_values)
                result = dialog.exec()
                if result:
                    fitted_model = dialog.get_model()
                else:
                    feedback.pushWarning("Variogram dialog cancelled by user. Using auto-fit.")
        except Exception as e:
            feedback.pushWarning(f"Could not open interactive dialog: {e}. Using auto-fit.")

        # ── Fallback: auto-fit ──
        if fitted_model is None:
            feedback.pushInfo("Running automatic variogram fitting...")
            params = auto_estimate_parameters(clean_coords, clean_values)
            lags, semiv, counts = compute_experimental_variogram(
                clean_coords, clean_values,
                lag_distance=params["lag_distance"],
                max_distance=params["max_distance"],
                n_lags=params["n_lags"],
            )
            fitted_model = fit_theoretical_model(lags, semiv, counts)

        # ── Save model ──
        fitted_model.to_json(output_path)
        feedback.pushInfo(f"Variogram model saved to: {output_path}")
        feedback.pushInfo(
            f"Model: {fitted_model.model_type}, "
            f"Nugget={fitted_model.nugget:.4f}, "
            f"Sill={fitted_model.sill:.4f}, "
            f"Range={fitted_model.range_:.2f}, "
            f"RMSE={fitted_model.fit_rmse:.6f}"
        )

        QgsMessageLog.logMessage(
            f"Variogram fitted: {fitted_model.model_type} "
            f"(N={fitted_model.nugget:.4f}, S={fitted_model.sill:.4f}, R={fitted_model.range_:.2f})",
            "GeoStats", Qgis.MessageLevel.Info,
        )

        return {self.OUTPUT_MODEL: output_path}
