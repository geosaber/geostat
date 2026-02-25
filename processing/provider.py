# -*- coding: utf-8 -*-
"""
GeoStats Processing Provider
==============================
Registers all geostatistical algorithms into the QGIS Processing Toolbox.
"""

import os

from qgis.core import QgsProcessingProvider

from .variogram_algorithm import VariogramAlgorithm
from .kriging_algorithm import KrigingAlgorithm


class GeoStatProvider(QgsProcessingProvider):
    """Processing provider that groups all geostatistical algorithms."""

    def id(self) -> str:
        return "geostats"

    def name(self) -> str:
        return "Geostatistics"

    def longName(self) -> str:
        return "Geostatistics — Variogram & Kriging Toolbox"

    def icon(self):
        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons", "icon.png")
        from qgis.PyQt.QtGui import QIcon
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return super().icon()

    def loadAlgorithms(self):
        """Load all algorithms into the provider."""
        self.addAlgorithm(VariogramAlgorithm())
        self.addAlgorithm(KrigingAlgorithm())
