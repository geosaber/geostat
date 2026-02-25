# -*- coding: utf-8 -*-
"""
GeoStats Plugin - Main class
=============================
Manages plugin lifecycle: GUI integration, Processing provider registration,
and resource cleanup.
"""

import os

from qgis.core import QgsApplication, QgsMessageLog, Qgis

from .processing.provider import GeoStatProvider


class GeoStatPlugin:
    """Main plugin class registered via classFactory."""

    def __init__(self, iface):
        """
        Args:
            iface: QgisInterface providing access to QGIS GUI.
        """
        self.iface = iface
        self.provider = None
        self.plugin_dir = os.path.dirname(__file__)

    # ------------------------------------------------------------------
    # QGIS lifecycle hooks
    # ------------------------------------------------------------------

    def initProcessing(self):
        """Register the Processing provider (called by QGIS)."""
        self.provider = GeoStatProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)
        QgsMessageLog.logMessage(
            "GeoStats provider registered.", "GeoStats", Qgis.MessageLevel.Info
        )

    def initGui(self):
        """Initialize the plugin GUI elements."""
        self.initProcessing()

    def unload(self):
        """Clean up resources when plugin is unloaded."""
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
        QgsMessageLog.logMessage(
            "GeoStats plugin unloaded.", "GeoStats", Qgis.MessageLevel.Info
        )
