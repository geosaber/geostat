# -*- coding: utf-8 -*-
"""
GeoStats - QGIS Geostatistics Plugin
=====================================
Plugin entry point for QGIS 4.x (PyQt6).
Provides variogram modeling, kriging interpolation,
cross-validation, and audit-trail report generation.
"""


def classFactory(iface):
    """QGIS plugin entry point.

    Args:
        iface: QgisInterface instance providing access to the QGIS application.

    Returns:
        GeoStatPlugin instance.
    """
    from .main import GeoStatPlugin
    return GeoStatPlugin(iface)
