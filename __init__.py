# -*- coding: utf-8 -*-
"""
QGIS Geostatistics Plugin for Mineral Exploration
Compliant with Python 3.12.12 and PyQt6
"""

def classFactory(iface):
    """
    Entry point for QGIS to load the plugin.
    
    :param iface: A QGIS interface instance (QgisInterface)
    :type iface: QgsInterface
    """
    from .main import GeostatMineralPlugin
    return GeostatMineralPlugin(iface)
