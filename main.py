import os
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis.core import QgsProcessingAlgorithm, QgsApplication, QgsMessageLog, Qgis

# Import the extracted provider
from .processing.provider import GeostatProvider

class GeostatMineralPlugin:
    """
    Main class for the QGIS Geostatistics Plugin.
    Manages UI integration and Processing Toolbox registration for the QP.
    """
    def __init__(self, iface):
        self.iface = iface
        self.provider = None
        self.plugin_dir = os.path.dirname(__file__)

    def initGui(self):
        """
        Initializes the plugin GUI and registers the Processing Provider.
        """
        # 1. Initialize and register the Processing Provider (Toolbox)
        self.provider = GeostatProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

        # 2. Add an icon/action to the QGIS Toolbar for quick access
        icon_path = os.path.join(self.plugin_dir, 'icons', 'icon.png')
        self.action = QAction(
            QIcon(icon_path),
            "Geostatistics for Mineral Exploration (QP Tool)",
            self.iface.mainWindow()
        )
        self.action.triggered.connect(self.run_quick_analysis)
        
        # Add to the 'Plugins' menu and Toolbar
        self.iface.addPluginToMenu("&Mineral Exploration", self.action)
        self.iface.addToolBarIcon(self.action)
        
        QgsMessageLog.logMessage("Geostat Plugin Initialized successfully", "Geostatistics Analysis", Qgis.Info)

    def unload(self):
        """
        Cleans up the UI and unregisters the provider when the plugin is disabled.
        """
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)
        
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("&Mineral Exploration", self.action)

    def run_quick_analysis(self):
        """
        Helper method to open the Processing Toolbox focused on our Geostat algorithms.
        """
        import processing
        try:
            processing.execAlgorithmDialog("geostat_exploration:ordinarykriging")
        except AttributeError:
            pass # Fallback if processing dialog fails to open
            
        # In a full implementation, you could trigger the Variogram Dialog directly here.
        QgsMessageLog.logMessage("Geostat Tool: Ready for Qualified Person validation.", "Geostatistics Analysis", Qgis.Info)

