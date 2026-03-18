import os
from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .kriging_algorithm import UnifiedKrigingAlgorithm
from .variography_algorithm import InteractiveVariographyAlgorithm
from .variogram_map_algorithm import VariogramMapAlgorithm
from .validation_algorithm import ValidationReportAlgorithm

class GeostatProvider(QgsProcessingProvider):
    def __init__(self):
        super().__init__()

    def unload(self):
        pass

    def loadAlgorithms(self):
        self.addAlgorithm(InteractiveVariographyAlgorithm())
        self.addAlgorithm(VariogramMapAlgorithm())
        self.addAlgorithm(UnifiedKrigingAlgorithm())
        self.addAlgorithm(ValidationReportAlgorithm())

    def id(self):
        return 'geostat_exploration'

    def name(self):
        return 'Geostatistics Analysis'

    def icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), '..', 'icons', 'icon.png')
        return QIcon(icon_path)
