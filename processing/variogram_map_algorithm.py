from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterField,
                       QgsProcessingParameterNumber,
                       QgsMessageLog, Qgis)
import numpy as np

class VariogramMapAlgorithm(QgsProcessingAlgorithm):
    """
    Algorithm to generate a 2D Variogram Map (Anisotropy Heatmap).
    Maps experimental semivariance as a function of distance and direction (lag dx, dy).
    Useful for Qualified Persons to identify main anisotropy axes.
    Restricted to Main Thread for safe Dialog opening.
    """
    
    INPUT = 'INPUT'
    VALUE_FIELD = 'VALUE_FIELD'
    MAX_LAG = 'MAX_LAG'
    N_CELLS = 'N_CELLS'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return VariogramMapAlgorithm()

    def name(self):
        return 'variogrammap'

    def displayName(self):
        return self.tr('Variogram Map Generator (Anisotropy)')

    def group(self):
        return self.tr('Geostatistics')

    def groupId(self):
        return 'geostatistics'
        
    def flags(self):
        return super().flags() | QgsProcessingAlgorithm.FlagNoThreading

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT, self.tr('Input Point Layer'), [QgsProcessing.TypeVectorPoint]))

        self.addParameter(QgsProcessingParameterField(
            self.VALUE_FIELD, self.tr('Value Field (Z)'), 
            parentLayerParameterName=self.INPUT, type=QgsProcessingParameterField.Numeric))

        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_LAG, self.tr('Maximum Search Distance (Max Lag)'), 
            type=QgsProcessingParameterNumber.Double, defaultValue=100.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.N_CELLS, self.tr('Map Resolution (Cells NxN)'), 
            type=QgsProcessingParameterNumber.Integer, defaultValue=20))

    def validate_data(self, points_x, points_y, values):
        # Convert to numpy arrays explicitly
        x = np.array(points_x)
        y = np.array(points_y)
        z = np.array(values)
        
        # Deduplication using numpy: combine x and y into a single coordinate array
        coords = np.column_stack((x, y))
        # np.unique with return_index=True finds the index of the first occurrence of each unique element
        _, unique_indices = np.unique(coords, axis=0, return_index=True)
        
        # Sort indices to maintain some original ordering (optional but good practice)
        unique_indices.sort()
        
        return x[unique_indices], y[unique_indices], z[unique_indices]

    def processAlgorithm(self, parameters, context, feedback):
        try:
            source = self.parameterAsSource(parameters, self.INPUT, context)
            if source is None:
                raise Exception('Invalid input layer.')

            field_name = self.parameterAsString(parameters, self.VALUE_FIELD, context)
            max_lag_val = self.parameterAsDouble(parameters, self.MAX_LAG, context)
            n_cells = self.parameterAsInt(parameters, self.N_CELLS, context)

            points_x, points_y, values = [], [], []
            for feature in source.getFeatures():
                geom = feature.geometry().asPoint()
                val = feature[field_name]
                if val is not None:
                    points_x.append(geom.x())
                    points_y.append(geom.y())
                    values.append(float(val))

            if len(values) < 3:
                raise Exception("Insufficient points.")

            x, y, z = self.validate_data(points_x, points_y, values)
            
            feedback.pushInfo("Calculating separation vectors (h)...")
            
            n = len(z)
            x_edges = np.linspace(-max_lag_val, max_lag_val, n_cells + 1)
            y_edges = np.linspace(-max_lag_val, max_lag_val, n_cells + 1)
            
            gamma_map = np.zeros((n_cells, n_cells))
            count_map = np.zeros((n_cells, n_cells))

            for i in range(n):
                dx = x[i+1:] - x[i]
                dy = y[i+1:] - y[i]
                dz = z[i+1:] - z[i]
                
                dist = np.sqrt(dx**2 + dy**2)
                valid = dist <= max_lag_val
                
                v_dx = dx[valid]
                v_dy = dy[valid]
                v_dz = dz[valid]
                v_gamma = 0.5 * (v_dz ** 2)
                
                ix = np.digitize(v_dx, x_edges) - 1
                iy = np.digitize(v_dy, y_edges) - 1
                
                in_bounds = (ix >= 0) & (ix < n_cells) & (iy >= 0) & (iy < n_cells)
                for idx in np.where(in_bounds)[0]:
                    gamma_map[iy[idx], ix[idx]] += v_gamma[idx]
                    count_map[iy[idx], ix[idx]] += 1
                
                ix_sym = np.digitize(-v_dx, x_edges) - 1
                iy_sym = np.digitize(-v_dy, y_edges) - 1
                in_bounds_sym = (ix_sym >= 0) & (ix_sym < n_cells) & (iy_sym >= 0) & (iy_sym < n_cells)
                for idx in np.where(in_bounds_sym)[0]:
                    gamma_map[iy_sym[idx], ix_sym[idx]] += v_gamma[idx]
                    count_map[iy_sym[idx], ix_sym[idx]] += 1

            with np.errstate(divide='ignore', invalid='ignore'):
                variogram_map = gamma_map / count_map

            feedback.pushInfo("Opening Internal Variogram Map Viewer...")
            from ..ui.variogram_map_viewer import VariogramMapDialog
            
            dialog = VariogramMapDialog(variogram_map, max_lag_val)
            dialog.exec()
            
            return {}
            
        except Exception as e:
            QgsMessageLog.logMessage(f"Error generating Variogram Map: {str(e)}", "Geostatistics Analysis", Qgis.Critical)
            feedback.reportError(str(e))
            raise e
