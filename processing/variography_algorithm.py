from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFileDestination,
                       QgsMessageLog, Qgis)
import numpy as np
import json

class InteractiveVariographyAlgorithm(QgsProcessingAlgorithm):
    """
    Interactive algorithm for Variogram modeling.
    Runs on the QGIS main thread to allow safe opening of the PyQt UI.
    """
    
    INPUT = 'INPUT'
    VALUE_FIELD = 'VALUE_FIELD'
    OUTPUT_JSON = 'OUTPUT_JSON'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return InteractiveVariographyAlgorithm()

    def name(self):
        return 'interactivevariography'

    def displayName(self):
        return self.tr('Variogram Modeling (Interactive)')

    def group(self):
        return self.tr('Geostatistics')

    def groupId(self):
        return 'geostatistics'
        
    def flags(self):
        # CRITICAL: Forces this algorithm to run on the main thread so the PyQt Window can open without crashing QGIS.
        return super().flags() | QgsProcessingAlgorithm.FlagNoThreading

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT, self.tr('Input Point Layer'), [QgsProcessing.TypeVectorPoint]))

        self.addParameter(QgsProcessingParameterField(
            self.VALUE_FIELD, self.tr('Value Field (Z)'), 
            parentLayerParameterName=self.INPUT, type=QgsProcessingParameterField.Numeric))

        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_JSON, self.tr('Output Parameter JSON File'), 'JSON Files (*.json)'))

    def validate_data(self, points_x, points_y, values):
        x = np.array(points_x)
        y = np.array(points_y)
        z = np.array(values)
        
        initial_count = len(z)
        
        # 1. Deduplication (Spatial)
        coords = np.column_stack((x, y))
        _, unique_indices = np.unique(coords, axis=0, return_index=True)
        unique_indices.sort()
        
        x, y, z = x[unique_indices], y[unique_indices], z[unique_indices]
        
        if len(z) < initial_count:
            QgsMessageLog.logMessage(f"Removed {initial_count - len(z)} duplicate coordinates.", "Geostatistics Analysis", Qgis.Warning)
        
        # 2. Outlier Detection (using numpy.percentile instead of pandas.quantile)
        if len(z) > 0:
            q1 = np.percentile(z, 25)
            q3 = np.percentile(z, 75)
            iqr = q3 - q1
            upper_bound = q3 + 3 * iqr
            lower_bound = q1 - 3 * iqr
            
            outlier_mask = (z > upper_bound) | (z < lower_bound)
            n_outliers = np.sum(outlier_mask)
            
            if n_outliers > 0:
                QgsMessageLog.logMessage(f"Detected {n_outliers} extreme outliers beyond 3 IQR. Consider investigating data quality.", "Geostatistics Analysis", Qgis.Warning)
            
        return x, y, z

    def processAlgorithm(self, parameters, context, feedback):
        try:
            source = self.parameterAsSource(parameters, self.INPUT, context)
            if source is None:
                raise Exception('Invalid input layer.')

            field_name = self.parameterAsString(parameters, self.VALUE_FIELD, context)
            output_json = self.parameterAsFileOutput(parameters, self.OUTPUT_JSON, context)

            points_x, points_y, values = [], [], []
            for feature in source.getFeatures():
                geom = feature.geometry().asPoint()
                points_x.append(geom.x())
                points_y.append(geom.y())
                val = feature[field_name]
                if val is not None:
                    values.append(float(val))

            if len(values) < 3:
                raise Exception("Insufficient points for geostatistics.")

            points_x, points_y, values = self.validate_data(points_x, points_y, values)
            
            feedback.pushInfo("Opening Interactive Variogram UI for validation...")
            QgsMessageLog.logMessage("Launching Variogram Dialog on Main Thread.", "Geostatistics Analysis", Qgis.Info)
            
            from ..ui.variogram_dialog import VariogramVisualizerDialog
            
            x_arr = np.array(points_x)
            y_arr = np.array(points_y)
            z_arr = np.array(values)
            
            qp_params = {}
            dialog = VariogramVisualizerDialog(x_arr, y_arr, z_arr)
            if dialog.exec():
                qp_params = dialog.get_parameters()
            
            if not qp_params:
                feedback.reportError("Geostatistical modeling cancelled by user.")
                return {}
                
            # Serialize parameters to JSON file
            with open(output_json, 'w') as f:
                json.dump(qp_params, f, indent=4)
                
            QgsMessageLog.logMessage(f"Variogram parameters saved to {output_json}", "Geostatistics Analysis", Qgis.Success)
            
            return {self.OUTPUT_JSON: output_json}
            
        except Exception as e:
            QgsMessageLog.logMessage(f"Geostatistical Error: {str(e)}", "Geostatistics Analysis", Qgis.Critical)
            feedback.reportError(str(e))
            raise e
