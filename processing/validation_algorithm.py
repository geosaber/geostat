import os
import sys
import json
import numpy as np
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFile,
                       QgsProcessingParameterFileDestination,
                       QgsMessageLog, Qgis)
from pykrige.ok import OrdinaryKriging

class ValidationReportAlgorithm(QgsProcessingAlgorithm):
    """
    Algorithm for Cross-Validation (Leave-One-Out) and Technical Auditing.
    Allows a Qualified Person (QP) to validate the spatial model before resource estimation.
    Generates a PDF Audit Report with charts (Variogram and CV Scatter).
    """
    
    INPUT = 'INPUT'
    VALUE_FIELD = 'VALUE_FIELD'
    VARIOGRAM_MODEL = 'VARIOGRAM_MODEL'
    OUTPUT_REPORT = 'OUTPUT_REPORT'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return ValidationReportAlgorithm()

    def name(self):
        return 'validationreport'

    def displayName(self):
        return self.tr('Cross-Validation and Audit Report')

    def group(self):
        return self.tr('Geostatistics')

    def groupId(self):
        return 'geostatistics'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT, self.tr('Input Point Layer'), [QgsProcessing.TypeVectorPoint]))

        self.addParameter(QgsProcessingParameterField(
            self.VALUE_FIELD, self.tr('Value Field (Z)'), 
            parentLayerParameterName=self.INPUT, type=QgsProcessingParameterField.Numeric))

        self.addParameter(QgsProcessingParameterFile(
            self.VARIOGRAM_MODEL, self.tr('Variogram Parameter File (.json)'),
            behavior=QgsProcessingParameterFile.File, extension='json'))

        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_REPORT, self.tr('Output PDF Report'), 'PDF Files (*.pdf)'))

    def validate_data(self, points_x, points_y, values):
        x, y, z = np.array(points_x), np.array(points_y), np.array(values)
        initial_count = len(z)
        
        # Deduplication
        coords = np.column_stack((x, y))
        _, unique_indices = np.unique(coords, axis=0, return_index=True)
        unique_indices.sort()
        x, y, z = x[unique_indices], y[unique_indices], z[unique_indices]
        
        # Outlier count
        q1, q3 = np.percentile(z, 25), np.percentile(z, 75)
        iqr = q3 - q1
        outliers = np.sum((z > q3 + 3 * iqr) | (z < q1 - 3 * iqr))
        
        return x, y, z, (initial_count - len(z)), outliers

    def processAlgorithm(self, parameters, context, feedback):
        try:
            source = self.parameterAsSource(parameters, self.INPUT, context)
            if source is None:
                raise Exception('Invalid input layer.')

            field_name = self.parameterAsString(parameters, self.VALUE_FIELD, context)
            json_path = self.parameterAsFile(parameters, self.VARIOGRAM_MODEL, context)
            output_report_path = self.parameterAsFileOutput(parameters, self.OUTPUT_REPORT, context)

            if not os.path.exists(json_path):
                raise Exception("Missing Variogram parameter file JSON.")

            with open(json_path, 'r') as f:
                qp_params = json.load(f)

            # Data Loading
            points_x, points_y, values = [], [], []
            for feature in source.getFeatures():
                geom = feature.geometry().asPoint()
                val = feature[field_name]
                if val is not None:
                    points_x.append(geom.x())
                    points_y.append(geom.y())
                    values.append(float(val))

            if len(values) < 5:
                raise Exception("Too few points for cross-validation (at least 5 required).")

            x, y, z, dups, outliers = self.validate_data(points_x, points_y, values)
            data_stats = {'total_points': len(z), 'duplicates_removed': dups, 'outliers_detected': outliers}

            # Map JSON to PyKrige
            model_name = qp_params.get('model', 'spherical')
            nugget = float(qp_params.get('nugget', 0))
            sill = float(qp_params.get('sill', 1.0))
            v_range = float(qp_params.get('range', 10.0))
            angle = float(qp_params.get('anisotropy_angle', 0.0))
            scaling = float(qp_params.get('anisotropy_scaling', 1.0))

            feedback.pushInfo(f"Starting Cross-Validation (LOO) with {model_name} model...")
            
            # Implementation of manual Leave-One-Out (LOO) Cross-Validation
            # OrdinaryKriging.execute('cv') is not supported in recent versions of PyKrige
            n = len(z)
            predicted = np.zeros(n)
            variances = np.zeros(n)
            
            # For each point, train model with all other points and predict at its location
            for i in range(n):
                if feedback.isCanceled():
                     break
                
                # Progress update every 10%
                if i % max(1, n // 10) == 0:
                     feedback.setProgress(int(100 * i / n))
                
                # Prepare training data (all points except i)
                x_train = np.delete(x, i)
                y_train = np.delete(y, i)
                z_train = np.delete(z, i)
                
                # Fit model on N-1 points
                ok_temp = OrdinaryKriging(
                    x_train, y_train, z_train,
                    variogram_model=model_name,
                    variogram_parameters=[sill, v_range, nugget],
                    anisotropy_scaling=scaling,
                    anisotropy_angle=angle,
                    verbose=False, enable_plotting=False
                )
                
                # Predict at the omitted point location
                res, ss = ok_temp.execute('points', [x[i]], [y[i]])
                predicted[i] = res[0]
                variances[i] = ss[0]

            observed = z
            errors = predicted - observed
            
            # Metrics
            me = np.mean(errors)
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))
            
            # R2 Calculation
            ss_res = np.sum(errors**2)
            ss_tot = np.sum((observed - np.mean(observed))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # MSDR (Mean Standardized Squared Error) - ideal is closer to 1.0
            msdr = np.mean((errors**2)/variances) if not np.any(variances == 0) else 0

            cv_metrics = {
                'me': me, 'mae': mae, 'rmse': rmse, 'r2': r2, 'msdr': msdr
            }

            feedback.pushInfo(f"CV RMSE: {rmse:.4f} | R²: {r2:.4f}")

            # Generate Graphs for Report
            # Experimental variogram reconstruct from JSON if available
            exp_lags = np.array(qp_params.get('exp_lags', []))
            exp_gamma = np.array(qp_params.get('exp_gamma', []))
            
            # Generate theoretical curve for plotting
            from pykrige.variogram_models import (spherical_variogram_model, 
                                                 exponential_variogram_model, 
                                                 gaussian_variogram_model)
            model_funcs = {
                'spherical': spherical_variogram_model,
                'exponential': exponential_variogram_model,
                'gaussian': gaussian_variogram_model
            }
            model_lags, model_gamma = None, None
            if model_name in model_funcs:
                model_lags = np.linspace(0, v_range * 1.5, 100)
                model_gamma = model_funcs[model_name]([sill, v_range, nugget], model_lags)

            # Generate PDF
            from ..reports.report_generator import QPReportGenerator
            report_dir = os.path.dirname(output_report_path)
            # Need to ensure the filename is what the user asked (output_report_path)
            generator = QPReportGenerator(report_dir)
            
            # Call enhanced generator
            actual_pdf = generator.generate_pdf_report(
                  qp_params, data_stats, cv_metrics,
                  exp_lags=exp_lags, exp_gamma=exp_gamma,
                  model_lags=model_lags, model_gamma=model_gamma,
                  observed=observed, predicted=predicted
            )
            
            # Rename if the default Geostat_Audit_Report.pdf conflicts with user target
            if actual_pdf and actual_pdf != output_report_path:
                 try:
                      if os.path.exists(output_report_path):
                           os.remove(output_report_path)
                      os.rename(actual_pdf, output_report_path)
                 except:
                      pass

            QgsMessageLog.logMessage(f"Validation and Report: SUCCESS. Output: {output_report_path}", "Geostatistics Analysis", Qgis.Success)
            
            return {self.OUTPUT_REPORT: output_report_path}
            
        except Exception as e:
            QgsMessageLog.logMessage(f"Validation Error: {str(e)}", "Geostatistics Analysis", Qgis.Critical)
            feedback.reportError(str(e))
            raise e
