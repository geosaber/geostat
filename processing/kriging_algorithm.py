from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterField,
                       QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterFile,
                       QgsMessageLog, Qgis, QgsRasterLayer, QgsCoordinateReferenceSystem, QgsProviderRegistry)
from pykrige.ok import OrdinaryKriging
from osgeo import gdal, osr
import numpy as np
import os
import sys

class UnifiedKrigingAlgorithm(QgsProcessingAlgorithm):
    """
    Unified Ordinary and Indicator Kriging Algorithm for Mineral Resource Estimation.
    Developed for compliance with Qualified Person (QP) standards.
    """
    
    INPUT = 'INPUT'
    VALUE_FIELD = 'VALUE_FIELD'
    KRIGING_TYPE = 'KRIGING_TYPE'
    THRESHOLD = 'THRESHOLD'
    MASK = 'MASK'
    VARIOGRAM_MODEL = 'VARIOGRAM_MODEL'
    CELL_SIZE = 'CELL_SIZE'
    OUTPUT = 'OUTPUT'
    
    KTYPES = ['Ordinary Kriging (Grades)', 'Indicator Kriging (Probability)']

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return UnifiedKrigingAlgorithm()

    def name(self):
        return 'unifiedkriging'

    def displayName(self):
        return self.tr('Ordinary and Indicator Kriging')

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

        self.addParameter(QgsProcessingParameterEnum(
            self.KRIGING_TYPE, self.tr('Kriging Type'), options=self.KTYPES, defaultValue=0))

        self.addParameter(QgsProcessingParameterNumber(
            self.THRESHOLD, self.tr('Cut-off (Indicator Kriging Only)'), 
            type=QgsProcessingParameterNumber.Double, defaultValue=1.0))

        self.addParameter(QgsProcessingParameterFeatureSource(
            self.MASK, self.tr('Boundary Polygon (Optional - GDAL Clip)'), 
            [QgsProcessing.TypeVectorPolygon], optional=True))

        self.addParameter(QgsProcessingParameterFile(
            self.VARIOGRAM_MODEL, self.tr('Variogram Parameter File (.json)'),
            behavior=QgsProcessingParameterFile.File, extension='json'))

        self.addParameter(QgsProcessingParameterNumber(
            self.CELL_SIZE, self.tr('Cell Size (Block Size)'), 
            type=QgsProcessingParameterNumber.Double, defaultValue=10.0))

        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, self.tr('Output Raster (Estimation)')))

    def validate_data(self, points_x, points_y, values):
        x = np.array(points_x)
        y = np.array(points_y)
        z = np.array(values)
        
        initial_count = len(z)
        
        # 1. Deduplication
        coords = np.column_stack((x, y))
        _, unique_indices = np.unique(coords, axis=0, return_index=True)
        unique_indices.sort()
        
        x, y, z = x[unique_indices], y[unique_indices], z[unique_indices]
        
        if len(z) < initial_count:
            QgsMessageLog.logMessage(f"Removed {initial_count - len(z)} duplicate coordinates.", "Geostatistics Analysis", Qgis.Warning)
            
        # 2. Outlier Detection
        if len(z) > 0:
            q1 = np.percentile(z, 25)
            q3 = np.percentile(z, 75)
            iqr = q3 - q1
            upper_bound = q1 + 3 * iqr 
            lower_bound = q1 - 3 * iqr
            
            outlier_mask = (z > upper_bound) | (z < lower_bound)
            n_outliers = np.sum(outlier_mask)
            
            if n_outliers > 0:
                QgsMessageLog.logMessage(f"Detected {n_outliers} extreme outliers beyond 3x IQR.", "Geostatistics Analysis", Qgis.Warning)
            
        return x, y, z

    def create_raster(self, z_array, grid_x, grid_y, cell_size, crs, output_path, is_indicator=False):
        cols = len(grid_x)
        rows = len(grid_y)
        
        origin_x = np.min(grid_x) - cell_size/2.0
        origin_y = np.max(grid_y) + cell_size/2.0
        
        driver = gdal.GetDriverByName('GTiff')
        out_raster = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
        out_raster.SetGeoTransform((origin_x, cell_size, 0, origin_y, 0, -cell_size))
        
        if crs.isValid():
            out_srs = osr.SpatialReference()
            out_srs.ImportFromWkt(crs.toWkt())
            out_raster.SetProjection(out_srs.ExportToWkt())
            
        out_band = out_raster.GetRasterBand(1)
        z_array = np.flipud(z_array) 
        
        if is_indicator:
             z_array = np.clip(z_array, 0.0, 1.0)
             
        out_band.WriteArray(z_array)
        
        z_array[np.isnan(z_array)] = -9999
        out_band.SetNoDataValue(-9999)
        out_band.FlushCache()
        out_raster = None
        return output_path

    def processAlgorithm(self, parameters, context, feedback):
        try:
            source = self.parameterAsSource(parameters, self.INPUT, context)
            if source is None:
                raise Exception('Invalid input layer.')

            field_name = self.parameterAsString(parameters, self.VALUE_FIELD, context)
            krig_type = self.parameterAsInt(parameters, self.KRIGING_TYPE, context)
            threshold = self.parameterAsDouble(parameters, self.THRESHOLD, context)
            cell_size = self.parameterAsDouble(parameters, self.CELL_SIZE, context)
            final_output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
            
            # Mask layer string/ID needed for GDAL clip
            mask_layer_val = parameters.get(self.MASK, None)
            
            # Mask geometry for PyKrige performance acceleration
            mask_source = self.parameterAsSource(parameters, self.MASK, context)
            mask_polygons = []
            mask_bbox = None
            if mask_source is not None:
                for f in mask_source.getFeatures():
                    geom = f.geometry()
                    if mask_bbox is None:
                        mask_bbox = geom.boundingBox()
                    else:
                        mask_bbox.combineExtentWith(geom.boundingBox())
                    if geom.isMultipart():
                        polygons = geom.asMultiPolygon()
                    else:
                        polygons = [geom.asPolygon()]
                    for poly in polygons:
                        if poly and len(poly) > 0:
                            mask_polygons.append([(pt.x(), pt.y()) for pt in poly[0]])

            is_indicator = (krig_type == 1)

            points_x, points_y, values = [], [], []
            for feature in source.getFeatures():
                geom = feature.geometry().asPoint()
                val = feature[field_name]
                if val is not None:
                    points_x.append(geom.x())
                    points_y.append(geom.y())
                    
                    if is_indicator:
                         values.append(1.0 if float(val) >= threshold else 0.0)
                    else:
                         values.append(float(val))

            if len(values) < 3:
                raise Exception("Insufficient points for Kriging.")

            initial_count = len(values)
            points_x, points_y, values = self.validate_data(points_x, points_y, values)
            
            data_stats = {
                'total_points': len(values),
                'duplicates_removed': initial_count - len(values),
                'outliers_detected': 0
            }

            qp_params = {}
            json_path = self.parameterAsFile(parameters, self.VARIOGRAM_MODEL, context)
            import json
            if json_path and os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    qp_params = json.load(f)
            else:
                 raise Exception("No Variogram parameter file provided. Run the Variography module first.")

            model_name = qp_params.get('model', 'spherical')
            nugget = float(qp_params.get('nugget', 0))
            sill = float(qp_params.get('sill', 1.0))
            v_range = float(qp_params.get('range', 10.0))
            angle = float(qp_params.get('anisotropy_angle', 0.0))
            scaling = float(qp_params.get('anisotropy_scaling', 1.0))

            feedback.pushInfo(f'Starting {self.KTYPES[krig_type]} with model: {model_name}')

            ok = OrdinaryKriging(
                points_x, points_y, values,
                variogram_model=model_name,
                variogram_parameters=[sill, v_range, nugget],
                anisotropy_scaling=scaling,
                anisotropy_angle=angle,
                verbose=False, enable_plotting=False
            )

            if mask_bbox is not None:
                min_x = mask_bbox.xMinimum()
                max_x = mask_bbox.xMaximum()
                min_y = mask_bbox.yMinimum()
                max_y = mask_bbox.yMaximum()
            else:
                min_x, max_x = min(points_x), max(points_x)
                min_y, max_y = min(points_y), max(points_y)

            grid_x = np.arange(min_x, max_x + cell_size, cell_size)
            grid_y = np.arange(min_y, max_y + cell_size, cell_size)
            
            mask_array = None
            if mask_polygons:
                feedback.pushInfo("Calculating grid geometry within polygon boundaries for optimization...")
                import matplotlib.path as mpltPath
                XX, YY = np.meshgrid(grid_x, grid_y)
                points_grid = np.vstack((XX.ravel(), YY.ravel())).T
                inside_mask = np.zeros(points_grid.shape[0], dtype=bool)
                for exterior in mask_polygons:
                    path = mpltPath.Path(exterior)
                    inside_mask |= path.contains_points(points_grid)
                mask_array = (~inside_mask).reshape(XX.shape)

            feedback.pushInfo("Calculating interpolation. This may take a few minutes...")
            
            if mask_array is not None:
                z, ss = ok.execute('masked', grid_x, grid_y, mask=mask_array)
            else:
                z, ss = ok.execute('grid', grid_x, grid_y)

            # File Management for GDAL Clipping
            temp_output = final_output_path if mask_layer_val is None else final_output_path.replace(".tif", "_temp.tif")
            crs = source.sourceCrs()
            
            self.create_raster(z.data, grid_x, grid_y, cell_size, crs, temp_output, is_indicator)

            # Native QGIS Geometric Masking (GDAL Clip)
            if mask_layer_val is not None:
                feedback.pushInfo("Applying strict geometric clipping using GDAL mask...")
                import processing
                clip_params = {
                    'INPUT': temp_output,
                    'MASK': mask_layer_val,
                    'SOURCE_CRS': crs,
                    'TARGET_CRS': crs,
                    'NODATA': -9999,
                    'ALPHA_BAND': False,
                    'CROP_TO_CUTLINE': True,
                    'KEEP_RESOLUTION': True,
                    'OPTIONS': '',
                    'DATA_TYPE': 0, # float32
                    'OUTPUT': final_output_path
                }
                # Run standard Processing gdal algorithm
                processing.run("gdal:cliprasterbymasklayer", clip_params, context=context, feedback=feedback)
                
                # Cleanup temporal raster
                try:
                     os.remove(temp_output)
                except:
                     pass

            QgsMessageLog.logMessage(f"Raster successfully generated at {final_output_path}", "Geostatistics Analysis", Qgis.Success)
            
            # Generate QP Audit Report
            try:
                # FIX: Absolute path resolution to avoid relative import faults inside processing threads
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if base_dir not in sys.path:
                    sys.path.append(base_dir)
                from reports.report_generator import QPReportGenerator
                
                report_dir = os.path.dirname(final_output_path) if final_output_path and os.path.dirname(final_output_path) else os.path.expanduser("~")
                generator = QPReportGenerator(report_dir)
                
                # For Kriging report, we don't have CV data but we can provide model parameters
                reporter_params = {
                    'model': model_name,
                    'nugget': nugget,
                    'sill': sill,
                    'range': v_range,
                    'anisotropy_angle': angle,
                    'cell_size': cell_size
                }
                generator.generate_pdf_report(reporter_params, data_stats)
            except Exception as report_err:
                QgsMessageLog.logMessage(f"Report generation failed: {report_err}.", "Geostatistics Analysis", Qgis.Warning)

            feedback.setProgress(100)
            return {self.OUTPUT: final_output_path}
            
        except Exception as e:
            QgsMessageLog.logMessage(f"Geostatistical Error: {str(e)}", "Geostatistics Analysis", Qgis.Critical)
            feedback.reportError(str(e))
            raise e
