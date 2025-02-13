import numpy as np
from PyQt5.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterExtent,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException,
    QgsCoordinateTransform,
    QgsProject,
    QgsGeometry,
    QgsPointXY,
    QgsRasterBlock,
    QgsRasterFileWriter,
    QgsRectangle,
    Qgis
)
import processing
import shutil
from pathlib import Path

class VariogramKriging(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    FIELD = 'FIELD'
    POLYGON = 'POLYGON'
    MODEL_TYPE = 'MODEL_TYPE'
    LAG_DISTANCE = 'LAG_DISTANCE'
    N_LAGS = 'N_LAGS'
    EXTENT = 'EXTENT'
    CELL_SIZE = 'CELL_SIZE'
    OUTPUT = 'OUTPUT'

    def initAlgorithm(self, config=None):
        # Input parameters
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Input point layer'),
                [QgsProcessing.TypeVectorPoint]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD,
                self.tr('Field to interpolate'),
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.POLYGON,
                self.tr('Boundary polygon layer'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )
        self.model_types = ['spherical', 'exponential', 'gaussian', 'linear']
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODEL_TYPE,
                self.tr('Variogram model type'),
                options=self.model_types,
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.LAG_DISTANCE,
                self.tr('Lag distance (meters)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=100.0,
                minValue=0.1
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_LAGS,
                self.tr('Number of lags'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=10,
                minValue=3
            )
        )
        self.addParameter(
            QgsProcessingParameterExtent(
                self.EXTENT,
                self.tr('Interpolation extent')
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CELL_SIZE,
                self.tr('Cell size (meters)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=100.0,
                minValue=1.0
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                self.tr('Output raster')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        try:
            import gstools as gs
        except ImportError:
            raise QgsProcessingException('Install GSTools: pip install gstools')

        # Parameter retrieval
        source = self.parameterAsSource(parameters, self.INPUT, context)
        field = self.parameterAsString(parameters, self.FIELD, context)
        polygon_layer = self.parameterAsVectorLayer(parameters, self.POLYGON, context)
        model_type = self.model_types[self.parameterAsEnum(parameters, self.MODEL_TYPE, context)]
        lag_distance = self.parameterAsDouble(parameters, self.LAG_DISTANCE, context)
        n_lags = self.parameterAsInt(parameters, self.N_LAGS, context)
        extent = self.parameterAsExtent(parameters, self.EXTENT, context)
        cell_size = self.parameterAsDouble(parameters, self.CELL_SIZE, context)
        output = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        # Data validation
        features = list(source.getFeatures())
        if len(features) < 4:
            raise QgsProcessingException('Minimum 4 points required for variogram analysis')

        coords, values = [], []
        for f in features:
            geom = f.geometry()
            if geom and not geom.isEmpty():
                point = geom.asPoint()
                coords.append([point.x(), point.y()])
                values.append(f[field])
        
        coords = np.array(coords).T
        values = np.array(values)

        # Variogram analysis
        max_dist = lag_distance * n_lags
        bin_edges = np.linspace(0, max_dist, n_lags + 1)
        bin_center, gamma = gs.vario_estimate(
            pos=coords,
            field=values,
            bin_edges=bin_edges,
            sampling_dim=2
        )

        # Model fitting
        model_class = {
            'spherical': gs.Spherical,
            'exponential': gs.Exponential,
            'gaussian': gs.Gaussian,
            'linear': gs.Linear
        }[model_type]
        model = model_class(dim=2)
        model.fit_variogram(bin_center, gamma)

        # Grid creation
        xmin, xmax = extent.xMinimum(), extent.xMaximum()
        ymin, ymax = extent.yMinimum(), extent.yMaximum()
        cols = int((xmax - xmin) / cell_size) + 1
        rows = int((ymax - ymin) / cell_size) + 1

        # Mask creation
        transform = QgsCoordinateTransform(
            polygon_layer.crs(),
            source.sourceCrs(),
            QgsProject.instance()
        )
        polygon_feature = next(polygon_layer.getFeatures())
        polygon_geom = polygon_feature.geometry()
        polygon_geom.transform(transform)

        mask = np.zeros((rows, cols), dtype=bool)
        for row in range(rows):
            y = ymax - (row * cell_size) - cell_size/2
            for col in range(cols):
                x = xmin + (col * cell_size) + cell_size/2
                point = QgsGeometry.fromPointXY(QgsPointXY(x, y))
                if polygon_geom.contains(point):
                    mask[row, col] = True

        # Kriging
        x_coords = np.linspace(xmin + cell_size/2, xmax - cell_size/2, cols)
        y_coords = np.linspace(ymax - cell_size/2, ymin + cell_size/2, rows)
        xx, yy = np.meshgrid(x_coords, y_coords)
        grid_points = np.vstack([xx.ravel(), yy.ravel()])

        krige = gs.krige.Ordinary(model, cond_pos=coords, cond_val=values)
        valid_points = grid_points[:, mask.ravel()]
        krige_values, _ = krige(valid_points)

        # Output
        full_grid = np.full((rows, cols), np.nan, dtype=np.float32)
        full_grid[mask] = krige_values

        writer = QgsRasterFileWriter(output)
        writer.setOutputFormat("GTiff")
        provider = writer.createOneBandRaster(
            Qgis.Float32,
            cols,
            rows,
            QgsRectangle(xmin, ymin, xmax, ymax),
            source.sourceCrs()
        )
        block = QgsRasterBlock(Qgis.Float32, cols, rows)
        block.setData(full_grid.tobytes())
        provider.writeBlock(block, 1)
        provider.setNoDataValue(1, np.nan)

        return {self.OUTPUT: output}

    def name(self):
        return 'variogram_kriging'

    def displayName(self):
        return self.tr('Variogram Modeling & Kriging')

    def group(self):
        return self.tr('Interpolation')

    def groupId(self):
        return 'interpolation'

    def tr(self, text):
        return QCoreApplication.translate('Processing', text)

    def createInstance(self):
        return VariogramKriging()