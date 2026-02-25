# -*- coding: utf-8 -*-
"""
Grid Manager
==============
Grid generation, polygon boundary masking, and raster output creation.
Uses GDAL for GeoTIFF generation with CRS and NoData handling.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from osgeo import gdal, ogr, osr


def generate_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    cell_size: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Generate a regular grid within a bounding box.

    Args:
        x_min, x_max, y_min, y_max: Bounding box coordinates.
        cell_size: Grid cell size (same units as coordinates).

    Returns:
        Tuple of (grid_x, grid_y, n_cols, n_rows).
        grid_x: 1D array of X cell centers.
        grid_y: 1D array of Y cell centers.
    """
    grid_x = np.arange(x_min + cell_size / 2, x_max, cell_size)
    grid_y = np.arange(y_min + cell_size / 2, y_max, cell_size)
    n_cols = len(grid_x)
    n_rows = len(grid_y)
    return grid_x, grid_y, n_cols, n_rows


def compute_convex_hull(coords: np.ndarray) -> ogr.Geometry:
    """Compute the convex hull of a set of points.

    Args:
        coords: (N, 2) array of XY coordinates.

    Returns:
        OGR Geometry (Polygon) representing the convex hull.
    """
    multipoint = ogr.Geometry(ogr.wkbMultiPoint)
    for x, y in coords:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(float(x), float(y))
        multipoint.AddGeometry(point)
    return multipoint.ConvexHull()


def create_boundary_mask(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    boundary_geom: ogr.Geometry,
) -> np.ndarray:
    """Create a binary mask from a boundary polygon.

    Points inside the boundary = True, outside = False.

    Args:
        grid_x: 1D array of X grid coordinates (cell centers).
        grid_y: 1D array of Y grid coordinates (cell centers).
        boundary_geom: OGR Geometry (Polygon/MultiPolygon).

    Returns:
        (n_rows, n_cols) boolean mask array.
    """
    n_rows = len(grid_y)
    n_cols = len(grid_x)
    mask = np.zeros((n_rows, n_cols), dtype=bool)

    for j in range(n_rows):
        for i in range(n_cols):
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(float(grid_x[i]), float(grid_y[j]))
            if boundary_geom.Contains(point):
                mask[j, i] = True

    return mask


def boundary_from_layer_wkt(wkt_geom: str) -> ogr.Geometry:
    """Create OGR Geometry from WKT string.

    Args:
        wkt_geom: Polygon geometry in WKT format.

    Returns:
        OGR Geometry.
    """
    return ogr.CreateGeometryFromWkt(wkt_geom)


def apply_mask(
    data: np.ndarray,
    mask: np.ndarray,
    nodata: float = -9999.0,
) -> np.ndarray:
    """Apply a boundary mask to a data array.

    Args:
        data: (n_rows, n_cols) data array.
        mask: (n_rows, n_cols) boolean mask (True = inside boundary).
        nodata: Value to assign outside the boundary.

    Returns:
        Masked data array with nodata where mask is False.
    """
    result = data.copy()
    result[~mask] = nodata
    return result


def create_output_raster(
    filepath: str,
    data: np.ndarray,
    x_min: float,
    y_max: float,
    cell_size: float,
    crs_wkt: str,
    nodata: float = -9999.0,
) -> str:
    """Write a 2D array to a GeoTIFF file using GDAL.

    Args:
        filepath: Output GeoTIFF path.
        data: (n_rows, n_cols) array of values.
        x_min: X coordinate of the left edge.
        y_max: Y coordinate of the top edge.
        cell_size: Cell size in map units.
        crs_wkt: CRS in WKT format.
        nodata: NoData value.

    Returns:
        Output filepath.
    """
    n_rows, n_cols = data.shape

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        filepath, n_cols, n_rows, 1, gdal.GDT_Float64,
        options=["COMPRESS=LZW", "TILED=YES"],
    )

    # Geotransform: [x_origin, pixel_width, 0, y_origin, 0, -pixel_height]
    ds.SetGeoTransform([x_min, cell_size, 0, y_max, 0, -cell_size])

    # CRS
    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs_wkt)
    ds.SetProjection(srs.ExportToWkt())

    # Write data (flip vertically because GDAL writes top-to-bottom)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(np.flipud(data))
    band.FlushCache()

    ds = None  # Close dataset
    return filepath
