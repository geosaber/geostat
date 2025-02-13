# geostat
QGIS Processing Toolbox tool for Variogram Modeling and Ordinary Kriging using GSTools
---
## Usage:

Step 1: Install Required Libraries
Ensure gstools and numpy are installed in QGIS's Python environment:
pip install gstools numpy

Step 2: Create the Python Script
Save the following code as variogram_kriging.py in your QGIS Processing scripts directory (e.g., ~/.local/share/QGIS/QGIS3/profiles/default/processing/scripts/)

Step 3: Using the Tool in QGIS
Open QGIS and go to the Processing Toolbox.

Navigate to Scripts > Tools and find the script named "Variogram Modeling & Ordinary Kriging".

Input Parameters:

- Input point layer: Select your point vector layer.
- Field to interpolate: Choose the numeric field for interpolation.
- Input polygon layer: Polygon Boundary (Only interpolates within polygon area)
- Variogram model type: Select the desired model (e.g., spherical).
- Lag distance: Set the distance between bins. (max_dist = lag_distance * n_lags)
- Number of lags: Define how many lag bins to use.
- Extent: Specify the output raster's extent.
- Cell size: Set the resolution of the output raster.
- Output raster: Choose the output file path.

## Notes
Dependencies: Ensure gstools and numpy are installed in your QGIS Python environment.

CRS Warning: Use a projected CRS (in meters) for accurate distance measurements.

Performance: Large datasets or small cell sizes may increase processing time.

This tool automates variogram modeling and kriging within QGIS, providing a user-friendly interface for spatial interpolation.

## Final Checklist for Future Use:
1. Input Validation:
 - Ensure your point layer has ≥4 points with meaningful variation
 - Confirm the polygon covers part of the grid
2. Performance Tips:
 - Start with coarse cell sizes (e.g., 100m) for large areas
 - Use spherical/exponential models for better convergence
3. Troubleshooting:
 - Check QGIS Log for model parameters
 - Verify output raster statistics in Properties > Information
