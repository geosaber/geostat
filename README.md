# ⛏️ GeoStats — QGIS Geostatistics Plugin

Professional geostatistical toolkit for mineral resource estimation in QGIS.
Variogram modeling · Kriging interpolation · Cross-validation · Audit-trail reporting

![QGIS Version](https://img.shields.io/badge/QGIS-3.44_→_4.x-93b023?logo=qgis&logoColor=white)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![GPLv3 License](https://img.shields.io/badge/License-GPLv3-blue)
![PyQt6](https://img.shields.io/badge/Qt-6_(PyQt6)-41CD52?logo=qt&logoColor=white)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18803707.svg)](https://doi.org/10.5281/zenodo.18803707)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Building the Plugin](#building-the-plugin)
- [Compliance & Standards](#compliance--standards)
- [License & Disclaimer](#license--disclaimer)
- [Citation](#citation)

---

## Overview

**GeoStats** is an open-source QGIS plugin that provides a complete geostatistical workflow for mineral resource estimation. It is designed for **Qualified Persons (QP)** working under CRIRSCO-aligned reporting codes such as **JORC**, **NI 43-101**, and **CBRR**.

The plugin covers the full pipeline from exploratory data validation to kriging interpolation, delivering auditable PDF reports with every analysis.

---

## Key Features

### 🔬 Data Validation & QA/QC

- Automatic detection of **duplicate coordinates** (with configurable tolerance)
- **Null/NaN/Inf** value identification and removal
- Statistical **outlier detection** (IQR method with adjustable factor)
- **CRS validation** — warns if using geographic (lat/lon) instead of projected coordinates
- **Descriptive statistics**: count, mean, std, min, Q1, median, Q3, max, skewness, kurtosis, CV

### 📈 Variography

- **Experimental variogram** computation with customizable lag distance, max distance, and number of lags
- **Directional variograms** with azimuth, angular tolerance, and bandwidth parameters
- **Theoretical model fitting** with four models:
  - Spherical
  - Exponential
  - Gaussian
  - Linear (bounded)
- **Auto-fit** with automatic parameter estimation and best-model selection (lowest RMSE)
- **Variogram map** generation for 2D anisotropy visualization
- **Anisotropy detection** with automatic range-ratio classification
- Model serialization to/from **JSON** for inter-module communication

### 🗺️ Kriging Interpolation

| Method | Description |
| ------ | ----------- |
| **Ordinary Kriging (OK)** | Standard method assuming a stationary mean |
| **Universal Kriging (UK)** | Accounts for polynomial drift (regional linear/quadratic) |
| **External Drift Kriging (EDK)** | Uses an external raster variable as drift function |
| **Indicator Kriging (IK)** | Probability mapping above/below a cutoff grade |

- Configurable **cell size** and **grid generation**
- **Anisotropy support** (scaling and angle parameters)
- **Boundary masking** via polygon layer or convex hull
- **GeoTIFF raster output** via GDAL with CRS and NoData handling

### ✅ Cross-Validation

- **Leave-One-Out (LOO)** cross-validation
- Validation metrics: **ME**, **MAE**, **RMSE**, **R²**, **MSDR**
- Diagnostic plots:
  - Observed vs. Predicted scatter
  - Error histogram
  - Spatial distribution of errors

### 🖥️ Interactive UI

- **PyQt6 dialog** with embedded Matplotlib for real-time variogram modeling
- Manual parameter adjustment with instant visual feedback
- Auto-fit button with one-click model selection
- Parameter reset functionality
- Variogram map tab with anisotropy ellipse overlay
- Directional variogram visualization

### 📊 PDF Reporting

- Comprehensive **audit-trail PDF reports** with:
  - Title page
  - Input data summary
  - Variogram model details with fitted plot
  - Kriging configuration
  - Cross-validation results with diagnostic charts
  - Metadata page (software versions, timestamps)
- Designed for compliance with **JORC / NI 43-101 / CBRR** auditing requirements

### 🔗 QGIS Integration

- Registered as a **Processing Toolbox** provider
- Algorithms accessible via the Processing framework (batch mode supported)
- Automatic fallback to auto-fit when running headless

---

## Screenshots

> 🚧 *Add screenshots of the variogram dialog, kriging output, and PDF report here.*

---

## Requirements

### QGIS Environment

| Component | Version              |
| --------- | -------------------- |
| QGIS      | >= 3.28 (up to 4.x)  |
| Python    | 3.12+                |
| Qt        | 6.x (PyQt6)          |
| NumPy     | Bundled              |
| SciPy     | Bundled              |
| Matplotlib| Bundled              |

### Python Dependencies

Listed in [`requirements.txt`](requirements.txt):

```text
scikit-gstat>=1.0
pykrige>=1.7
```

> **Note:** QGIS bundles NumPy, SciPy, and Matplotlib. You only need to install `scikit-gstat` and `pykrige` manually.

### System Libraries

- **GDAL/OGR** — bundled with QGIS; used for raster output and geometry operations.

---

## Installation

### Option 1: Install from ZIP (recommended)

1. Download or [build](#building-the-plugin) the `geostats_plugin.zip` file.
2. Open QGIS → **Plugins** → **Manage and Install Plugins** → **Install from ZIP**.
3. Select the `geostats_plugin.zip` file and click **Install Plugin**.

### Option 2: Manual Installation

1. Clone or download this repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/KRIGE.git
   ```

2. Copy the folder to the QGIS plugins directory:
   - **Windows:** `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\geostats\`
   - **Linux:** `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/geostats/`
   - **macOS:** `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/geostats/`
3. Restart QGIS and enable **GeoStats** in the Plugin Manager.

### Installing Extra Dependencies

If `scikit-gstat` or `pykrige` are not available in your QGIS Python:

```bash
# Find the QGIS Python executable
# Windows (typical path):
"C:\Program Files\QGIS 3.44\apps\Python312\python.exe" -m pip install scikit-gstat pykrige

# Linux:
python3 -m pip install scikit-gstat pykrige --user
```

---

## Quick Start

### Step 1: Variogram Modeling

1. Load a **point vector layer** with a numeric Z attribute (e.g., grade, concentration).
2. Open the **Processing Toolbox** → **Variography** → **Variogram Modeling**.
3. Select your input layer and Z field.
4. The interactive dialog opens with:
   - Auto-estimated experimental variogram
   - Auto-fitted theoretical model
   - Adjustable parameters (nugget, sill, range, model type)
5. Click **Accept** — the model is saved as a JSON file.

### Step 2: Kriging Interpolation

1. In the **Processing Toolbox** → **Geostatistics** → **Kriging Interpolation**.
2. Configure:
   - **Input layer** and **Z field**
   - **Variogram model** (the JSON file from Step 1)
   - **Kriging method** (Ordinary, Universal, External Drift, or Indicator)
   - **Cell size** for the output grid
   - Optional: boundary polygon for masking, cross-validation, and PDF report generation
3. Click **Run** — output is a GeoTIFF raster loaded automatically into QGIS.

### Step 3: Review the Report

If report generation was enabled, a PDF file is created with the full audit trail: input summary, variogram parameters, kriging configuration, cross-validation metrics, and software metadata.

---

## Project Structure

```text
geostats/
├── __init__.py                     # Plugin entry point (classFactory)
├── main.py                         # Plugin lifecycle (GUI, provider registration)
├── metadata.txt                    # Official QGIS plugin metadata
├── requirements.txt                # Python dependencies
│
├── core/                           # Geostatistical engine
│   ├── variography.py              # Experimental & theoretical variograms,
│   │                               #   anisotropy detection, variogram map
│   ├── kriging_engine.py           # OK, UK, EDK, IK implementations (PyKrige backend)
│   ├── validation.py               # LOO cross-validation and metrics
│   ├── data_validation.py          # QA/QC: duplicates, nulls, outliers, CRS
│   └── grid_manager.py             # Grid generation, boundary masking, GDAL raster output
│
├── processing/                     # QGIS Processing Toolbox integration
│   ├── provider.py                 # Registers algorithms in the Toolbox
│   ├── variogram_algorithm.py      # Variogram modeling algorithm wrapper
│   └── kriging_algorithm.py        # Kriging interpolation algorithm wrapper
│
├── ui/                             # User interface
│   └── variogram_dialog.py         # Interactive PyQt6 variogram dialog (Matplotlib)
│
├── reports/                        # Output & auditing
│   └── report_generator.py         # PDF report generator (Matplotlib-based)
│
├── tests/                          # Unit tests
│   └── test_geostat_engine.py      # Pytest suite (variograms, fitting, validation, QA/QC)
│
├── build_plugin.py                 # Packaging script → geostats_plugin.zip
├── LICENSE                         # GPLv3 + geoscientific disclaimer
└── README.md                       # This file
```

---

## Running Tests

The test suite validates mathematical precision of the geostatistical engine using synthetic datasets:

```bash
# From the project root
python -m pytest tests/ -v
```

Tests cover:

- Theoretical variogram models (boundary conditions, monotonicity)
- Experimental variogram computation (omnidirectional & directional)
- Model fitting and auto-fit accuracy
- JSON serialization round-trip
- Data validation (duplicates, nulls, outliers, descriptive stats)
- Cross-validation metrics (perfect prediction, bias detection, R² range)
- Grid generation dimensions

---

## Building the Plugin

To package the plugin for distribution:

```bash
python build_plugin.py
```

This generates `geostats_plugin.zip` in the project root, ready for installation in QGIS via **Plugins → Install from ZIP**.

The build script automatically excludes development files (tests, caches, IDE configs, `.git`, etc.).

---

## Compliance & Standards

This plugin is designed to support the workflow of **Qualified Persons (QP)** under:

| Standard      | Jurisdiction              |
| ------------- | ------------------------- |
| **JORC Code** | Australia / international |
| **NI 43-101** | Canada                    |
| **CBRR**      | Brazil                    |
| **CRIRSCO**   | International template    |

Key compliance features:

- **Audit trail**: every PDF report includes input parameters, model choices, validation metrics, and software versions.
- **Reproducibility**: variogram models are serialized as JSON and can be reloaded for re-execution.
- **Transparency**: all cross-validation metrics (ME, MAE, RMSE, R², MSDR) are reported for model quality assessment.

---

## License & Disclaimer

This project is licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE) for details.

### ⚠️ Geoscientific Disclaimer

> This software is provided as a computational aid for geostatistical analysis. All outputs (variograms, kriging estimates, cross-validation metrics, and reports) are generated by automated algorithms and **MUST be reviewed by a Qualified Person (QP)** before use in mineral resource estimation, reserve classification, or any public disclosure under JORC, NI 43-101, CBRR, or equivalent CRIRSCO-aligned reporting codes.
>
> The authors assume **NO liability** for resource estimates, investment decisions, or regulatory submissions derived from outputs of this software.

---

## Citation

> Sidney Schaberle Goveia, “Geostatistical Analysis for QGIS Processing Toolbox”. Zenodo, fev. 27, 2026. doi: 10.5281/zenodo.18803707
