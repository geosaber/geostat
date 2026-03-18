import sys
import numpy as np
import scipy.spatial.distance as distance
from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QDoubleSpinBox, QComboBox, QPushButton, QGroupBox)
from qgis.PyQt.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from skgstat import Variogram
from pykrige.variogram_models import (spherical_variogram_model, 
                                      exponential_variogram_model, 
                                      gaussian_variogram_model, 
                                      linear_variogram_model, 
                                      power_variogram_model, 
                                      hole_effect_variogram_model)

class VariogramVisualizerDialog(QDialog):
    """
    Interactive UI for Variogram Modeling.
    Allows a Qualified Person to visually fit the theoretical model to experimental data
    using actual scikit-gstat or pykrige internal calculations.
    """
    def __init__(self, x, y, z, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Variogram Modeling - Audit Tool")
        self.resize(950, 650)
        
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.coords = np.column_stack((self.x, self.y))
        
        # Valid PyKrige/scikit-gstat models
        self.models = ['spherical', 'exponential', 'gaussian', 'linear', 'power', 'hole-effect']
        self.model_funcs = {
            'spherical': spherical_variogram_model,
            'exponential': exponential_variogram_model,
            'gaussian': gaussian_variogram_model,
            'linear': linear_variogram_model,
            'power': power_variogram_model,
            'hole-effect': hole_effect_variogram_model
        }
        self.current_variogram = None
        
        self.init_ui()
        self.compute_experimental_variogram()
        self.update_plot()

    def init_ui(self):
        layout = QHBoxLayout(self)

        # --- Left Panel: Controls ---
        controls_group = QGroupBox("Variogram Parameters")
        controls_layout = QVBoxLayout()

        # Model Selection
        controls_layout.addWidget(QLabel("Model Type:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(self.models)
        self.combo_model.setCurrentIndex(0) # Default: spherical
        self.combo_model.currentTextChanged.connect(self.update_plot)
        controls_layout.addWidget(self.combo_model)

        # Lags Input for Experimental Variogram
        controls_layout.addWidget(QLabel("Number of Lags:"))
        self.spin_lags = QDoubleSpinBox()
        self.spin_lags.setDecimals(0)
        self.spin_lags.setRange(5, 200)
        self.spin_lags.setValue(15)
        self.spin_lags.valueChanged.connect(self.compute_experimental_variogram)
        controls_layout.addWidget(self.spin_lags)

        # Max Distance
        max_dist = np.max(distance.pdist(self.coords)) if len(self.coords) > 1 else 100
        controls_layout.addWidget(QLabel("Max Distance:"))
        self.spin_max_lag = QDoubleSpinBox()
        self.spin_max_lag.setDecimals(2)
        self.spin_max_lag.setRange(0.1, max_dist * 1.5)
        self.spin_max_lag.setValue(max_dist / 2.0)
        self.spin_max_lag.setSingleStep(self.spin_max_lag.value() * 0.1)
        self.spin_max_lag.valueChanged.connect(self.compute_experimental_variogram)
        controls_layout.addWidget(self.spin_max_lag)

        # --- Anisotropy Controls (New) ---
        anisotropy_group = QGroupBox("Anisotropy (Directional)")
        aniso_layout = QVBoxLayout()
        
        aniso_layout.addWidget(QLabel("Azimuth (Degrees):"))
        self.spin_azimuth = QDoubleSpinBox()
        self.spin_azimuth.setRange(0, 360)
        self.spin_azimuth.setValue(0) # 0 = Omnidirectional placeholder basically unless tolerance is used
        self.spin_azimuth.setSingleStep(15)
        self.spin_azimuth.valueChanged.connect(self.compute_experimental_variogram)
        aniso_layout.addWidget(self.spin_azimuth)

        aniso_layout.addWidget(QLabel("Tolerance (Degrees):"))
        self.spin_tolerance = QDoubleSpinBox()
        self.spin_tolerance.setRange(1, 180)
        self.spin_tolerance.setValue(45) # 45 is standard wide tolerance. If az=0, tol=180 -> omni
        self.spin_tolerance.setSingleStep(5)
        self.spin_tolerance.valueChanged.connect(self.compute_experimental_variogram)
        aniso_layout.addWidget(self.spin_tolerance)
        
        anisotropy_group.setLayout(aniso_layout)
        controls_layout.addWidget(anisotropy_group)

        # Nugget Input
        controls_layout.addWidget(QLabel("Nugget Effect (C0):"))
        self.spin_nugget = QDoubleSpinBox()
        self.spin_nugget.setRange(0, 1000000)
        self.spin_nugget.setValue(np.var(self.z) * 0.1)
        self.spin_nugget.setSingleStep(max(1.0, self.spin_nugget.value() * 0.1))
        self.spin_nugget.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.spin_nugget)

        # Partial Sill Input (C)
        controls_layout.addWidget(QLabel("Partial Sill (C):"))
        self.spin_sill = QDoubleSpinBox()
        self.spin_sill.setRange(0.0001, 5000000)
        self.spin_sill.setValue(np.var(self.z) * 0.9)
        self.spin_sill.setSingleStep(max(1.0, self.spin_sill.value() * 0.1))
        self.spin_sill.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.spin_sill)

        # Range Input
        controls_layout.addWidget(QLabel("Range (a):"))
        self.spin_range = QDoubleSpinBox()
        self.spin_range.setRange(0.1, max_dist * 2)
        self.spin_range.setValue(self.spin_max_lag.value() * 0.8)
        self.spin_range.setSingleStep(self.spin_range.value() * 0.1)
        self.spin_range.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.spin_range)

        # Buttons
        self.btn_apply = QPushButton("Confirm Parameters")
        self.btn_apply.clicked.connect(self.accept)
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_apply)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group, 1)

        # --- Right Panel: Matplotlib Canvas ---
        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 3)

    def compute_experimental_variogram(self):
        """
        Uses scikit-gstat to strictly calculate the true experimental variogram.
        Now supports directional variograms using SkGStat's DirectionalVariogram if azimuth/tolerance are set.
        """
        n_lags = int(self.spin_lags.value())
        model = self.combo_model.currentText()
        maxlag = self.spin_max_lag.value()
        
        azimuth = self.spin_azimuth.value()
        tolerance = self.spin_tolerance.value()
        
        if model in ['power', 'hole-effect', 'linear']:
             model = 'spherical'
             
        try:
             # Skgstat azimuth is counter-clockwise from East (math standard).
             # We assume geological azimuth which is clockwise from North. 
             # skgstat: math_angle = 90 - azimuth
             math_angle = 90 - azimuth
             
             # Use the base Variogram class which is more robust and handles direction
             self.current_variogram = Variogram(
                 self.coords, self.z, 
                 azimuth=math_angle, tolerance=tolerance,
                 n_lags=n_lags, maxlag=maxlag, model=model
             )
        except Exception as e:
             print(f"Error calculating experimental variogram: {e}")
             self.current_variogram = None
             
        self.update_plot()

    def cleanup_resources(self):
        """
        Explicitly clear matplotlib figure and force garbage collection 
        to prevent QThread cross-talk that causes QGIS fatal crashes.
        """
        try:
            self.figure.clf()
            import gc
            gc.collect()
        except:
            pass

    def closeEvent(self, event):
        self.cleanup_resources()
        super().closeEvent(event)
        
    def accept(self):
        self.cleanup_resources()
        super().accept()

    def reject(self):
        self.cleanup_resources()
        super().reject()

    def update_plot(self):
        """
        Updates experimental plot and theoretical overlay using pure math formulas.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Get Current Parameters
        model = self.combo_model.currentText()
        sill = self.spin_sill.value()
        v_range = self.spin_range.value()
        nugget = self.spin_nugget.value()
        max_dist = self.spin_max_lag.value()

        # 1. Experimental points from scikit-gstat
        exp_lags = []
        if self.current_variogram is not None:
             exp_lags = self.current_variogram.bins
             exp_gamma = self.current_variogram.experimental
             ax.scatter(exp_lags, exp_gamma, color='black', label='Experimental Variogram', marker='o')
        else:
             ax.text(0.5, 0.5, "Insufficient data for Variogram", ha='center', transform=ax.transAxes)
             self.canvas.draw()
             return

        # 2. Theoretical Models using PyKrige's actual mathematical functions
        max_plot_lag = max(np.max(exp_lags), v_range) * 1.2 if len(exp_lags) > 0 else v_range * 1.5
        lags_smooth = np.linspace(0.001, max_plot_lag, 100)
        
        try:
            func = self.model_funcs[model]
            
            # PyKrige parameters mapping
            if model == 'linear':
                # Linear PyKrige param is [slope, nugget]
                slope = sill / v_range if v_range > 0 else 0
                params = [slope, nugget]
            elif model == 'power':
                # Power PyKrige param is [scale, exponent, nugget]
                params = [sill, 1.0, nugget]
            else:
                # Spherical, Exponential, Gaussian are [partial_sill, range, nugget]
                params = [sill, v_range, nugget]
                
            gamma_smooth = func(params, lags_smooth)
            ax.plot(lags_smooth, gamma_smooth, 'r-', linewidth=2, label=f'Theoretical Model ({model})')
        except Exception as e:
            ax.plot([], [], 'r-', label=f'Model Error: {str(e)[:20]}')

        # Total Sill happens exactly at C0 + C. For Spherical it asymptotes strictly here. 
        # For exponential it asymptotes here at 3x range.
        if model not in ['linear', 'power']:
             ax.axhline(nugget + sill, color='blue', linestyle='--', linewidth=1.5, label='Total Sill (C0+C)')
             
        ax.axhline(nugget, color='green', linestyle=':', linewidth=1.5, label='Nugget (C0)')
        
        # Visual range marker
        ax.axvline(v_range, color='grey', linestyle='-.', linewidth=1, label='Range (a)')
        
        title_extra = ""
        azimuth = self.spin_azimuth.value()
        tol = self.spin_tolerance.value()
        if tol < 180:
             title_extra = f" | Az: {azimuth}° ± {tol}°"
             
        ax.set_title(f"Variogram Modeling{title_extra}", fontsize=12)
        ax.set_xlabel("Distance (Lag)")
        ax.set_ylabel("Semivariance γ(h)")
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()

    def get_parameters(self):
        # Translate Azimuth to PyKrige Anisotropy scaling
        # PyKrige 'anisotropy_angle' is the angle of maximum range CCW from East.
        # However, for simplicity in standard kriging, if tolerance is wide (Omni), we set scaling to 1.
        azimuth = self.spin_azimuth.value()
        tol = self.spin_tolerance.value()
        
        scaling = 1.0
        angle = 0.0
        
        if tol < 90: # It's directional
             # PyKrige angle is CCW from East.
             # Geological Azimuth is CW from North.
             angle = 90 - azimuth
             # Default scaling assumption to enforce anisotropy. Real workflow needs another param for perpendicular range.
             scaling = 0.5 
             
        return {
            "model": self.combo_model.currentText(),
            "nugget": self.spin_nugget.value(),
            "sill": self.spin_sill.value(),
            "range": self.spin_range.value(),
            "lags": int(self.spin_lags.value()),
            "maxlag": self.spin_max_lag.value(),
            "azimuth": azimuth,
            "tolerance": tol,
            "anisotropy_scaling": scaling,
            "anisotropy_angle": angle,
            "exp_lags": self.current_variogram.bins.tolist() if self.current_variogram else [],
            "exp_gamma": self.current_variogram.experimental.tolist() if self.current_variogram else []
        }
