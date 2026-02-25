# -*- coding: utf-8 -*-
"""
Interactive Variogram Dialog
==============================
PyQt6 dialog with embedded Matplotlib for interactive variogram modeling.
Allows auto-fit with manual parameter override in real-time.

Features:
- Experimental variogram plot with theoretical model overlay
- Variogram map with anisotropy ellipse overlay
- Spin boxes for nugget, sill, range, lag, max distance
- Combo box for model type
- Direction and aniso ratio controls (when anisotropy detected)
- Auto-Fit / Reset / Accept buttons
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QLabel,
    QSplitter, QTabWidget, QWidget, QCheckBox, QDialogButtonBox,
    QMessageBox,
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from ..core.variography import (
    compute_experimental_variogram,
    compute_variogram_map,
    detect_anisotropy,
    fit_theoretical_model,
    auto_estimate_parameters,
    VariogramModel,
    MODEL_FUNCTIONS,
)


class VariogramDialog(QDialog):
    """Interactive variogram modeling dialog."""

    def __init__(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.coords = coords
        self.values = values
        self.fitted_model: Optional[VariogramModel] = None
        self.anisotropy_result: Optional[Dict[str, Any]] = None

        self.setWindowTitle("Variogram Modeling")
        self.setMinimumSize(1100, 700)

        self._init_data()
        self._init_ui()
        self._connect_signals()
        self._auto_fit()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_data(self):
        """Auto-estimate initial parameters from data."""
        self.params = auto_estimate_parameters(self.coords, self.values)
        self.anisotropy_result = detect_anisotropy(
            self.coords, self.values,
            lag_distance=self.params["lag_distance"],
            max_distance=self.params["max_distance"],
        )

    def _init_ui(self):
        """Build the dialog layout."""
        main_layout = QHBoxLayout(self)

        # ---- Left: Plot area (tabs) ----
        self.tab_widget = QTabWidget()

        # Tab 1: Variogram plot
        self.vario_fig = Figure(figsize=(7, 5), dpi=100)
        self.vario_ax = self.vario_fig.add_subplot(111)
        self.vario_canvas = FigureCanvas(self.vario_fig)
        self.vario_toolbar = NavigationToolbar(self.vario_canvas, self)

        vario_tab = QWidget()
        vl = QVBoxLayout(vario_tab)
        vl.addWidget(self.vario_toolbar)
        vl.addWidget(self.vario_canvas)
        self.tab_widget.addTab(vario_tab, "Variogram")

        # Tab 2: Variogram map
        self.map_fig = Figure(figsize=(7, 5), dpi=100)
        self.map_ax = self.map_fig.add_subplot(111)
        self.map_canvas = FigureCanvas(self.map_fig)
        self.map_toolbar = NavigationToolbar(self.map_canvas, self)

        map_tab = QWidget()
        ml = QVBoxLayout(map_tab)
        ml.addWidget(self.map_toolbar)
        ml.addWidget(self.map_canvas)
        self.tab_widget.addTab(map_tab, "Variogram Map")

        # Tab 3: Directional variograms
        self.dir_fig = Figure(figsize=(7, 5), dpi=100)
        self.dir_ax = self.dir_fig.add_subplot(111)
        self.dir_canvas = FigureCanvas(self.dir_fig)

        dir_tab = QWidget()
        dl = QVBoxLayout(dir_tab)
        dl.addWidget(self.dir_canvas)
        self.tab_widget.addTab(dir_tab, "Directional")

        # ---- Right: Controls panel ----
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)

        # Lag parameters group
        lag_group = QGroupBox("Lag Parameters")
        lag_form = QFormLayout(lag_group)

        self.lag_spin = QDoubleSpinBox()
        self.lag_spin.setRange(0.01, 1e8)
        self.lag_spin.setDecimals(2)
        self.lag_spin.setValue(self.params["lag_distance"])
        lag_form.addRow("Lag Distance:", self.lag_spin)

        self.maxdist_spin = QDoubleSpinBox()
        self.maxdist_spin.setRange(0.01, 1e8)
        self.maxdist_spin.setDecimals(2)
        self.maxdist_spin.setValue(self.params["max_distance"])
        lag_form.addRow("Max Distance:", self.maxdist_spin)

        self.nlags_spin = QSpinBox()
        self.nlags_spin.setRange(3, 50)
        self.nlags_spin.setValue(self.params["n_lags"])
        lag_form.addRow("N Lags:", self.nlags_spin)

        ctrl_layout.addWidget(lag_group)

        # Model parameters group
        model_group = QGroupBox("Model Parameters")
        model_form = QFormLayout(model_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["spherical", "exponential", "gaussian", "linear"])
        model_form.addRow("Model:", self.model_combo)

        self.nugget_spin = QDoubleSpinBox()
        self.nugget_spin.setRange(0.0, 1e8)
        self.nugget_spin.setDecimals(4)
        self.nugget_spin.setValue(self.params["nugget"])
        model_form.addRow("Nugget:", self.nugget_spin)

        self.sill_spin = QDoubleSpinBox()
        self.sill_spin.setRange(0.001, 1e8)
        self.sill_spin.setDecimals(4)
        self.sill_spin.setValue(self.params["sill"])
        model_form.addRow("Sill:", self.sill_spin)

        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(0.01, 1e8)
        self.range_spin.setDecimals(2)
        self.range_spin.setValue(self.params["range"])
        model_form.addRow("Range:", self.range_spin)

        ctrl_layout.addWidget(model_group)

        # Anisotropy group
        aniso_group = QGroupBox("Anisotropy")
        aniso_form = QFormLayout(aniso_group)

        is_aniso = (
            self.anisotropy_result is not None
            and self.anisotropy_result.get("is_anisotropic", False)
        )

        self.aniso_check = QCheckBox("Enable Anisotropy")
        self.aniso_check.setChecked(is_aniso)
        aniso_form.addRow(self.aniso_check)

        self.direction_spin = QDoubleSpinBox()
        self.direction_spin.setRange(0.0, 360.0)
        self.direction_spin.setDecimals(1)
        self.direction_spin.setSuffix("°")
        if self.anisotropy_result:
            self.direction_spin.setValue(self.anisotropy_result.get("major_direction", 0.0))
        aniso_form.addRow("Direction:", self.direction_spin)

        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(0.01, 1.0)
        self.ratio_spin.setDecimals(3)
        self.ratio_spin.setSingleStep(0.05)
        if self.anisotropy_result:
            self.ratio_spin.setValue(self.anisotropy_result.get("aniso_ratio", 1.0))
        else:
            self.ratio_spin.setValue(1.0)
        aniso_form.addRow("Ratio (minor/major):", self.ratio_spin)

        self.direction_spin.setEnabled(is_aniso)
        self.ratio_spin.setEnabled(is_aniso)
        ctrl_layout.addWidget(aniso_group)

        # Anisotropy status label
        if is_aniso and self.anisotropy_result:
            ratio = self.anisotropy_result["aniso_ratio"]
            major = self.anisotropy_result["major_direction"]
            self.aniso_label = QLabel(
                f"⚠ Anisotropy detected: ratio={ratio:.3f}, direction={major:.1f}°"
            )
            self.aniso_label.setStyleSheet("color: #e67e22; font-weight: bold;")
        else:
            self.aniso_label = QLabel("✓ No significant anisotropy detected")
            self.aniso_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        ctrl_layout.addWidget(self.aniso_label)

        # Fit info
        self.fit_label = QLabel("RMSE: —")
        ctrl_layout.addWidget(self.fit_label)

        # Buttons
        btn_layout = QHBoxLayout()

        self.auto_fit_btn = QPushButton("⚡ Auto Fit")
        self.auto_fit_btn.setToolTip("Automatically fit the best model to experimental data")
        btn_layout.addWidget(self.auto_fit_btn)

        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.setToolTip("Reset parameters to auto-estimated values")
        btn_layout.addWidget(self.reset_btn)

        ctrl_layout.addLayout(btn_layout)

        # Accept / Cancel
        ctrl_layout.addStretch()
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Accept Model")
        ctrl_layout.addWidget(self.button_box)

        # ---- Assemble splitter ----
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tab_widget)
        splitter.addWidget(ctrl_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

    def _connect_signals(self):
        """Connect UI signals to update methods."""
        self.lag_spin.valueChanged.connect(self._on_params_changed)
        self.maxdist_spin.valueChanged.connect(self._on_params_changed)
        self.nlags_spin.valueChanged.connect(self._on_params_changed)
        self.model_combo.currentTextChanged.connect(self._on_model_params_changed)
        self.nugget_spin.valueChanged.connect(self._on_model_params_changed)
        self.sill_spin.valueChanged.connect(self._on_model_params_changed)
        self.range_spin.valueChanged.connect(self._on_model_params_changed)
        self.direction_spin.valueChanged.connect(self._on_model_params_changed)
        self.ratio_spin.valueChanged.connect(self._on_model_params_changed)

        self.aniso_check.toggled.connect(self._on_aniso_toggled)
        self.auto_fit_btn.clicked.connect(self._auto_fit)
        self.reset_btn.clicked.connect(self._reset_params)
        self.button_box.accepted.connect(self._accept_model)
        self.button_box.rejected.connect(self.reject)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_params_changed(self):
        """Lag/distance params changed → recompute experimental variogram."""
        self._update_experimental()
        self._update_plot()

    def _on_model_params_changed(self):
        """Model params changed → update theoretical curve only."""
        self._update_plot()

    def _on_aniso_toggled(self, checked: bool):
        """Toggle anisotropy controls visibility."""
        self.direction_spin.setEnabled(checked)
        self.ratio_spin.setEnabled(checked)
        self._update_plot()

    def _auto_fit(self):
        """Run auto-fit: compute experimental variogram and fit best model."""
        self._update_experimental()

        if not hasattr(self, "_exp_lags") or len(self._exp_lags) < 3:
            return

        model = fit_theoretical_model(
            self._exp_lags, self._exp_semiv, self._exp_counts,
        )

        # Update UI without triggering recursive signals
        self._block_signals(True)
        self.model_combo.setCurrentText(model.model_type)
        self.nugget_spin.setValue(model.nugget)
        self.sill_spin.setValue(model.sill)
        self.range_spin.setValue(model.range_)
        self.fit_label.setText(f"RMSE: {model.fit_rmse:.6f}")
        self.fitted_model = model
        self._block_signals(False)

        self._update_plot()
        self._update_variogram_map()
        self._update_directional_plot()

    def _reset_params(self):
        """Reset all parameters to auto-estimated values."""
        self._init_data()
        self._block_signals(True)
        self.lag_spin.setValue(self.params["lag_distance"])
        self.maxdist_spin.setValue(self.params["max_distance"])
        self.nlags_spin.setValue(self.params["n_lags"])
        self.nugget_spin.setValue(self.params["nugget"])
        self.sill_spin.setValue(self.params["sill"])
        self.range_spin.setValue(self.params["range"])
        self.model_combo.setCurrentIndex(0)
        self._block_signals(False)
        self._auto_fit()

    def _accept_model(self):
        """Save the fitted model and close dialog."""
        model = VariogramModel(
            model_type=self.model_combo.currentText(),
            nugget=self.nugget_spin.value(),
            sill=self.sill_spin.value(),
            range_=self.range_spin.value(),
            direction=self.direction_spin.value() if self.aniso_check.isChecked() else 0.0,
            aniso_ratio=self.ratio_spin.value() if self.aniso_check.isChecked() else 1.0,
            is_anisotropic=self.aniso_check.isChecked(),
            lags=self._exp_lags.tolist() if hasattr(self, "_exp_lags") else [],
            semivariance=self._exp_semiv.tolist() if hasattr(self, "_exp_semiv") else [],
            pair_counts=self._exp_counts.tolist() if hasattr(self, "_exp_counts") else [],
            fit_rmse=self.fitted_model.fit_rmse if self.fitted_model else 0.0,
        )
        self.fitted_model = model
        self.accept()

    # ------------------------------------------------------------------
    # Plot updates
    # ------------------------------------------------------------------

    def _update_experimental(self):
        """Recompute experimental variogram from current parameters."""
        direction = (
            self.direction_spin.value() if self.aniso_check.isChecked() else None
        )

        self._exp_lags, self._exp_semiv, self._exp_counts = (
            compute_experimental_variogram(
                self.coords, self.values,
                lag_distance=self.lag_spin.value(),
                max_distance=self.maxdist_spin.value(),
                n_lags=self.nlags_spin.value(),
                direction=direction,
            )
        )

    def _update_plot(self):
        """Redraw the variogram plot with current parameters."""
        ax = self.vario_ax
        ax.clear()

        # Experimental points
        if hasattr(self, "_exp_lags") and len(self._exp_lags) > 0:
            sizes = np.clip(self._exp_counts / np.max(self._exp_counts) * 80 + 20, 20, 100)
            ax.scatter(
                self._exp_lags, self._exp_semiv,
                s=sizes, c="#2c3e50", edgecolors="#ecf0f1",
                linewidths=0.8, zorder=5, label="Experimental",
            )

            # Theoretical curve
            model_type = self.model_combo.currentText()
            if model_type in MODEL_FUNCTIONS:
                h_smooth = np.linspace(0, self._exp_lags[-1] * 1.2, 200)
                gamma = MODEL_FUNCTIONS[model_type](
                    h_smooth,
                    self.nugget_spin.value(),
                    self.sill_spin.value(),
                    self.range_spin.value(),
                )
                ax.plot(
                    h_smooth, gamma, "-",
                    color="#e74c3c", linewidth=2, label=f"{model_type.capitalize()} model",
                )

            # Reference lines
            ax.axhline(
                self.sill_spin.value(), color="#95a5a6",
                linestyle="--", linewidth=0.8, alpha=0.7, label="Sill",
            )
            ax.axhline(
                self.nugget_spin.value(), color="#7f8c8d",
                linestyle=":", linewidth=0.8, alpha=0.7, label="Nugget",
            )
            ax.axvline(
                self.range_spin.value(), color="#3498db",
                linestyle="--", linewidth=0.8, alpha=0.7, label="Range",
            )

        ax.set_xlabel("Lag Distance (h)")
        ax.set_ylabel("Semivariance γ(h)")
        ax.set_title("Experimental & Theoretical Variogram")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        self.vario_fig.tight_layout()
        self.vario_canvas.draw()

    def _update_variogram_map(self):
        """Draw the variogram map with anisotropy ellipse and axes overlay."""
        ax = self.map_ax
        ax.clear()

        vmap, xedges, yedges = compute_variogram_map(
            self.coords, self.values,
            max_distance=self.maxdist_spin.value(),
        )

        im = ax.imshow(
            vmap,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            origin="lower",
            cmap="RdYlBu_r",
            interpolation="nearest",
        )
        self.map_fig.colorbar(im, ax=ax, label="Semivariance")

        # Anisotropy ellipse + axes overlay
        if self.anisotropy_result and self.anisotropy_result.get("ellipse_params"):
            ep = self.anisotropy_result["ellipse_params"]
            cx, cy = ep["center_x"], ep["center_y"]
            major_len = ep["major_axis"]
            minor_len = ep["minor_axis"]
            azimuth = ep["angle"]  # Degrees, clockwise from north

            # Convert azimuth (CW from N) to math angle (CCW from E)
            angle_mpl = 90.0 - azimuth
            angle_rad = np.radians(angle_mpl)

            # ── Ellipse ──
            ellipse = Ellipse(
                (cx, cy),
                width=major_len * 2,
                height=minor_len * 2,
                angle=angle_mpl,
                fill=False,
                edgecolor="#00ff00",
                linewidth=2.0,
                linestyle="--",
                label=f"Ellipse (ratio={self.anisotropy_result['aniso_ratio']:.2f})",
            )
            ax.add_patch(ellipse)

            # ── Major axis (direction of max continuity) ──
            dx_major = major_len * np.cos(angle_rad)
            dy_major = major_len * np.sin(angle_rad)
            ax.plot(
                [cx - dx_major, cx + dx_major],
                [cy - dy_major, cy + dy_major],
                color="#00ff00", linewidth=2.0, linestyle="-",
                label=f"Major axis ({azimuth:.0f}deg)",
            )

            # ── Minor axis (perpendicular, min continuity) ──
            angle_minor_rad = angle_rad + np.pi / 2
            dx_minor = minor_len * np.cos(angle_minor_rad)
            dy_minor = minor_len * np.sin(angle_minor_rad)
            ax.plot(
                [cx - dx_minor, cx + dx_minor],
                [cy - dy_minor, cy + dy_minor],
                color="#ff4444", linewidth=1.8, linestyle="-.",
                label=f"Minor axis ({(azimuth + 90) % 360:.0f}deg)",
            )

            ax.legend(fontsize=7, loc="upper right")

        ax.set_xlabel("dx")
        ax.set_ylabel("dy")
        ax.set_title("Variogram Map")
        ax.set_aspect("equal")
        self.map_fig.tight_layout()
        self.map_canvas.draw()

    def _update_directional_plot(self):
        """Draw directional variograms."""
        ax = self.dir_ax
        ax.clear()

        if self.anisotropy_result and self.anisotropy_result.get("directional_variograms"):
            colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
            for idx, (direction, data) in enumerate(
                self.anisotropy_result["directional_variograms"].items()
            ):
                color = colors[idx % len(colors)]
                ax.plot(
                    data["lags"], data["semivariance"],
                    "o-", color=color, markersize=5,
                    label=f"Az {direction}° (range≈{data['effective_range']:.1f})",
                )

        ax.set_xlabel("Lag Distance (h)")
        ax.set_ylabel("Semivariance γ(h)")
        ax.set_title("Directional Variograms")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self.dir_fig.tight_layout()
        self.dir_canvas.draw()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _block_signals(self, block: bool):
        """Block/unblock signals from all spin boxes to prevent recursion."""
        for widget in [
            self.lag_spin, self.maxdist_spin, self.nlags_spin,
            self.nugget_spin, self.sill_spin, self.range_spin,
            self.direction_spin, self.ratio_spin, self.model_combo,
        ]:
            widget.blockSignals(block)

    def get_model(self) -> Optional[VariogramModel]:
        """Return the accepted variogram model, or None if cancelled."""
        return self.fitted_model if self.result() == QDialog.DialogCode.Accepted else None
