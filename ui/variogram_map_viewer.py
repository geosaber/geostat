from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout
from qgis.PyQt.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

class VariogramMapDialog(QDialog):
    """
    Internal Viewer for the Variogram Map (Anisotropy Heatmap).
    Replaces the previous HTML export for a more integrated QP experience.
    """
    def __init__(self, variogram_map, max_lag, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Variogram Map - Anisotropy Analysis")
        self.resize(800, 700)
        
        self.variogram_map = variogram_map
        self.max_lag = max_lag
        
        self.init_ui()
        self.plot_map()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.figure = Figure(figsize=(8, 7), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # Add Matplotlib Toolbar for Zoom/Pan
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def plot_map(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        cmap = plt.cm.jet
        cmap.set_bad(color='white')
        
        extent = [-self.max_lag, self.max_lag, -self.max_lag, self.max_lag]
        im = ax.imshow(self.variogram_map, origin='lower', extent=extent, cmap=cmap)
        
        # Add Anisotropy Indicators (Ellipse and Axes)
        self.plot_anisotropy_indicators(ax, extent)
        
        cbar = self.figure.colorbar(im, ax=ax)
        cbar.set_label('Semivariance γ(h)')
        
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
        
        ax.set_title("Variogram Map - Anisotropy Analysis", fontsize=12, pad=15)
        ax.set_xlabel("Distance Lag East (dx)")
        ax.set_ylabel("Distance Lag North (dy)")
        
        # Help text for the QP
        ax.text(0.5, -0.15, 
                "Directions with lower values (blue) indicate greater spatial continuity (Major Axis).", 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=9, style='italic', color='gray')
        
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_anisotropy_indicators(self, ax, extent):
        """
        Calculates the principal direction of continuity (Anisotropy) 
        and draws visual indicators (Ellipse and Axes).
        """
        try:
            from matplotlib.patches import Ellipse
            
            # Find Major Axis direction (direction of minimum semivariance)
            # We sample the map in 5-degree increments
            angles = np.linspace(0, 180, 36)
            sums = []
            
            ny, nx = self.variogram_map.shape
            cy, cx = ny // 2, nx // 2
            
            # Use a radius slightly smaller than max lag to avoid edge effects
            radius = min(cy, cx) * 0.8
            
            for angle in angles:
                rad = np.radians(angle)
                # Sample points along a line through the center
                samples = []
                for r in np.linspace(1, radius, 10):
                    # Rotate relative to X-axis (East) CCW
                    ix = int(cx + r * np.cos(rad))
                    iy = int(cy + r * np.sin(rad))
                    if 0 <= ix < nx and 0 <= iy < ny:
                        samples.append(self.variogram_map[iy, ix])
                sums.append(np.mean(samples) if samples else np.inf)
            
            # Major axis is the direction of minimum sum
            major_idx = np.argmin(sums)
            major_angle = angles[major_idx]
            minor_angle = (major_angle + 90) % 180
            
            # Draw Axes
            max_r = self.max_lag * 0.95
            
            # Major Axis (Green line)
            rad_maj = np.radians(major_angle)
            ax.plot([-max_r * np.cos(rad_maj), max_r * np.cos(rad_maj)],
                    [-max_r * np.sin(rad_maj), max_r * np.sin(rad_maj)],
                    'g--', linewidth=1.5, alpha=0.8, label=f'Major Axis ({major_angle:.0f}°)')
            
            # Minor Axis (Red line)
            rad_min = np.radians(minor_angle)
            ax.plot([-max_r * np.cos(rad_min), max_r * np.cos(rad_min)],
                    [-max_r * np.sin(rad_min), max_r * np.sin(rad_min)],
                    'r:', linewidth=1.5, alpha=0.8, label=f'Minor Axis ({minor_angle:.0f}°)')
            
            # Draw Representative Ellipse (at 75% of max lag)
            # Estimating the ratio from the 'sums' (semivariance ratio is roughly inverse of distance ratio)
            # This is a visual aid, not a strict fit.
            ratio = sums[major_idx] / sums[(major_idx + 18) % 36] if sums[(major_idx + 18) % 36] > 0 else 1.0
            # Clip ratio for visualization sanity
            ratio = np.clip(ratio, 0.3, 1.0)
            
            # Length of ellipse axes in lag distance units
            width = self.max_lag * 1.5
            height = width * ratio
            
            ellipse = Ellipse((0, 0), width, height, angle=major_angle, 
                              edgecolor='yellow', facecolor='none', 
                              linestyle='-', linewidth=2.0, alpha=0.6, label='Anisotropy Ellipse')
            ax.add_patch(ellipse)
            ax.legend(loc='upper right', fontsize=8, framealpha=0.5)
            
        except Exception as e:
            print(f"Error drawing anisotropy indicators: {e}")
