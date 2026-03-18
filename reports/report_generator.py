import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from qgis.core import QgsMessageLog, Qgis

class QPReportGenerator:
    """
    Generates a professional Qualified Person (QP) Audit Report in PDF format.
    Includes technical metadata, variogram models, and cross-validation charts.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def generate_pdf_report(self, kriging_params, data_stats, cv_metrics=None, 
                             exp_lags=None, exp_gamma=None, model_lags=None, model_gamma=None,
                             observed=None, predicted=None):
        """
        Creates a multi-page PDF report with charts using Matplotlib.
        """
        report_path = os.path.join(self.output_dir, "Geostat_Audit_Report.pdf")
        
        try:
            with PdfPages(report_path) as pdf:
                # --- Page 1: Metadata & Variogram ---
                fig1 = plt.figure(figsize=(8.5, 11))
                
            with PdfPages(report_path) as pdf:
                # --- Page 1: Metadata & Variogram ---
                fig1 = plt.figure(figsize=(8.5, 11))
                
                # Title
                fig1.text(0.5, 0.96, "GEOSTATISTICS ANALYSIS - AUDIT REPORT", 
                         ha='center', fontsize=16, fontweight='bold')
                fig1.text(0.5, 0.94, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                         ha='center', fontsize=10)
                
                # Section 1: Data Validation
                fig1.text(0.1, 0.88, "1. DATA VALIDATION SUMMARY", fontweight='bold')
                stats_text = (f"Total Sample Points: {data_stats.get('total_points', 0)}\n"
                             f"Duplicate Coordinates Removed: {data_stats.get('duplicates_removed', 0)}\n"
                             f"Outliers Identified (>3x IQR): {data_stats.get('outliers_detected', 0)}")
                fig1.text(0.12, 0.86, stats_text, va='top', fontsize=10)
                
                # Section 2: Model Parameters (Moved lower to avoid overcrowding)
                fig1.text(0.1, 0.75, "2. VARIOGRAM MODEL PARAMETERS", fontweight='bold')
                params_text = (f"Model Type: {kriging_params.get('model', 'N/A')}\n"
                              f"Nugget Effect (C0): {kriging_params.get('nugget', 0):.4f}\n"
                              f"Partial Sill (C): {kriging_params.get('sill', 0):.4f}\n"
                              f"Major Range (a): {kriging_params.get('range', 0):.2f}\n"
                              f"Anisotropy Angle: {kriging_params.get('anisotropy_angle', 0):.1f}°\n"
                              f"Cell Resolution: {kriging_params.get('cell_size', 'N/A')}")
                fig1.text(0.12, 0.73, params_text, va='top', fontsize=10)

                # Section 3: Variogram Visualization
                if exp_lags is not None and len(exp_lags) > 0:
                     fig1.text(0.1, 0.55, "VARIOGRAM MODEL VISUALLY FITTED", fontweight='bold', fontsize=9)
                     ax_v = fig1.add_axes([0.15, 0.15, 0.7, 0.35])
                     ax_v.scatter(exp_lags, exp_gamma, color='black', label='Experimental', marker='o', s=20)
                     if model_lags is not None and model_gamma is not None:
                          ax_v.plot(model_lags, model_gamma, 'r-', linewidth=2, label='Theoretical')
                     ax_v.set_title("Experimental vs. Theoretical Variogram")
                     ax_v.set_xlabel("Distance (Lag)")
                     ax_v.set_ylabel("Semivariance")
                     ax_v.legend(loc='lower right', fontsize=8)
                     ax_v.grid(True, alpha=0.3)

                pdf.savefig(fig1)
                plt.close(fig1)
                
                # --- Page 2: Cross-Validation ---
                if (cv_metrics is not None) or (observed is not None and predicted is not None):
                    fig2 = plt.figure(figsize=(8.5, 11))
                    
                    fig2.text(0.5, 0.96, "GEOSTATISTICS ANALYSIS - CROSS-VALIDATION", 
                             ha='center', fontsize=14, fontweight='bold')
                    
                    # Section Title Metrics
                    fig2.text(0.1, 0.88, "3. CROSS-VALIDATION DIAGNOSTICS (LOO)", fontweight='bold')
                    
                    if cv_metrics:
                        metrics_text = (f"Mean Error (ME): {cv_metrics.get('me', 0):.4f}\n"
                                       f"Mean Absolute Error (MAE): {cv_metrics.get('mae', 0):.4f}\n"
                                       f"Root Mean Square Error (RMSE): {cv_metrics.get('rmse', 0):.4f}\n"
                                       f"R-Squared (R²): {cv_metrics.get('r2', 0):.4f}\n"
                                       f"Mean Std. Squared Error (MSDR): {cv_metrics.get('msdr', 0):.4f}")
                        fig2.text(0.12, 0.86, metrics_text, va='top', fontsize=10, linespacing=1.5)
                    
                    # Scatter Plot: Observed vs Predicted (Lowered to avoid metrics)
                    if observed is not None and predicted is not None:
                        ax_s = fig2.add_axes([0.15, 0.40, 0.7, 0.30]) # Reduced height slightly and lowered
                        min_val = min(np.min(observed), np.min(predicted))
                        max_val = max(np.max(observed), np.max(predicted))
                        
                        ax_s.scatter(observed, predicted, alpha=0.5, color='blue', edgecolors='k', s=20)
                        ax_s.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
                        ax_s.set_title("Scatter: Predicted vs Observed")
                        ax_s.set_xlabel("Observed Value")
                        ax_s.set_ylabel("Predicted Value (LOO)")
                        ax_s.grid(True, alpha=0.3)
                        
                        # Error Histogram
                        ax_h = fig2.add_axes([0.15, 0.08, 0.7, 0.22]) # Placed at the bottom
                        errors = predicted - observed
                        ax_h.hist(errors, bins=15, color='gray', alpha=0.7, edgecolor='black')
                        ax_h.set_title("Residuals Error Distribution")
                        ax_h.set_xlabel("Error (Pred - Obs)")
                        ax_h.grid(True, axis='y', alpha=0.3)
                    
                    pdf.savefig(fig2)
                    plt.close(fig2)

            QgsMessageLog.logMessage(f"Professional PDF Audit Report: {report_path}", "Geostatistics Analysis", Qgis.Success)
            return report_path
            
        except Exception as e:
            QgsMessageLog.logMessage(f"Failed to generate PDF report: {e}", "Geostatistics Analysis", Qgis.Warning)
            # Fallback to Text if PDF fails (e.g. environment issues)
            return self.generate_text_report(kriging_params, data_stats, report_path.replace(".pdf", ".txt"))

    def generate_text_report(self, kriging_params, data_stats, report_path):
        """
        Legacy text report fallback.
        """
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("      GEOSTATISTICS ANALYSIS - AUDIT TRAIL     \n")
                f.write(f"Date of Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Points Used: {data_stats.get('total_points', 0)}\n")
                f.write(f"Model: {kriging_params.get('model', 'N/A')}\n")
            return report_path
        except:
            return None
