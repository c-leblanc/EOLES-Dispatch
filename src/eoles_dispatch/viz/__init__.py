"""Visualization package for EOLES-Dispatch runs.

Generates an interactive HTML report with Plotly charts for run inputs
(and outputs when available). Open the resulting file in any browser.

Usage:
    eoles-dispatch viz my_run
    eoles-dispatch viz run1 run2  # compare two runs
"""

from .report import generate_report

__all__ = ["generate_report"]
