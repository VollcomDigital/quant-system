"""Reporting module for portfolio analysis and visualization."""

from __future__ import annotations

# The DetailedPortfolioReporter implementation lives in collection_report.py.
# Expose it at package level for callers that import src.reporting.DetailedPortfolioReporter
from .collection_report import DetailedPortfolioReporter

__all__ = ["DetailedPortfolioReporter"]
