"""
Data Analytics Projects - Source Code Package

This package contains utility modules for data analytics projects.
"""

from .utils import load_data, missing_values_summary, remove_outliers, save_data
from .data_processing import (
    clean_column_names,
    handle_missing_values,
    encode_categorical,
    normalize_data
)
from .visualization import (
    plot_distribution,
    plot_correlation_matrix,
    plot_categorical_distribution,
    plot_scatter,
    plot_boxplot
)

__version__ = '1.0.0'

__all__ = [
    # utils
    'load_data',
    'missing_values_summary',
    'remove_outliers',
    'save_data',
    # data_processing
    'clean_column_names',
    'handle_missing_values',
    'encode_categorical',
    'normalize_data',
    # visualization
    'plot_distribution',
    'plot_correlation_matrix',
    'plot_categorical_distribution',
    'plot_scatter',
    'plot_boxplot',
]
