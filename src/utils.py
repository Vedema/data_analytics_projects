"""
Utility functions for data analytics projects.

This module contains common utility functions used across different projects.
"""

import pandas as pd
import numpy as np
from typing import Union, List


def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    **kwargs : dict
        Additional arguments to pass to pandas read functions
        
    Returns:
    --------
    pd.DataFrame
        Loaded data as a pandas DataFrame
        
    Examples:
    ---------
    >>> df = load_data('data/raw/sample.csv')
    >>> df = load_data('data/raw/sample.xlsx', sheet_name='Sheet1')
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath, **kwargs)
    elif filepath.endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of missing values in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to analyze
        
    Returns:
    --------
    pd.DataFrame
        Summary with columns for missing count and percentage
        
    Examples:
    ---------
    >>> summary = missing_values_summary(df)
    >>> print(summary)
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    summary = pd.DataFrame({
        'Missing_Count': missing_count,
        'Missing_Percent': missing_percent
    })
    
    return summary[summary['Missing_Count'] > 0].sort_values(
        'Missing_Percent', ascending=False
    )


def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', 
                   threshold: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name to remove outliers from
    method : str, default='iqr'
        Method to use: 'iqr' or 'zscore'
    threshold : float, default=1.5
        Threshold for outlier detection (1.5 for IQR, 3 for z-score typically)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with outliers removed
        
    Examples:
    ---------
    >>> df_clean = remove_outliers(df, 'price', method='iqr')
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return df[z_scores < threshold]
    
    else:
        raise ValueError(f"Unsupported method: {method}")


def save_data(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """
    Save DataFrame to various file formats.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str
        Path where to save the file
    **kwargs : dict
        Additional arguments to pass to pandas save functions
        
    Examples:
    ---------
    >>> save_data(df, 'data/processed/cleaned_data.csv', index=False)
    >>> save_data(df, 'data/processed/cleaned_data.xlsx', index=False)
    """
    if filepath.endswith('.csv'):
        df.to_csv(filepath, **kwargs)
    elif filepath.endswith(('.xlsx', '.xls')):
        df.to_excel(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
