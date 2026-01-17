"""
Data processing functions for cleaning and transforming data.

This module contains functions for common data processing tasks.
"""

import pandas as pd
import numpy as np
from typing import List, Union


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by removing spaces and special characters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with cleaned column names
        
    Examples:
    ---------
    >>> df_clean = clean_column_names(df)
    """
    df = df.copy()
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(' ', '_')
                  .str.replace('[^a-zA-Z0-9_]', '', regex=True))
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', 
                         columns: Union[List[str], None] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    strategy : str, default='drop'
        Strategy to handle missing values: 'drop', 'mean', 'median', 'mode', 'ffill', 'bfill'
    columns : list of str, optional
        Specific columns to apply strategy to. If None, applies to all columns.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values handled
        
    Examples:
    ---------
    >>> df_clean = handle_missing_values(df, strategy='mean', columns=['age', 'salary'])
    """
    df = df.copy()
    
    if columns is None:
        columns = df.columns
    
    if strategy == 'drop':
        df = df.dropna(subset=columns)
    elif strategy == 'mean':
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in columns:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None, inplace=True)
    elif strategy == 'ffill':
        df[columns] = df[columns].ffill()
    elif strategy == 'bfill':
        df[columns] = df[columns].bfill()
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    
    return df


def encode_categorical(df: pd.DataFrame, columns: List[str], 
                       method: str = 'onehot') -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list of str
        Columns to encode
    method : str, default='onehot'
        Encoding method: 'onehot' or 'label'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded categorical variables
        
    Examples:
    ---------
    >>> df_encoded = encode_categorical(df, ['category', 'type'], method='onehot')
    """
    df = df.copy()
    
    if method == 'onehot':
        df = pd.get_dummies(df, columns=columns, drop_first=True)
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in columns:
            df[col] = le.fit_transform(df[col].astype(str))
    else:
        raise ValueError(f"Unsupported encoding method: {method}")
    
    return df


def normalize_data(df: pd.DataFrame, columns: List[str], 
                  method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize numerical data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list of str
        Columns to normalize
    method : str, default='minmax'
        Normalization method: 'minmax' or 'standard'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with normalized columns
        
    Examples:
    ---------
    >>> df_norm = normalize_data(df, ['age', 'salary'], method='standard')
    """
    df = df.copy()
    
    if method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
    elif method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return df
