"""
Visualization functions for data analytics projects.

This module contains functions for creating common data visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


# Set default style for all plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_distribution(df: pd.DataFrame, column: str, bins: int = 30, 
                     figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot the distribution of a numerical column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name to plot
    bins : int, default=30
        Number of bins for histogram
    figsize : tuple, default=(10, 6)
        Figure size
        
    Examples:
    ---------
    >>> plot_distribution(df, 'age', bins=20)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram
    ax.hist(df[column].dropna(), bins=bins, alpha=0.7, edgecolor='black')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {column}')
    
    # Add mean and median lines
    mean_val = df[column].mean()
    median_val = df[column].median()
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10),
                           annot: bool = True) -> None:
    """
    Plot correlation matrix heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    figsize : tuple, default=(12, 10)
        Figure size
    annot : bool, default=True
        Whether to annotate cells with correlation values
        
    Examples:
    ---------
    >>> plot_correlation_matrix(df)
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numerical_df.corr()
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(df: pd.DataFrame, column: str, 
                                 top_n: Optional[int] = None,
                                 figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot the distribution of a categorical column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name to plot
    top_n : int, optional
        Show only top N categories by frequency
    figsize : tuple, default=(10, 6)
        Figure size
        
    Examples:
    ---------
    >>> plot_categorical_distribution(df, 'category', top_n=10)
    """
    value_counts = df[column].value_counts()
    
    if top_n is not None:
        value_counts = value_counts.head(top_n)
    
    plt.figure(figsize=figsize)
    value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Distribution of {column}' + (f' (Top {top_n})' if top_n else ''))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
                figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Create a scatter plot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    hue : str, optional
        Column name for color encoding
    figsize : tuple, default=(10, 6)
        Figure size
        
    Examples:
    ---------
    >>> plot_scatter(df, 'age', 'salary', hue='department')
    """
    plt.figure(figsize=figsize)
    
    if hue:
        sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.7)
    else:
        sns.scatterplot(data=df, x=x, y=y, alpha=0.7)
    
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{y} vs {x}')
    plt.tight_layout()
    plt.show()


def plot_boxplot(df: pd.DataFrame, columns: List[str], 
                figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Create box plots for multiple columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list of str
        Columns to plot
    figsize : tuple, default=(12, 6)
        Figure size
        
    Examples:
    ---------
    >>> plot_boxplot(df, ['age', 'salary', 'experience'])
    """
    fig, ax = plt.subplots(figsize=figsize)
    df[columns].boxplot(ax=ax)
    ax.set_ylabel('Value')
    ax.set_title('Box Plots')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
