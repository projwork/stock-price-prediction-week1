"""
Correlation Analysis Module for Financial News and Stock Price Integration Dataset (FNSPID)

This module provides functionality to analyze correlations between news sentiment scores
and stock returns, including statistical testing and correlation strength analysis.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class CorrelationAnalyzer:
    """
    A class to analyze correlations between sentiment scores and stock returns.
    """
    
    def __init__(self, 
                 date_column: str = 'trading_date',
                 return_column: str = 'daily_return',
                 sentiment_column: str = 'ensemble_sentiment_score'):
        """
        Initialize the CorrelationAnalyzer.
        
        Args:
            date_column: Column name for trading dates
            return_column: Column name for stock returns
            sentiment_column: Column name for sentiment scores
        """
        self.date_column = date_column
        self.return_column = return_column
        self.sentiment_column = sentiment_column
    
    def calculate_daily_returns(self, 
                              stock_df: pd.DataFrame,
                              price_column: str = 'Close') -> pd.DataFrame:
        """
        Calculate daily stock returns from price data.
        
        Args:
            stock_df: DataFrame with stock price data
            price_column: Column name for closing prices
            
        Returns:
            DataFrame with daily returns added
        """
        stock_df = stock_df.copy()
        
        # Ensure data is sorted by date
        stock_df = stock_df.sort_values(self.date_column)
        
        # Calculate daily returns (percentage change)
        stock_df['daily_return'] = stock_df[price_column].pct_change() * 100
        
        # Calculate additional return metrics
        stock_df['daily_return_abs'] = abs(stock_df['daily_return'])
        stock_df['return_direction'] = np.where(
            stock_df['daily_return'] > 0, 'positive',
            np.where(stock_df['daily_return'] < 0, 'negative', 'neutral')
        )
        
        # Calculate rolling volatility (20-day)
        stock_df['rolling_volatility'] = stock_df['daily_return'].rolling(window=20).std()
        
        # Calculate rolling returns
        stock_df['rolling_return_5d'] = stock_df['daily_return'].rolling(window=5).mean()
        stock_df['rolling_return_10d'] = stock_df['daily_return'].rolling(window=10).mean()
        
        return stock_df
    
    def prepare_correlation_data(self, 
                               sentiment_df: pd.DataFrame,
                               stock_df: pd.DataFrame,
                               lag_days: List[int] = [0, 1, 2, 3]) -> pd.DataFrame:
        """
        Prepare merged dataset for correlation analysis with sentiment lags.
        
        Args:
            sentiment_df: DataFrame with sentiment data aggregated by date
            stock_df: DataFrame with stock return data
            lag_days: List of lag days to test (0 = same day, 1 = next day, etc.)
            
        Returns:
            DataFrame with merged sentiment and return data
        """
        # Ensure both dataframes have the required columns
        if self.date_column not in sentiment_df.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in sentiment data")
        
        if self.date_column not in stock_df.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in stock data")
        
        # Convert date columns to the same type
        sentiment_df[self.date_column] = pd.to_datetime(sentiment_df[self.date_column])
        stock_df[self.date_column] = pd.to_datetime(stock_df[self.date_column])
        
        # Sort by date
        sentiment_df = sentiment_df.sort_values(self.date_column)
        stock_df = stock_df.sort_values(self.date_column)
        
        # Start with stock data as base
        merged_df = stock_df.copy()
        
        # Add sentiment data for different lags
        for lag in lag_days:
            # Shift sentiment data by lag days
            sentiment_shifted = sentiment_df.copy()
            sentiment_shifted[self.date_column] = sentiment_shifted[self.date_column] + pd.Timedelta(days=lag)
            
            # Merge with stock data
            suffix = f'_lag{lag}' if lag > 0 else ''
            merged_df = merged_df.merge(
                sentiment_shifted,
                on=self.date_column,
                how='left',
                suffixes=('', suffix)
            )
        
        return merged_df
    
    def calculate_correlation_matrix(self, 
                                   correlation_df: pd.DataFrame,
                                   sentiment_columns: List[str] = None,
                                   return_columns: List[str] = None,
                                   method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix between sentiment and return variables.
        
        Args:
            correlation_df: DataFrame with merged sentiment and return data
            sentiment_columns: List of sentiment columns to include
            return_columns: List of return columns to include
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            DataFrame with correlation matrix
        """
        if sentiment_columns is None:
            sentiment_columns = [col for col in correlation_df.columns if 'sentiment' in col.lower()]
        
        if return_columns is None:
            return_columns = [col for col in correlation_df.columns if 'return' in col.lower()]
        
        # Select relevant columns
        analysis_columns = sentiment_columns + return_columns
        analysis_df = correlation_df[analysis_columns].dropna()
        
        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = analysis_df.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = analysis_df.corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = analysis_df.corr(method='kendall')
        else:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
        
        return corr_matrix
    
    def test_correlation_significance(self, 
                                    correlation_df: pd.DataFrame,
                                    sentiment_col: str,
                                    return_col: str,
                                    method: str = 'pearson',
                                    alpha: float = 0.05) -> Dict:
        """
        Test statistical significance of correlation between sentiment and returns.
        
        Args:
            correlation_df: DataFrame with merged data
            sentiment_col: Sentiment column name
            return_col: Return column name
            method: Correlation method
            alpha: Significance level
            
        Returns:
            Dictionary with correlation test results
        """
        # Remove missing values
        clean_df = correlation_df[[sentiment_col, return_col]].dropna()
        
        if len(clean_df) < 3:
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'significant': False,
                'n_observations': len(clean_df),
                'method': method,
                'interpretation': 'Insufficient data'
            }
        
        x = clean_df[sentiment_col]
        y = clean_df[return_col]
        
        # Calculate correlation and p-value
        if method == 'pearson':
            correlation, p_value = pearsonr(x, y)
        elif method == 'spearman':
            correlation, p_value = spearmanr(x, y)
        elif method == 'kendall':
            correlation, p_value = kendalltau(x, y)
        else:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
        
        # Interpret correlation strength
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            strength = 'negligible'
        elif abs_corr < 0.3:
            strength = 'weak'
        elif abs_corr < 0.5:
            strength = 'moderate'
        elif abs_corr < 0.7:
            strength = 'strong'
        else:
            strength = 'very strong'
        
        # Determine direction
        direction = 'positive' if correlation > 0 else 'negative'
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'significant': p_value < alpha,
            'strength': strength,
            'direction': direction,
            'n_observations': len(clean_df),
            'method': method,
            'alpha': alpha,
            'interpretation': f"{strength.capitalize()} {direction} correlation"
        }
    
    def analyze_lag_correlations(self, 
                               correlation_df: pd.DataFrame,
                               sentiment_base_col: str = 'ensemble_sentiment_score',
                               return_col: str = 'daily_return',
                               max_lag: int = 5) -> pd.DataFrame:
        """
        Analyze correlations across different time lags.
        
        Args:
            correlation_df: DataFrame with merged data including lags
            sentiment_base_col: Base sentiment column name
            return_col: Return column name
            max_lag: Maximum lag to analyze
            
        Returns:
            DataFrame with lag correlation analysis
        """
        lag_results = []
        
        for lag in range(max_lag + 1):
            sentiment_col = f"{sentiment_base_col}_lag{lag}" if lag > 0 else sentiment_base_col
            
            if sentiment_col in correlation_df.columns:
                # Test correlation
                result = self.test_correlation_significance(
                    correlation_df, sentiment_col, return_col
                )
                result['lag_days'] = lag
                result['sentiment_column'] = sentiment_col
                result['return_column'] = return_col
                
                lag_results.append(result)
        
        return pd.DataFrame(lag_results)
    
    def rolling_correlation_analysis(self, 
                                   correlation_df: pd.DataFrame,
                                   sentiment_col: str,
                                   return_col: str,
                                   window: int = 60,
                                   min_periods: int = 30) -> pd.DataFrame:
        """
        Calculate rolling correlations over time.
        
        Args:
            correlation_df: DataFrame with merged data
            sentiment_col: Sentiment column name
            return_col: Return column name
            window: Rolling window size in days
            min_periods: Minimum number of observations required
            
        Returns:
            DataFrame with rolling correlations
        """
        # Ensure data is sorted by date
        correlation_df = correlation_df.sort_values(self.date_column)
        
        # Calculate rolling correlation
        rolling_corr = correlation_df[sentiment_col].rolling(
            window=window, min_periods=min_periods
        ).corr(correlation_df[return_col])
        
        # Create result dataframe
        rolling_df = pd.DataFrame({
            self.date_column: correlation_df[self.date_column],
            'rolling_correlation': rolling_corr,
            'window_size': window,
            'sentiment_col': sentiment_col,
            'return_col': return_col
        })
        
        # Add trend analysis
        rolling_df['correlation_trend'] = np.where(
            rolling_df['rolling_correlation'].diff() > 0, 'increasing',
            np.where(rolling_df['rolling_correlation'].diff() < 0, 'decreasing', 'stable')
        )
        
        return rolling_df.dropna()
    
    def segment_analysis(self, 
                        correlation_df: pd.DataFrame,
                        sentiment_col: str = 'ensemble_sentiment_score',
                        return_col: str = 'daily_return',
                        segments: Dict[str, tuple] = None) -> pd.DataFrame:
        """
        Analyze correlations in different market conditions or time segments.
        
        Args:
            correlation_df: DataFrame with merged data
            sentiment_col: Sentiment column name
            return_col: Return column name
            segments: Dictionary of segment definitions
            
        Returns:
            DataFrame with segment correlation analysis
        """
        if segments is None:
            # Default segments based on return volatility
            return_std = correlation_df[return_col].std()
            segments = {
                'high_volatility': (correlation_df[return_col].abs() > return_std),
                'low_volatility': (correlation_df[return_col].abs() <= return_std),
                'positive_returns': (correlation_df[return_col] > 0),
                'negative_returns': (correlation_df[return_col] < 0),
                'large_positive': (correlation_df[return_col] > return_std),
                'large_negative': (correlation_df[return_col] < -return_std)
            }
        
        segment_results = []
        
        for segment_name, condition in segments.items():
            if isinstance(condition, tuple):
                # If condition is a tuple, assume it's (start_date, end_date)
                start_date, end_date = condition
                segment_data = correlation_df[
                    (correlation_df[self.date_column] >= start_date) &
                    (correlation_df[self.date_column] <= end_date)
                ]
            else:
                # If condition is a boolean mask
                segment_data = correlation_df[condition]
            
            if len(segment_data) > 5:  # Minimum observations for meaningful analysis
                result = self.test_correlation_significance(
                    segment_data, sentiment_col, return_col
                )
                result['segment'] = segment_name
                result['segment_size'] = len(segment_data)
                segment_results.append(result)
        
        return pd.DataFrame(segment_results)
    
    def correlation_by_sentiment_level(self, 
                                     correlation_df: pd.DataFrame,
                                     sentiment_col: str = 'ensemble_sentiment_score',
                                     return_col: str = 'daily_return',
                                     quantiles: List[float] = [0.25, 0.75]) -> pd.DataFrame:
        """
        Analyze how correlations vary by sentiment intensity levels.
        
        Args:
            correlation_df: DataFrame with merged data
            sentiment_col: Sentiment column name
            return_col: Return column name
            quantiles: Quantiles to define sentiment levels
            
        Returns:
            DataFrame with sentiment level analysis
        """
        # Calculate sentiment quantiles
        q_low, q_high = correlation_df[sentiment_col].quantile(quantiles)
        
        # Define sentiment levels
        conditions = {
            'very_negative': correlation_df[sentiment_col] <= correlation_df[sentiment_col].quantile(0.1),
            'negative': (correlation_df[sentiment_col] > correlation_df[sentiment_col].quantile(0.1)) & 
                       (correlation_df[sentiment_col] <= q_low),
            'neutral': (correlation_df[sentiment_col] > q_low) & 
                      (correlation_df[sentiment_col] <= q_high),
            'positive': (correlation_df[sentiment_col] > q_high) & 
                       (correlation_df[sentiment_col] <= correlation_df[sentiment_col].quantile(0.9)),
            'very_positive': correlation_df[sentiment_col] > correlation_df[sentiment_col].quantile(0.9)
        }
        
        level_results = []
        
        for level_name, condition in conditions.items():
            level_data = correlation_df[condition]
            
            if len(level_data) > 5:
                # Calculate return statistics for this sentiment level
                returns_stats = {
                    'avg_return': level_data[return_col].mean(),
                    'return_std': level_data[return_col].std(),
                    'positive_return_pct': (level_data[return_col] > 0).mean() * 100,
                    'median_return': level_data[return_col].median()
                }
                
                result = {
                    'sentiment_level': level_name,
                    'n_observations': len(level_data),
                    'sentiment_range': (level_data[sentiment_col].min(), level_data[sentiment_col].max()),
                    'avg_sentiment': level_data[sentiment_col].mean(),
                    **returns_stats
                }
                
                level_results.append(result)
        
        return pd.DataFrame(level_results)
    
    def generate_correlation_report(self, 
                                  correlation_df: pd.DataFrame,
                                  sentiment_col: str = 'ensemble_sentiment_score',
                                  return_col: str = 'daily_return') -> Dict:
        """
        Generate comprehensive correlation analysis report.
        
        Args:
            correlation_df: DataFrame with merged data
            sentiment_col: Sentiment column name
            return_col: Return column name
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        report = {
            'data_summary': {
                'total_observations': len(correlation_df),
                'date_range': (correlation_df[self.date_column].min(), correlation_df[self.date_column].max()),
                'sentiment_stats': correlation_df[sentiment_col].describe().to_dict(),
                'return_stats': correlation_df[return_col].describe().to_dict()
            }
        }
        
        # Basic correlation test
        report['basic_correlation'] = self.test_correlation_significance(
            correlation_df, sentiment_col, return_col
        )
        
        # Lag analysis
        report['lag_analysis'] = self.analyze_lag_correlations(
            correlation_df, sentiment_col, return_col
        ).to_dict('records')
        
        # Segment analysis
        report['segment_analysis'] = self.segment_analysis(
            correlation_df, sentiment_col, return_col
        ).to_dict('records')
        
        # Sentiment level analysis
        report['sentiment_level_analysis'] = self.correlation_by_sentiment_level(
            correlation_df, sentiment_col, return_col
        ).to_dict('records')
        
        # Calculate additional statistics
        report['additional_stats'] = {
            'data_completeness': 1 - correlation_df[[sentiment_col, return_col]].isnull().any(axis=1).mean(),
            'sentiment_skewness': stats.skew(correlation_df[sentiment_col].dropna()),
            'return_skewness': stats.skew(correlation_df[return_col].dropna()),
            'sentiment_kurtosis': stats.kurtosis(correlation_df[sentiment_col].dropna()),
            'return_kurtosis': stats.kurtosis(correlation_df[return_col].dropna())
        }
        
        return report
    
    def plot_correlation_analysis(self, 
                                correlation_df: pd.DataFrame,
                                sentiment_col: str = 'ensemble_sentiment_score',
                                return_col: str = 'daily_return',
                                figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Create comprehensive correlation analysis plots.
        
        Args:
            correlation_df: DataFrame with merged data
            sentiment_col: Sentiment column name
            return_col: Return column name
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Sentiment-Return Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Remove NaN values for plotting
        plot_data = correlation_df[[sentiment_col, return_col, self.date_column]].dropna()
        
        # 1. Scatter plot with trend line
        axes[0, 0].scatter(plot_data[sentiment_col], plot_data[return_col], alpha=0.6)
        z = np.polyfit(plot_data[sentiment_col], plot_data[return_col], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(plot_data[sentiment_col], p(plot_data[sentiment_col]), "r--", alpha=0.8)
        axes[0, 0].set_xlabel('Sentiment Score')
        axes[0, 0].set_ylabel('Daily Return (%)')
        axes[0, 0].set_title('Sentiment vs Returns Scatter Plot')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Rolling correlation
        rolling_corr = self.rolling_correlation_analysis(correlation_df, sentiment_col, return_col)
        axes[0, 1].plot(rolling_corr[self.date_column], rolling_corr['rolling_correlation'])
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Rolling Correlation')
        axes[0, 1].set_title('60-Day Rolling Correlation')
        axes[0, 1].grid(True, alpha=0.3)
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Sentiment distribution
        axes[0, 2].hist(plot_data[sentiment_col], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Sentiment Score')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Sentiment Score Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Return distribution
        axes[1, 0].hist(plot_data[return_col], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Daily Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Lag correlation analysis
        lag_analysis = self.analyze_lag_correlations(correlation_df, sentiment_col, return_col)
        axes[1, 1].bar(lag_analysis['lag_days'], lag_analysis['correlation'])
        axes[1, 1].set_xlabel('Lag Days')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].set_title('Correlation by Lag')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Sentiment level returns
        level_analysis = self.correlation_by_sentiment_level(correlation_df, sentiment_col, return_col)
        axes[1, 2].bar(level_analysis['sentiment_level'], level_analysis['avg_return'])
        axes[1, 2].set_xlabel('Sentiment Level')
        axes[1, 2].set_ylabel('Average Return (%)')
        axes[1, 2].set_title('Returns by Sentiment Level')
        axes[1, 2].grid(True, alpha=0.3)
        plt.setp(axes[1, 2].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig 