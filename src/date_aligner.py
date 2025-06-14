"""
Date Alignment Module for Financial News and Stock Price Integration Dataset (FNSPID)

This module provides functionality to align dates between news and stock datasets,
normalizing timestamps to ensure proper correlation analysis.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import pytz
from typing import Dict, List, Tuple, Optional
import warnings

class DateAligner:
    """
    A class to handle date alignment between financial news and stock price datasets.
    """
    
    def __init__(self, 
                 news_date_column: str = 'date',
                 stock_date_column: str = 'Date',
                 timezone: str = 'US/Eastern'):
        """
        Initialize the DateAligner.
        
        Args:
            news_date_column: Column name for dates in news dataset
            stock_date_column: Column name for dates in stock dataset
            timezone: Timezone for normalization (default: US/Eastern for US markets)
        """
        self.news_date_column = news_date_column
        self.stock_date_column = stock_date_column
        self.timezone = pytz.timezone(timezone)
        
    def normalize_news_dates(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize news dataset dates.
        
        Args:
            news_df: DataFrame with news data
            
        Returns:
            DataFrame with normalized dates
        """
        news_df = news_df.copy()
        
        # Convert date column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(news_df[self.news_date_column]):
            print(f"🔄 Converting {self.news_date_column} to datetime format...")
            
            # Simple, robust datetime conversion
            try:
                # Method 1: Try parsing without UTC first (handles mixed timezone formats better)
                news_df[self.news_date_column] = pd.to_datetime(news_df[self.news_date_column], 
                                                               errors='coerce')
                
                # Check success rate
                null_count = news_df[self.news_date_column].isnull().sum()
                success_count = len(news_df) - null_count
                
                print(f"✅ Initial parsing: {success_count}/{len(news_df)} dates converted")
                
                # Convert timezone-aware dates to target timezone, localize timezone-naive dates
                if success_count > 0:
                    datetime_col = news_df[self.news_date_column]
                    
                    # Check if the series has timezone info
                    if datetime_col.dt.tz is not None:
                        # Convert timezone-aware dates to target timezone
                        news_df[self.news_date_column] = datetime_col.dt.tz_convert(self.timezone)
                        print(f"✅ Converted timezone-aware dates to {self.timezone}")
                    else:
                        # Localize timezone-naive dates to target timezone
                        news_df[self.news_date_column] = datetime_col.dt.tz_localize(self.timezone)
                        print(f"✅ Localized timezone-naive dates to {self.timezone}")
                    
                    print(f"✅ Timezone conversion completed")
                
                if null_count > 0:
                    print(f"⚠️ Removing {null_count} rows with invalid dates...")
                    news_df = news_df.dropna(subset=[self.news_date_column])
                
            except Exception as e:
                print(f"❌ Datetime conversion failed: {e}")
                raise ValueError(f"Could not convert {self.news_date_column} to datetime")
        
        # Check for successful datetime conversion
        datetime_col = news_df[self.news_date_column]
        null_count = datetime_col.isnull().sum()
        total_count = len(datetime_col)
        
        print(f"📊 Datetime conversion result: {total_count - null_count}/{total_count} successful conversions")
        
        if null_count > 0:
            print(f"⚠️ Warning: {null_count} dates could not be converted. Removing invalid entries...")
            # Remove rows with invalid dates
            news_df = news_df.dropna(subset=[self.news_date_column])
            datetime_col = news_df[self.news_date_column]
        
        # Verify we have datetime data before using .dt accessor
        if not pd.api.types.is_datetime64_any_dtype(datetime_col):
            raise ValueError(f"Failed to convert {self.news_date_column} to datetime format. "
                           f"Data type: {datetime_col.dtype}")
        
        # Extract date part (remove time component for daily alignment)
        news_df['aligned_date'] = datetime_col.dt.date
        
        # Convert to trading date (if weekend, move to next Monday)
        news_df['trading_date'] = news_df['aligned_date'].apply(self._adjust_to_trading_day)
        
        # Add additional temporal features
        news_df['hour'] = datetime_col.dt.hour
        news_df['weekday'] = datetime_col.dt.dayofweek
        news_df['is_weekend'] = news_df['weekday'].isin([5, 6])
        news_df['is_market_hours'] = self._is_market_hours(datetime_col)
        
        return news_df
    
    def normalize_stock_dates(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize stock dataset dates.
        
        Args:
            stock_df: DataFrame with stock data
            
        Returns:
            DataFrame with normalized dates
        """
        stock_df = stock_df.copy()
        
        # Convert date column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(stock_df[self.stock_date_column]):
            print(f"🔄 Converting {self.stock_date_column} to datetime format...")
            # Use more flexible datetime parsing
            try:
                stock_df[self.stock_date_column] = pd.to_datetime(stock_df[self.stock_date_column], 
                                                                 format='mixed', 
                                                                 errors='coerce')
            except (ValueError, TypeError):
                # Fallback to basic parsing
                print("⚠️ Mixed format failed, trying basic parsing...")
                stock_df[self.stock_date_column] = pd.to_datetime(stock_df[self.stock_date_column], 
                                                                 errors='coerce')
        
        # Check for successful datetime conversion
        datetime_col = stock_df[self.stock_date_column]
        null_count = datetime_col.isnull().sum()
        total_count = len(datetime_col)
        
        print(f"📊 Stock datetime conversion result: {total_count - null_count}/{total_count} successful conversions")
        
        if null_count > 0:
            print(f"⚠️ Warning: {null_count} stock dates could not be converted. Removing invalid entries...")
            # Remove rows with invalid dates
            stock_df = stock_df.dropna(subset=[self.stock_date_column])
            datetime_col = stock_df[self.stock_date_column]
        
        # Verify we have datetime data before using .dt accessor
        if not pd.api.types.is_datetime64_any_dtype(datetime_col):
            raise ValueError(f"Failed to convert {self.stock_date_column} to datetime format. "
                           f"Data type: {datetime_col.dtype}")
        
        # Extract date part
        stock_df['trading_date'] = datetime_col.dt.date
        
        return stock_df
    
    def _adjust_to_trading_day(self, date_obj) -> date:
        """
        Adjust date to next trading day if it falls on weekend.
        
        Args:
            date_obj: Date object
            
        Returns:
            Adjusted date
        """
        if isinstance(date_obj, str):
            date_obj = pd.to_datetime(date_obj).date()
        
        # Convert to datetime if it's a date
        if isinstance(date_obj, date):
            dt = datetime.combine(date_obj, datetime.min.time())
        else:
            dt = date_obj
        
        # If Saturday (5) or Sunday (6), move to Monday
        weekday = dt.weekday()
        if weekday == 5:  # Saturday
            dt += timedelta(days=2)
        elif weekday == 6:  # Sunday
            dt += timedelta(days=1)
        
        return dt.date()
    
    def _is_market_hours(self, datetime_series: pd.Series) -> pd.Series:
        """
        Check if datetime falls within market hours (9:30 AM - 4:00 PM ET).
        
        Args:
            datetime_series: Series of datetime objects
            
        Returns:
            Boolean series indicating market hours
        """
        # Ensure we have datetime data
        if not pd.api.types.is_datetime64_any_dtype(datetime_series):
            print("⚠️ Warning: Non-datetime data passed to _is_market_hours, returning all False")
            return pd.Series([False] * len(datetime_series), index=datetime_series.index)
        
        # Handle timezone conversion more gracefully
        try:
            # Convert to ET timezone if not already
            if datetime_series.dt.tz is None:
                # Assume UTC if no timezone information
                datetime_series = datetime_series.dt.tz_localize('UTC', errors='coerce').dt.tz_convert(self.timezone)
            else:
                datetime_series = datetime_series.dt.tz_convert(self.timezone)
        except Exception as e:
            # If timezone conversion fails, assume already in correct timezone
            print(f"⚠️ Timezone conversion failed ({e}), assuming data is already in {self.timezone}")
        
        # Extract hour and minute safely
        try:
            hour = datetime_series.dt.hour
            minute = datetime_series.dt.minute
        except Exception as e:
            print(f"⚠️ Failed to extract hour/minute from datetime series: {e}")
            return pd.Series([False] * len(datetime_series), index=datetime_series.index)
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open_minutes = 9 * 60 + 30  # 9:30 AM in minutes
        market_close_minutes = 16 * 60     # 4:00 PM in minutes
        
        current_minutes = hour * 60 + minute
        
        return (current_minutes >= market_open_minutes) & (current_minutes <= market_close_minutes)
    
    def align_datasets(self, 
                      news_df: pd.DataFrame, 
                      stock_df: pd.DataFrame,
                      stock_symbol: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align news and stock datasets by date.
        
        Args:
            news_df: News dataset
            stock_df: Stock dataset
            stock_symbol: Stock symbol to filter news (optional)
            
        Returns:
            Tuple of (aligned_news_df, aligned_stock_df)
        """
        # Filter news by stock symbol if provided
        if stock_symbol:
            news_df = news_df[news_df['stock'] == stock_symbol].copy()
        
        # Normalize dates
        news_normalized = self.normalize_news_dates(news_df)
        stock_normalized = self.normalize_stock_dates(stock_df)
        
        # Find common dates
        news_dates = set(news_normalized['trading_date'])
        stock_dates = set(stock_normalized['trading_date'])
        common_dates = news_dates.intersection(stock_dates)
        
        # Filter to common dates
        aligned_news = news_normalized[news_normalized['trading_date'].isin(common_dates)].copy()
        aligned_stock = stock_normalized[stock_normalized['trading_date'].isin(common_dates)].copy()
        
        # Sort by date
        aligned_news = aligned_news.sort_values('trading_date')
        aligned_stock = aligned_stock.sort_values('trading_date')
        
        return aligned_news, aligned_stock
    
    def get_date_alignment_summary(self, 
                                 news_df: pd.DataFrame, 
                                 stock_df: pd.DataFrame,
                                 stock_symbol: str = None) -> Dict:
        """
        Get summary statistics about date alignment.
        
        Args:
            news_df: News dataset
            stock_df: Stock dataset
            stock_symbol: Stock symbol to filter news (optional)
            
        Returns:
            Dictionary with alignment statistics
        """
        # Filter news by stock symbol if provided
        if stock_symbol:
            news_df = news_df[news_df['stock'] == stock_symbol].copy()
        
        # Normalize dates
        news_normalized = self.normalize_news_dates(news_df)
        stock_normalized = self.normalize_stock_dates(stock_df)
        
        # Calculate statistics
        news_dates = set(news_normalized['trading_date'])
        stock_dates = set(stock_normalized['trading_date'])
        common_dates = news_dates.intersection(stock_dates)
        
        # Date ranges
        news_date_range = (news_normalized['trading_date'].min(), news_normalized['trading_date'].max())
        stock_date_range = (stock_normalized['trading_date'].min(), stock_normalized['trading_date'].max())
        
        # Market hours analysis
        market_hours_news = news_normalized['is_market_hours'].sum()
        total_news = len(news_normalized)
        
        summary = {
            'stock_symbol': stock_symbol,
            'total_news_articles': len(news_normalized),
            'total_stock_days': len(stock_normalized),
            'unique_news_dates': len(news_dates),
            'unique_stock_dates': len(stock_dates),
            'common_dates': len(common_dates),
            'alignment_ratio': len(common_dates) / max(len(news_dates), len(stock_dates)) if max(len(news_dates), len(stock_dates)) > 0 else 0,
            'news_date_range': news_date_range,
            'stock_date_range': stock_date_range,
            'news_in_market_hours': market_hours_news,
            'news_in_market_hours_pct': (market_hours_news / total_news * 100) if total_news > 0 else 0,
            'weekend_news': news_normalized['is_weekend'].sum(),
            'weekend_news_pct': (news_normalized['is_weekend'].sum() / total_news * 100) if total_news > 0 else 0
        }
        
        return summary
    
    def create_daily_news_aggregation(self, 
                                    news_df: pd.DataFrame,
                                    agg_columns: List[str] = None) -> pd.DataFrame:
        """
        Aggregate news data by trading date for correlation analysis.
        
        Args:
            news_df: News dataset with normalized dates
            agg_columns: Columns to aggregate (default: count articles)
            
        Returns:
            DataFrame with daily aggregated news data
        """
        if agg_columns is None:
            agg_columns = ['headline']
        
        # Group by trading date and aggregate
        daily_agg = news_df.groupby('trading_date').agg({
            'headline': 'count',  # Number of articles per day
            'is_market_hours': 'sum',  # Articles during market hours
            'hour': ['min', 'max', 'mean'],  # Time statistics
        }).round(2)
        
        # Flatten column names
        daily_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in daily_agg.columns]
        
        # Rename for clarity
        daily_agg = daily_agg.rename(columns={
            'headline_count': 'articles_count',
            'is_market_hours_sum': 'market_hours_articles',
            'hour_min': 'earliest_hour',
            'hour_max': 'latest_hour',
            'hour_mean': 'avg_hour'
        })
        
        # Calculate additional metrics
        daily_agg['market_hours_ratio'] = daily_agg['market_hours_articles'] / daily_agg['articles_count']
        daily_agg['market_hours_ratio'] = daily_agg['market_hours_ratio'].fillna(0)
        
        # Reset index to make trading_date a column
        daily_agg = daily_agg.reset_index()
        
        return daily_agg
    
    def validate_alignment(self, 
                         aligned_news: pd.DataFrame, 
                         aligned_stock: pd.DataFrame) -> Dict:
        """
        Validate the quality of date alignment.
        
        Args:
            aligned_news: Aligned news dataset
            aligned_stock: Aligned stock dataset
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'news_records': len(aligned_news),
            'stock_records': len(aligned_stock),
            'date_match': True,
            'missing_dates': [],
            'duplicate_dates': []
        }
        
        # Check if dates match
        news_dates = set(aligned_news['trading_date'])
        stock_dates = set(aligned_stock['trading_date'])
        
        if news_dates != stock_dates:
            validation['date_match'] = False
            validation['missing_in_news'] = list(stock_dates - news_dates)
            validation['missing_in_stock'] = list(news_dates - stock_dates)
        
        # Check for duplicates in stock data (should be unique dates)
        stock_date_counts = aligned_stock['trading_date'].value_counts()
        duplicates = stock_date_counts[stock_date_counts > 1]
        if len(duplicates) > 0:
            validation['duplicate_stock_dates'] = duplicates.to_dict()
        
        # Check data quality
        validation['news_null_dates'] = aligned_news['trading_date'].isnull().sum()
        validation['stock_null_dates'] = aligned_stock['trading_date'].isnull().sum()
        
        return validation 