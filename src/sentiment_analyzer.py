"""
Sentiment Analysis Module for Financial News and Stock Price Integration Dataset (FNSPID)

This module provides functionality to perform sentiment analysis on news headlines
using multiple approaches including TextBlob and VADER sentiment analyzers.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
import re
from collections import defaultdict

# Sentiment analysis libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    warnings.warn("TextBlob not available. Install with: pip install textblob")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    warnings.warn("VADER Sentiment not available. Install with: pip install vaderSentiment")

class SentimentAnalyzer:
    """
    A class to perform sentiment analysis on financial news headlines.
    """
    
    def __init__(self, 
                 text_column: str = 'headline',
                 include_textblob: bool = True,
                 include_vader: bool = True,
                 include_financial_lexicon: bool = True):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            text_column: Column name containing text to analyze
            include_textblob: Whether to include TextBlob sentiment analysis
            include_vader: Whether to include VADER sentiment analysis
            include_financial_lexicon: Whether to include financial lexicon-based analysis
        """
        self.text_column = text_column
        self.include_textblob = include_textblob and TEXTBLOB_AVAILABLE
        self.include_vader = include_vader and VADER_AVAILABLE
        self.include_financial_lexicon = include_financial_lexicon
        
        # Initialize VADER analyzer if available
        if self.include_vader:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Financial sentiment lexicons
        self.positive_financial_words = {
            'gain', 'gains', 'profit', 'profits', 'beat', 'beats', 'exceed', 'exceeds',
            'strong', 'growth', 'rise', 'rises', 'up', 'increase', 'increases',
            'outperform', 'outperforms', 'bullish', 'positive', 'optimistic',
            'surge', 'surges', 'rally', 'rallies', 'boost', 'boosts', 'upgrade',
            'upgrades', 'buy', 'recommendation', 'target', 'higher', 'momentum',
            'expansion', 'revenue', 'earnings', 'dividend', 'success', 'successful',
            'breakthrough', 'innovation', 'opportunity', 'opportunities'
        }
        
        self.negative_financial_words = {
            'loss', 'losses', 'miss', 'misses', 'fall', 'falls', 'decline', 'declines',
            'drop', 'drops', 'weak', 'weakness', 'down', 'decrease', 'decreases',
            'underperform', 'underperforms', 'bearish', 'negative', 'pessimistic',
            'crash', 'crashes', 'sell', 'downgrade', 'downgrades', 'lower', 'cut',
            'cuts', 'warning', 'warnings', 'concern', 'concerns', 'risk', 'risks',
            'debt', 'lawsuit', 'investigation', 'scandal', 'fraud', 'bankruptcy',
            'layoffs', 'restructuring', 'disappointing', 'missed', 'shortfall'
        }
        
        self.neutral_financial_words = {
            'maintain', 'maintains', 'hold', 'holds', 'neutral', 'steady', 'stable',
            'unchanged', 'flat', 'mixed', 'sideways', 'consolidation', 'range'
        }
        
    def analyze_textblob_sentiment(self, text_series: pd.Series) -> pd.DataFrame:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text_series: Series containing text to analyze
            
        Returns:
            DataFrame with TextBlob sentiment scores
        """
        if not self.include_textblob:
            return pd.DataFrame()
        
        sentiments = []
        for text in text_series:
            if pd.isna(text):
                sentiments.append({'polarity': 0, 'subjectivity': 0})
            else:
                blob = TextBlob(str(text))
                sentiments.append({
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                })
        
        df = pd.DataFrame(sentiments)
        df.columns = ['textblob_polarity', 'textblob_subjectivity']
        
        # Add categorical sentiment
        df['textblob_sentiment'] = pd.cut(
            df['textblob_polarity'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['negative', 'neutral', 'positive']
        )
        
        return df
    
    def analyze_vader_sentiment(self, text_series: pd.Series) -> pd.DataFrame:
        """
        Analyze sentiment using VADER.
        
        Args:
            text_series: Series containing text to analyze
            
        Returns:
            DataFrame with VADER sentiment scores
        """
        if not self.include_vader:
            return pd.DataFrame()
        
        sentiments = []
        for text in text_series:
            if pd.isna(text):
                sentiments.append({
                    'negative': 0, 'neutral': 0, 'positive': 0, 'compound': 0
                })
            else:
                scores = self.vader_analyzer.polarity_scores(str(text))
                sentiments.append(scores)
        
        df = pd.DataFrame(sentiments)
        df.columns = ['vader_negative', 'vader_neutral', 'vader_positive', 'vader_compound']
        
        # Add categorical sentiment based on compound score
        df['vader_sentiment'] = pd.cut(
            df['vader_compound'],
            bins=[-1, -0.05, 0.05, 1],
            labels=['negative', 'neutral', 'positive']
        )
        
        return df
    
    def analyze_financial_lexicon_sentiment(self, text_series: pd.Series) -> pd.DataFrame:
        """
        Analyze sentiment using financial-specific lexicon.
        
        Args:
            text_series: Series containing text to analyze
            
        Returns:
            DataFrame with financial lexicon sentiment scores
        """
        if not self.include_financial_lexicon:
            return pd.DataFrame()
        
        sentiments = []
        for text in text_series:
            if pd.isna(text):
                sentiments.append({
                    'positive_count': 0, 'negative_count': 0, 'neutral_count': 0,
                    'net_sentiment': 0, 'sentiment_intensity': 0
                })
            else:
                # Convert to lowercase and split into words
                words = re.findall(r'\b\w+\b', str(text).lower())
                
                positive_count = sum(1 for word in words if word in self.positive_financial_words)
                negative_count = sum(1 for word in words if word in self.negative_financial_words)
                neutral_count = sum(1 for word in words if word in self.neutral_financial_words)
                
                total_sentiment_words = positive_count + negative_count + neutral_count
                net_sentiment = positive_count - negative_count
                
                # Calculate intensity (normalized by total words)
                sentiment_intensity = net_sentiment / len(words) if len(words) > 0 else 0
                
                sentiments.append({
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'neutral_count': neutral_count,
                    'total_sentiment_words': total_sentiment_words,
                    'net_sentiment': net_sentiment,
                    'sentiment_intensity': sentiment_intensity
                })
        
        df = pd.DataFrame(sentiments)
        df.columns = [f'financial_{col}' for col in df.columns]
        
        # Add categorical sentiment
        df['financial_sentiment'] = pd.cut(
            df['financial_net_sentiment'],
            bins=[-float('inf'), -0.5, 0.5, float('inf')],
            labels=['negative', 'neutral', 'positive']
        )
        
        return df
    
    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive sentiment analysis on the dataset.
        
        Args:
            df: DataFrame containing text to analyze
            
        Returns:
            DataFrame with sentiment analysis results
        """
        result_df = df.copy()
        text_series = df[self.text_column]
        
        # TextBlob analysis
        if self.include_textblob:
            textblob_results = self.analyze_textblob_sentiment(text_series)
            result_df = pd.concat([result_df, textblob_results], axis=1)
        
        # VADER analysis
        if self.include_vader:
            vader_results = self.analyze_vader_sentiment(text_series)
            result_df = pd.concat([result_df, vader_results], axis=1)
        
        # Financial lexicon analysis
        if self.include_financial_lexicon:
            financial_results = self.analyze_financial_lexicon_sentiment(text_series)
            result_df = pd.concat([result_df, financial_results], axis=1)
        
        # Create ensemble sentiment score
        result_df = self._create_ensemble_sentiment(result_df)
        
        return result_df
    
    def _create_ensemble_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ensemble sentiment score combining multiple methods.
        
        Args:
            df: DataFrame with individual sentiment scores
            
        Returns:
            DataFrame with ensemble sentiment score
        """
        # Collect available sentiment scores
        sentiment_scores = []
        
        if 'textblob_polarity' in df.columns:
            sentiment_scores.append(df['textblob_polarity'])
        
        if 'vader_compound' in df.columns:
            sentiment_scores.append(df['vader_compound'])
        
        if 'financial_sentiment_intensity' in df.columns:
            # Normalize financial sentiment intensity to -1 to 1 range
            normalized_financial = np.clip(df['financial_sentiment_intensity'] * 5, -1, 1)
            sentiment_scores.append(normalized_financial)
        
        if sentiment_scores:
            # Calculate ensemble score as average
            df['ensemble_sentiment_score'] = np.mean(sentiment_scores, axis=0)
            
            # Create ensemble categorical sentiment
            df['ensemble_sentiment'] = pd.cut(
                df['ensemble_sentiment_score'],
                bins=[-1, -0.1, 0.1, 1],
                labels=['negative', 'neutral', 'positive']
            )
        else:
            df['ensemble_sentiment_score'] = 0
            df['ensemble_sentiment'] = 'neutral'
        
        return df
    
    def aggregate_daily_sentiment(self, 
                                sentiment_df: pd.DataFrame,
                                date_column: str = 'trading_date') -> pd.DataFrame:
        """
        Aggregate sentiment scores by date for correlation analysis.
        
        Args:
            sentiment_df: DataFrame with sentiment analysis results
            date_column: Column containing dates for aggregation
            
        Returns:
            DataFrame with daily aggregated sentiment scores
        """
        # Prepare aggregation functions
        agg_functions = {
            'headline': 'count'  # Number of articles
        }
        
        # Add sentiment score aggregations
        sentiment_columns = [col for col in sentiment_df.columns if 'sentiment' in col.lower() and 'score' in col]
        for col in sentiment_columns:
            if col in sentiment_df.columns:
                agg_functions[col] = ['mean', 'std', 'min', 'max']
        
        # Add categorical sentiment aggregations
        categorical_columns = [col for col in sentiment_df.columns if 'sentiment' in col.lower() and 'score' not in col and col != date_column]
        for col in categorical_columns:
            if col in sentiment_df.columns:
                agg_functions[col] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'neutral'
        
        # Perform aggregation
        daily_sentiment = sentiment_df.groupby(date_column).agg(agg_functions)
        
        # Flatten column names
        new_columns = []
        for col in daily_sentiment.columns:
            if isinstance(col, tuple):
                if col[1]:  # If there's a second level
                    new_columns.append(f"{col[0]}_{col[1]}")
                else:
                    new_columns.append(col[0])
            else:
                new_columns.append(col)
        
        daily_sentiment.columns = new_columns
        
        # Reset index
        daily_sentiment = daily_sentiment.reset_index()
        
        # Rename article count column
        if 'headline_count' in daily_sentiment.columns:
            daily_sentiment = daily_sentiment.rename(columns={'headline_count': 'daily_articles_count'})
        
        return daily_sentiment
    
    def get_sentiment_summary(self, sentiment_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of sentiment analysis.
        
        Args:
            sentiment_df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with sentiment summary statistics
        """
        summary = {
            'total_articles': len(sentiment_df),
            'sentiment_methods': []
        }
        
        # TextBlob summary
        if 'textblob_sentiment' in sentiment_df.columns:
            summary['sentiment_methods'].append('textblob')
            summary['textblob_sentiment_distribution'] = sentiment_df['textblob_sentiment'].value_counts().to_dict()
            summary['textblob_avg_polarity'] = sentiment_df['textblob_polarity'].mean()
            summary['textblob_avg_subjectivity'] = sentiment_df['textblob_subjectivity'].mean()
        
        # VADER summary
        if 'vader_sentiment' in sentiment_df.columns:
            summary['sentiment_methods'].append('vader')
            summary['vader_sentiment_distribution'] = sentiment_df['vader_sentiment'].value_counts().to_dict()
            summary['vader_avg_compound'] = sentiment_df['vader_compound'].mean()
        
        # Financial lexicon summary
        if 'financial_sentiment' in sentiment_df.columns:
            summary['sentiment_methods'].append('financial_lexicon')
            summary['financial_sentiment_distribution'] = sentiment_df['financial_sentiment'].value_counts().to_dict()
            summary['financial_avg_net_sentiment'] = sentiment_df['financial_net_sentiment'].mean()
        
        # Ensemble summary
        if 'ensemble_sentiment' in sentiment_df.columns:
            summary['ensemble_sentiment_distribution'] = sentiment_df['ensemble_sentiment'].value_counts().to_dict()
            summary['ensemble_avg_score'] = sentiment_df['ensemble_sentiment_score'].mean()
        
        return summary
    
    def analyze_sentiment_by_stock(self, 
                                 sentiment_df: pd.DataFrame,
                                 stock_column: str = 'stock') -> pd.DataFrame:
        """
        Analyze sentiment distribution by stock symbol.
        
        Args:
            sentiment_df: DataFrame with sentiment analysis results
            stock_column: Column containing stock symbols
            
        Returns:
            DataFrame with sentiment analysis by stock
        """
        if stock_column not in sentiment_df.columns:
            raise ValueError(f"Column '{stock_column}' not found in DataFrame")
        
        # Group by stock and calculate sentiment metrics
        stock_sentiment = sentiment_df.groupby(stock_column).agg({
            'headline': 'count',
            'ensemble_sentiment_score': ['mean', 'std', 'min', 'max'] if 'ensemble_sentiment_score' in sentiment_df.columns else 'count'
        })
        
        # Flatten column names
        stock_sentiment.columns = ['_'.join(col).strip() if isinstance(col, tuple) and col[1] else col for col in stock_sentiment.columns]
        
        # Add sentiment distribution
        if 'ensemble_sentiment' in sentiment_df.columns:
            sentiment_dist = sentiment_df.groupby([stock_column, 'ensemble_sentiment']).size().unstack(fill_value=0)
            sentiment_dist.columns = [f'sentiment_{col}' for col in sentiment_dist.columns]
            stock_sentiment = pd.concat([stock_sentiment, sentiment_dist], axis=1)
        
        # Reset index
        stock_sentiment = stock_sentiment.reset_index()
        
        return stock_sentiment
    
    def detect_sentiment_extremes(self, 
                                sentiment_df: pd.DataFrame,
                                score_column: str = 'ensemble_sentiment_score',
                                threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect articles with extreme sentiment scores.
        
        Args:
            sentiment_df: DataFrame with sentiment analysis results
            score_column: Column containing sentiment scores
            threshold: Standard deviation threshold for extreme detection
            
        Returns:
            DataFrame with extreme sentiment articles
        """
        if score_column not in sentiment_df.columns:
            return pd.DataFrame()
        
        # Calculate z-scores
        mean_score = sentiment_df[score_column].mean()
        std_score = sentiment_df[score_column].std()
        
        sentiment_df['sentiment_z_score'] = (sentiment_df[score_column] - mean_score) / std_score
        
        # Identify extremes
        extreme_mask = abs(sentiment_df['sentiment_z_score']) > threshold
        extremes = sentiment_df[extreme_mask].copy()
        
        # Add extreme type
        extremes['extreme_type'] = np.where(
            extremes['sentiment_z_score'] > threshold, 'very_positive',
            np.where(extremes['sentiment_z_score'] < -threshold, 'very_negative', 'normal')
        )
        
        return extremes.sort_values('sentiment_z_score', key=abs, ascending=False) 