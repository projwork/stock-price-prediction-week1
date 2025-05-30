import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
from datetime import datetime
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FinancialEDAAnalyzer:
    """Class to perform Exploratory Data Analysis on financial news data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the EDA analyzer.
        
        Args:
            data (pd.DataFrame): The financial news dataset
        """
        self.data = data.copy()
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare data for analysis by converting date columns and creating derived features."""
        try:
            # Convert date column to datetime
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
                
            # Create headline length column
            if 'headline' in self.data.columns:
                self.data['headline_length'] = self.data['headline'].str.len()
                
            # Extract date components
            if 'date' in self.data.columns:
                self.data['year'] = self.data['date'].dt.year
                self.data['month'] = self.data['date'].dt.month
                self.data['day'] = self.data['date'].dt.day
                self.data['day_of_week'] = self.data['date'].dt.day_name()
                self.data['hour'] = self.data['date'].dt.hour
                
            logger.info("Data preparation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise
    
    def get_descriptive_statistics(self) -> Dict[str, Any]:
        """
        Get descriptive statistics for textual lengths.
        
        Returns:
            Dict: Descriptive statistics
        """
        stats = {}
        
        # Headline length statistics
        if 'headline_length' in self.data.columns:
            headline_stats = self.data['headline_length'].describe()
            stats['headline_length'] = {
                'count': int(headline_stats['count']),
                'mean': float(headline_stats['mean']),
                'std': float(headline_stats['std']),
                'min': int(headline_stats['min']),
                'max': int(headline_stats['max']),
                'median': float(headline_stats['50%']),
                'q1': float(headline_stats['25%']),
                'q3': float(headline_stats['75%'])
            }
        
        # URL length statistics if available
        if 'url' in self.data.columns:
            self.data['url_length'] = self.data['url'].str.len()
            url_stats = self.data['url_length'].describe()
            stats['url_length'] = {
                'count': int(url_stats['count']),
                'mean': float(url_stats['mean']),
                'std': float(url_stats['std']),
                'min': int(url_stats['min']),
                'max': int(url_stats['max']),
                'median': float(url_stats['50%']),
                'q1': float(url_stats['25%']),
                'q3': float(url_stats['75%'])
            }
        
        return stats
    
    def analyze_publishers(self) -> pd.DataFrame:
        """
        Analyze publishers to identify the most active ones.
        
        Returns:
            pd.DataFrame: Publisher analysis results
        """
        if 'publisher' not in self.data.columns:
            raise ValueError("Publisher column not found in data")
        
        publisher_counts = self.data['publisher'].value_counts().reset_index()
        publisher_counts.columns = ['publisher', 'article_count']
        
        # Calculate percentage
        total_articles = len(self.data)
        publisher_counts['percentage'] = (publisher_counts['article_count'] / total_articles * 100).round(2)
        
        return publisher_counts
    
    def analyze_publication_trends(self) -> Dict[str, Any]:
        """
        Analyze publication dates to identify trends over time.
        
        Returns:
            Dict: Publication trend analysis
        """
        if 'date' not in self.data.columns:
            raise ValueError("Date column not found in data")
        
        trends = {}
        
        # Daily publication counts
        daily_counts = self.data.groupby(self.data['date'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'article_count']
        trends['daily_trends'] = daily_counts
        
        # Monthly trends
        monthly_counts = self.data.groupby(['year', 'month']).size().reset_index()
        monthly_counts.columns = ['year', 'month', 'article_count']
        trends['monthly_trends'] = monthly_counts
        
        # Day of week trends
        dow_counts = self.data['day_of_week'].value_counts().reset_index()
        dow_counts.columns = ['day_of_week', 'article_count']
        # Reorder by weekday
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts['day_of_week'] = pd.Categorical(dow_counts['day_of_week'], categories=day_order, ordered=True)
        dow_counts = dow_counts.sort_values('day_of_week').reset_index(drop=True)
        trends['day_of_week_trends'] = dow_counts
        
        # Hourly trends
        hourly_counts = self.data['hour'].value_counts().sort_index().reset_index()
        hourly_counts.columns = ['hour', 'article_count']
        trends['hourly_trends'] = hourly_counts
        
        return trends
    
    def plot_headline_length_distribution(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot the distribution of headline lengths.
        
        Args:
            figsize: Figure size for the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        axes[0].hist(self.data['headline_length'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title('Distribution of Headline Lengths')
        axes[0].set_xlabel('Headline Length (characters)')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(self.data['headline_length'])
        axes[1].set_title('Headline Length Box Plot')
        axes[1].set_ylabel('Headline Length (characters)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_publisher_analysis(self, top_n: int = 15, figsize: Tuple[int, int] = (14, 8)):
        """
        Plot publisher analysis showing the most active publishers.
        
        Args:
            top_n: Number of top publishers to show
            figsize: Figure size for the plot
        """
        publisher_data = self.analyze_publishers().head(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot
        axes[0].barh(range(len(publisher_data)), publisher_data['article_count'])
        axes[0].set_yticks(range(len(publisher_data)))
        axes[0].set_yticklabels(publisher_data['publisher'])
        axes[0].set_xlabel('Number of Articles')
        axes[0].set_title(f'Top {top_n} Most Active Publishers')
        axes[0].grid(True, alpha=0.3)
        
        # Pie chart for top 10
        top_10 = publisher_data.head(10)
        axes[1].pie(top_10['article_count'], labels=top_10['publisher'], autopct='%1.1f%%')
        axes[1].set_title('Top 10 Publishers Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def plot_temporal_trends(self, figsize: Tuple[int, int] = (16, 12)):
        """
        Plot temporal trends in publication patterns.
        
        Args:
            figsize: Figure size for the plot
        """
        trends = self.analyze_publication_trends()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Daily trends (showing last 30 days)
        daily_data = trends['daily_trends'].tail(30)
        axes[0, 0].plot(daily_data['date'], daily_data['article_count'], marker='o')
        axes[0, 0].set_title('Daily Publication Trends (Last 30 Days)')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Number of Articles')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Day of week trends
        dow_data = trends['day_of_week_trends']
        axes[0, 1].bar(dow_data['day_of_week'], dow_data['article_count'])
        axes[0, 1].set_title('Articles by Day of Week')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Number of Articles')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hourly trends
        hourly_data = trends['hourly_trends']
        axes[1, 0].bar(hourly_data['hour'], hourly_data['article_count'])
        axes[1, 0].set_title('Articles by Hour of Day')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Number of Articles')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Monthly trends
        monthly_data = trends['monthly_trends']
        monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
        axes[1, 1].plot(monthly_data['date'], monthly_data['article_count'], marker='o')
        axes[1, 1].set_title('Monthly Publication Trends')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Number of Articles')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show() 