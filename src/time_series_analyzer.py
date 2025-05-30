import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.signal import find_peaks
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FinancialTimeSeriesAnalyzer:
    """Class to perform advanced time series analysis on financial news publication patterns."""
    
    def __init__(self, data: pd.DataFrame, date_column: str = 'date'):
        """
        Initialize the time series analyzer.
        
        Args:
            data (pd.DataFrame): The financial news dataset
            date_column (str): Column containing datetime information
        """
        self.data = data.copy()
        self.date_column = date_column
        self.prepare_time_series_data()
        
    def prepare_time_series_data(self):
        """Prepare data for time series analysis."""
        try:
            # Convert date column to datetime
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column], errors='coerce')
            
            # Remove rows with invalid dates
            self.data = self.data.dropna(subset=[self.date_column])
            
            # Sort by date
            self.data = self.data.sort_values(self.date_column)
            
            # Extract time components
            self.data['date_only'] = self.data[self.date_column].dt.date
            self.data['year'] = self.data[self.date_column].dt.year
            self.data['month'] = self.data[self.date_column].dt.month
            self.data['day'] = self.data[self.date_column].dt.day
            self.data['hour'] = self.data[self.date_column].dt.hour
            self.data['minute'] = self.data[self.date_column].dt.minute
            self.data['day_of_week'] = self.data[self.date_column].dt.dayofweek
            self.data['week_of_year'] = self.data[self.date_column].dt.isocalendar().week
            self.data['quarter'] = self.data[self.date_column].dt.quarter
            
            logger.info("Time series data preparation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in time series data preparation: {str(e)}")
            raise
    
    def analyze_publication_frequency(self) -> Dict[str, Any]:
        """
        Analyze publication frequency patterns over different time periods.
        
        Returns:
            Dict: Publication frequency analysis
        """
        frequency_analysis = {}
        
        # Daily frequency
        daily_counts = self.data.groupby('date_only').size().reset_index()
        daily_counts.columns = ['date', 'article_count']
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        frequency_analysis['daily'] = {
            'data': daily_counts,
            'avg_daily': daily_counts['article_count'].mean(),
            'std_daily': daily_counts['article_count'].std(),
            'max_daily': daily_counts['article_count'].max(),
            'min_daily': daily_counts['article_count'].min(),
            'total_days': len(daily_counts)
        }
        
        # Weekly frequency
        self.data['year_week'] = self.data[self.date_column].dt.strftime('%Y-W%U')
        weekly_counts = self.data.groupby('year_week').size().reset_index()
        weekly_counts.columns = ['year_week', 'article_count']
        
        frequency_analysis['weekly'] = {
            'data': weekly_counts,
            'avg_weekly': weekly_counts['article_count'].mean(),
            'std_weekly': weekly_counts['article_count'].std(),
            'max_weekly': weekly_counts['article_count'].max(),
            'min_weekly': weekly_counts['article_count'].min()
        }
        
        # Monthly frequency
        monthly_counts = self.data.groupby(['year', 'month']).size().reset_index()
        monthly_counts.columns = ['year', 'month', 'article_count']
        monthly_counts['date'] = pd.to_datetime(monthly_counts[['year', 'month']].assign(day=1))
        
        frequency_analysis['monthly'] = {
            'data': monthly_counts,
            'avg_monthly': monthly_counts['article_count'].mean(),
            'std_monthly': monthly_counts['article_count'].std(),
            'max_monthly': monthly_counts['article_count'].max(),
            'min_monthly': monthly_counts['article_count'].min()
        }
        
        # Hourly patterns
        hourly_patterns = self.data.groupby('hour').size().reset_index()
        hourly_patterns.columns = ['hour', 'article_count']
        
        frequency_analysis['hourly_patterns'] = {
            'data': hourly_patterns,
            'peak_hour': hourly_patterns.loc[hourly_patterns['article_count'].idxmax(), 'hour'],
            'low_hour': hourly_patterns.loc[hourly_patterns['article_count'].idxmin(), 'hour']
        }
        
        return frequency_analysis
    
    def detect_publication_spikes(self, threshold_std: float = 2.0) -> Dict[str, Any]:
        """
        Detect spikes in publication frequency that might indicate market events.
        
        Args:
            threshold_std (float): Number of standard deviations above mean to consider a spike
            
        Returns:
            Dict: Spike detection analysis
        """
        daily_counts = self.data.groupby('date_only').size().reset_index()
        daily_counts.columns = ['date', 'article_count']
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        # Calculate statistical thresholds
        mean_articles = daily_counts['article_count'].mean()
        std_articles = daily_counts['article_count'].std()
        spike_threshold = mean_articles + (threshold_std * std_articles)
        
        # Identify spikes
        spikes = daily_counts[daily_counts['article_count'] > spike_threshold].copy()
        spikes = spikes.sort_values('article_count', ascending=False)
        
        # Analyze spike patterns
        spike_analysis = {
            'threshold_used': spike_threshold,
            'mean_daily_articles': mean_articles,
            'std_daily_articles': std_articles,
            'total_spike_days': len(spikes),
            'spike_percentage': (len(spikes) / len(daily_counts)) * 100,
            'top_spike_days': spikes.head(10).to_dict('records'),
            'spike_day_of_week_distribution': self.analyze_spike_day_patterns(spikes)
        }
        
        return spike_analysis
    
    def analyze_spike_day_patterns(self, spikes: pd.DataFrame) -> Dict[str, int]:
        """
        Analyze what days of the week spikes occur most frequently.
        
        Args:
            spikes (pd.DataFrame): DataFrame containing spike dates
            
        Returns:
            Dict: Day of week distribution for spikes
        """
        if len(spikes) == 0:
            return {}
        
        spikes['day_of_week'] = spikes['date'].dt.day_name()
        day_distribution = spikes['day_of_week'].value_counts().to_dict()
        
        return day_distribution
    
    def analyze_intraday_patterns(self) -> Dict[str, Any]:
        """
        Analyze intraday publication patterns for trading insights.
        
        Returns:
            Dict: Intraday pattern analysis
        """
        # Market hours analysis (assuming US market hours: 9:30 AM - 4:00 PM ET)
        # Note: Adjust timezone as needed based on your data
        
        market_hours = self.data[
            (self.data['hour'] >= 9) & (self.data['hour'] <= 16)
        ].copy()
        
        pre_market = self.data[
            (self.data['hour'] >= 4) & (self.data['hour'] < 9)
        ].copy()
        
        after_market = self.data[
            (self.data['hour'] > 16) & (self.data['hour'] < 24)
        ].copy()
        
        overnight = self.data[
            (self.data['hour'] >= 0) & (self.data['hour'] < 4)
        ].copy()
        
        # Minute-level analysis for market hours
        minute_patterns = market_hours.groupby(['hour', 'minute']).size().reset_index()
        minute_patterns.columns = ['hour', 'minute', 'article_count']
        minute_patterns['time_label'] = minute_patterns.apply(
            lambda x: f"{x['hour']:02d}:{x['minute']:02d}", axis=1
        )
        
        # Opening and closing hour analysis
        opening_hour = market_hours[market_hours['hour'] == 9]
        closing_hour = market_hours[market_hours['hour'] == 16]
        
        intraday_analysis = {
            'market_hours_articles': len(market_hours),
            'pre_market_articles': len(pre_market),
            'after_market_articles': len(after_market),
            'overnight_articles': len(overnight),
            'market_hours_percentage': (len(market_hours) / len(self.data)) * 100,
            'opening_hour_pattern': opening_hour.groupby('minute').size().to_dict(),
            'closing_hour_pattern': closing_hour.groupby('minute').size().to_dict(),
            'peak_minute_analysis': self.get_peak_minute_analysis(minute_patterns)
        }
        
        return intraday_analysis
    
    def get_peak_minute_analysis(self, minute_patterns: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze peak minutes for news publication.
        
        Args:
            minute_patterns (pd.DataFrame): Minute-level publication data
            
        Returns:
            Dict: Peak minute analysis
        """
        if len(minute_patterns) == 0:
            return {}
        
        # Find top publishing minutes
        top_minutes = minute_patterns.nlargest(10, 'article_count')
        
        # Analyze common minute patterns (e.g., on the hour, half hour)
        minute_patterns['minute_category'] = minute_patterns['minute'].apply(
            lambda x: 'on_hour' if x == 0 else 
                     'half_hour' if x == 30 else 
                     'quarter_hour' if x in [15, 45] else 'other'
        )
        
        minute_category_dist = minute_patterns.groupby('minute_category')['article_count'].sum().to_dict()
        
        return {
            'top_publishing_minutes': top_minutes.to_dict('records'),
            'minute_category_distribution': minute_category_dist
        }
    
    def analyze_seasonal_patterns(self) -> Dict[str, Any]:
        """
        Analyze seasonal and cyclical patterns in news publication.
        
        Returns:
            Dict: Seasonal pattern analysis
        """
        seasonal_analysis = {}
        
        # Monthly patterns
        monthly_avg = self.data.groupby('month').size().reset_index()
        monthly_avg.columns = ['month', 'avg_articles']
        seasonal_analysis['monthly_patterns'] = monthly_avg.to_dict('records')
        
        # Quarterly patterns
        quarterly_avg = self.data.groupby('quarter').size().reset_index()
        quarterly_avg.columns = ['quarter', 'avg_articles']
        seasonal_analysis['quarterly_patterns'] = quarterly_avg.to_dict('records')
        
        # Day of week patterns
        dow_patterns = self.data.groupby('day_of_week').size().reset_index()
        dow_patterns.columns = ['day_of_week', 'avg_articles']
        dow_patterns['day_name'] = dow_patterns['day_of_week'].map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        })
        seasonal_analysis['day_of_week_patterns'] = dow_patterns.to_dict('records')
        
        # Week of year patterns (to identify annual cycles)
        weekly_avg = self.data.groupby('week_of_year').size().reset_index()
        weekly_avg.columns = ['week_of_year', 'avg_articles']
        seasonal_analysis['weekly_patterns'] = weekly_avg.to_dict('records')
        
        return seasonal_analysis
    
    def plot_time_series_analysis(self, figsize: Tuple[int, int] = (20, 16)):
        """
        Plot comprehensive time series analysis visualizations.
        
        Args:
            figsize: Figure size for the plots
        """
        fig = plt.figure(figsize=figsize)
        
        # Create a more complex subplot layout
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Daily publication frequency over time
        ax1 = fig.add_subplot(gs[0, :])
        daily_counts = self.data.groupby('date_only').size().reset_index()
        daily_counts.columns = ['date', 'article_count']
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        ax1.plot(daily_counts['date'], daily_counts['article_count'], alpha=0.7)
        ax1.set_title('Daily Publication Frequency Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Articles')
        ax1.grid(True, alpha=0.3)
        
        # Add moving average
        if len(daily_counts) > 7:
            daily_counts['ma_7'] = daily_counts['article_count'].rolling(window=7).mean()
            ax1.plot(daily_counts['date'], daily_counts['ma_7'], color='red', label='7-day MA')
            ax1.legend()
        
        # 2. Hourly patterns
        ax2 = fig.add_subplot(gs[1, 0])
        hourly_data = self.data.groupby('hour').size()
        ax2.bar(hourly_data.index, hourly_data.values)
        ax2.set_title('Articles by Hour of Day')
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Number of Articles')
        ax2.grid(True, alpha=0.3)
        
        # 3. Day of week patterns
        ax3 = fig.add_subplot(gs[1, 1])
        dow_data = self.data.groupby('day_of_week').size()
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax3.bar(range(7), [dow_data.get(i, 0) for i in range(7)])
        ax3.set_title('Articles by Day of Week')
        ax3.set_xlabel('Day of Week')
        ax3.set_ylabel('Number of Articles')
        ax3.set_xticks(range(7))
        ax3.set_xticklabels(dow_labels)
        ax3.grid(True, alpha=0.3)
        
        # 4. Monthly patterns
        ax4 = fig.add_subplot(gs[1, 2])
        monthly_data = self.data.groupby('month').size()
        ax4.bar(monthly_data.index, monthly_data.values)
        ax4.set_title('Articles by Month')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Number of Articles')
        ax4.grid(True, alpha=0.3)
        
        # 5. Publication spikes
        ax5 = fig.add_subplot(gs[2, :])
        spike_analysis = self.detect_publication_spikes()
        
        ax5.plot(daily_counts['date'], daily_counts['article_count'], alpha=0.7, label='Daily Count')
        ax5.axhline(y=spike_analysis['threshold_used'], color='red', linestyle='--', 
                   label=f'Spike Threshold ({spike_analysis["threshold_used"]:.1f})')
        
        # Highlight spike days
        spike_days = spike_analysis['top_spike_days']
        if spike_days:
            spike_dates = [pd.to_datetime(day['date']) for day in spike_days]
            spike_counts = [day['article_count'] for day in spike_days]
            ax5.scatter(spike_dates, spike_counts, color='red', s=50, zorder=5, label='Spike Days')
        
        ax5.set_title('Publication Spikes Detection')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Number of Articles')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Intraday patterns (heatmap)
        ax6 = fig.add_subplot(gs[3, :2])
        
        # Create hour-minute heatmap
        pivot_data = self.data.groupby(['hour', 'minute']).size().reset_index()
        pivot_data.columns = ['hour', 'minute', 'count']
        
        # Sample every 5 minutes for readability
        pivot_data = pivot_data[pivot_data['minute'] % 5 == 0]
        
        if not pivot_data.empty:
            heatmap_data = pivot_data.pivot(index='hour', columns='minute', values='count').fillna(0)
            sns.heatmap(heatmap_data, ax=ax6, cmap='YlOrRd', cbar_kws={'label': 'Article Count'})
            ax6.set_title('Intraday Publication Heatmap (5-min intervals)')
            ax6.set_xlabel('Minute')
            ax6.set_ylabel('Hour')
        
        # 7. Weekly trends
        ax7 = fig.add_subplot(gs[3, 2])
        weekly_data = self.data.groupby('week_of_year').size()
        ax7.plot(weekly_data.index, weekly_data.values)
        ax7.set_title('Articles by Week of Year')
        ax7.set_xlabel('Week Number')
        ax7.set_ylabel('Number of Articles')
        ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_comprehensive_time_series_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive time series analysis results.
        
        Returns:
            Dict: Complete time series analysis
        """
        return {
            'publication_frequency': self.analyze_publication_frequency(),
            'spike_analysis': self.detect_publication_spikes(),
            'intraday_patterns': self.analyze_intraday_patterns(),
            'seasonal_patterns': self.analyze_seasonal_patterns(),
            'data_summary': {
                'date_range': {
                    'start': str(self.data[self.date_column].min()),
                    'end': str(self.data[self.date_column].max())
                },
                'total_articles': len(self.data),
                'unique_days': self.data['date_only'].nunique(),
                'avg_articles_per_day': len(self.data) / self.data['date_only'].nunique()
            }
        } 