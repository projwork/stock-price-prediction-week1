"""
Financial Metrics Analyzer using PyNance and custom calculations
Comprehensive financial analysis including risk, return, and performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Handle PyNance import gracefully
try:
    import pynance as pn
    PYNANCE_AVAILABLE = True
except ImportError:
    PYNANCE_AVAILABLE = False
    print("PyNance not available. Using custom financial calculations.")

class FinancialMetricsAnalyzer:
    """
    Comprehensive Financial Metrics Analysis using PyNance and custom calculations
    """
    
    def __init__(self, data: pd.DataFrame, 
                 date_column: str = 'Date',
                 close_col: str = 'Close',
                 volume_col: str = 'Volume',
                 risk_free_rate: float = 0.02):
        """
        Initialize Financial Metrics Analyzer
        
        Args:
            data: DataFrame with price data
            date_column: Name of date column
            close_col: Name of close price column
            volume_col: Name of volume column
            risk_free_rate: Risk-free rate for calculations (default 2%)
        """
        self.data = data.copy()
        self.date_column = date_column
        self.close_col = close_col
        self.volume_col = volume_col
        self.risk_free_rate = risk_free_rate
        
        # Prepare data
        self.prepare_data()
        
        # Calculate returns
        self.returns = self.calculate_returns()
        self.log_returns = self.calculate_log_returns()
        
    def prepare_data(self):
        """Prepare and clean the data"""
        # Convert date column to datetime
        if self.date_column in self.data.columns:
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
            self.data.set_index(self.date_column, inplace=True)
        
        # Sort by date
        self.data.sort_index(inplace=True)
        
        # Remove any rows with NaN values
        self.data.dropna(subset=[self.close_col], inplace=True)
        
        print(f"Financial data prepared: {len(self.data)} records from {self.data.index.min()} to {self.data.index.max()}")
    
    def calculate_returns(self) -> pd.Series:
        """Calculate simple returns"""
        returns = self.data[self.close_col].pct_change().dropna()
        return returns
    
    def calculate_log_returns(self) -> pd.Series:
        """Calculate log returns"""
        log_returns = np.log(self.data[self.close_col] / self.data[self.close_col].shift(1)).dropna()
        return log_returns
    
    def calculate_basic_metrics(self) -> Dict[str, float]:
        """
        Calculate basic financial metrics
        
        Returns:
            Dictionary of basic metrics
        """
        metrics = {}
        
        # Price metrics
        metrics['current_price'] = self.data[self.close_col].iloc[-1]
        metrics['max_price'] = self.data[self.close_col].max()
        metrics['min_price'] = self.data[self.close_col].min()
        metrics['price_range'] = metrics['max_price'] - metrics['min_price']
        
        # Return metrics
        metrics['total_return'] = (self.data[self.close_col].iloc[-1] / self.data[self.close_col].iloc[0]) - 1
        metrics['annualized_return'] = self.calculate_annualized_return()
        metrics['average_daily_return'] = self.returns.mean()
        metrics['median_daily_return'] = self.returns.median()
        
        # Volatility metrics
        metrics['daily_volatility'] = self.returns.std()
        metrics['annualized_volatility'] = self.calculate_annualized_volatility()
        
        # Risk metrics
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio()
        metrics['sortino_ratio'] = self.calculate_sortino_ratio()
        metrics['calmar_ratio'] = self.calculate_calmar_ratio()
        
        return metrics
    
    def calculate_annualized_return(self) -> float:
        """Calculate annualized return"""
        days = (self.data.index[-1] - self.data.index[0]).days
        total_return = (self.data[self.close_col].iloc[-1] / self.data[self.close_col].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (365.25 / days) - 1
        return annualized_return
    
    def calculate_annualized_volatility(self) -> float:
        """Calculate annualized volatility"""
        daily_vol = self.returns.std()
        annualized_vol = daily_vol * np.sqrt(252)  # 252 trading days
        return annualized_vol
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        excess_return = self.calculate_annualized_return() - self.risk_free_rate
        annualized_vol = self.calculate_annualized_volatility()
        if annualized_vol == 0:
            return 0
        return excess_return / annualized_vol
    
    def calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        excess_return = self.calculate_annualized_return() - self.risk_free_rate
        downside_returns = self.returns[self.returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        if downside_deviation == 0:
            return 0
        return excess_return / downside_deviation
    
    def calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (return/max drawdown)"""
        annualized_return = self.calculate_annualized_return()
        max_drawdown = self.calculate_max_drawdown()
        if max_drawdown == 0:
            return 0
        return annualized_return / abs(max_drawdown)
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def calculate_var_cvar(self, confidence_level: float = 0.05) -> Dict[str, float]:
        """
        Calculate Value at Risk and Conditional Value at Risk
        
        Args:
            confidence_level: Confidence level (default 5%)
            
        Returns:
            Dictionary with VaR and CVaR
        """
        if PYNANCE_AVAILABLE:
            try:
                var = pn.risk.var(self.returns, alpha=confidence_level)
                cvar = pn.risk.cvar(self.returns, alpha=confidence_level)
            except:
                # Fallback calculation
                var = np.percentile(self.returns, confidence_level * 100)
                cvar = self.returns[self.returns <= var].mean()
        else:
            # Custom VaR and CVaR calculation
            var = np.percentile(self.returns, confidence_level * 100)
            cvar = self.returns[self.returns <= var].mean()
        
        return {
            'VaR': var,
            'CVaR': cvar,
            'confidence_level': confidence_level
        }
    
    def calculate_beta(self, market_returns: pd.Series) -> float:
        """
        Calculate beta relative to market
        
        Args:
            market_returns: Market returns series
            
        Returns:
            Beta coefficient
        """
        # Align the series
        aligned_returns, aligned_market = self.returns.align(market_returns, join='inner')
        
        if len(aligned_returns) < 2:
            return np.nan
        
        covariance = np.cov(aligned_returns, aligned_market)[0, 1]
        market_variance = np.var(aligned_market)
        
        if market_variance == 0:
            return np.nan
        
        return covariance / market_variance
    
    def calculate_information_ratio(self, benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio
        
        Args:
            benchmark_returns: Benchmark returns series
            
        Returns:
            Information ratio
        """
        # Align the series
        aligned_returns, aligned_benchmark = self.returns.align(benchmark_returns, join='inner')
        
        excess_returns = aligned_returns - aligned_benchmark
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            return 0
        
        return excess_returns.mean() / tracking_error
    
    def calculate_treynor_ratio(self, market_returns: pd.Series) -> float:
        """
        Calculate Treynor ratio
        
        Args:
            market_returns: Market returns series
            
        Returns:
            Treynor ratio
        """
        beta = self.calculate_beta(market_returns)
        if beta == 0 or np.isnan(beta):
            return np.nan
        
        excess_return = self.calculate_annualized_return() - self.risk_free_rate
        return excess_return / beta
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        print("📊 Calculating performance metrics...")
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self.calculate_basic_metrics())
        
        # Drawdown analysis
        metrics['max_drawdown'] = self.calculate_max_drawdown()
        metrics['current_drawdown'] = self.calculate_current_drawdown()
        
        # Risk metrics
        var_cvar = self.calculate_var_cvar()
        metrics.update(var_cvar)
        
        # Additional metrics
        metrics['skewness'] = self.returns.skew()
        metrics['kurtosis'] = self.returns.kurtosis()
        metrics['positive_days_ratio'] = (self.returns > 0).sum() / len(self.returns)
        
        # Winning/Losing streaks
        streaks = self.calculate_winning_losing_streaks()
        metrics.update(streaks)
        
        return metrics
    
    def calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.max()
        current = cumulative.iloc[-1]
        return (current - peak) / peak
    
    def calculate_winning_losing_streaks(self) -> Dict[str, int]:
        """Calculate winning and losing streaks"""
        # Create binary series: 1 for positive returns, 0 for negative
        wins = (self.returns > 0).astype(int)
        
        # Calculate streaks
        streaks = []
        current_streak = 0
        
        for win in wins:
            if win == 1:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
            else:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1
            streaks.append(current_streak)
        
        streaks = pd.Series(streaks)
        
        return {
            'max_winning_streak': streaks.max(),
            'max_losing_streak': abs(streaks.min()),
            'current_streak': streaks.iloc[-1]
        }
    
    def analyze_returns_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of returns
        
        Returns:
            Dictionary with distribution analysis
        """
        print("📈 Analyzing returns distribution...")
        
        analysis = {}
        
        # Basic statistics
        analysis['mean'] = self.returns.mean()
        analysis['std'] = self.returns.std()
        analysis['skewness'] = self.returns.skew()
        analysis['kurtosis'] = self.returns.kurtosis()
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            analysis[f'percentile_{p}'] = np.percentile(self.returns, p)
        
        # Tail analysis
        analysis['left_tail_5pct'] = (self.returns <= np.percentile(self.returns, 5)).sum()
        analysis['right_tail_5pct'] = (self.returns >= np.percentile(self.returns, 95)).sum()
        
        return analysis
    
    def calculate_rolling_metrics(self, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling financial metrics
        
        Args:
            window: Rolling window size (default 252 trading days = 1 year)
            
        Returns:
            DataFrame with rolling metrics
        """
        print(f"📊 Calculating rolling metrics with {window}-day window...")
        
        rolling_data = pd.DataFrame(index=self.data.index)
        
        # Rolling returns
        rolling_data['Rolling_Return'] = self.returns.rolling(window=window).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Rolling volatility
        rolling_data['Rolling_Volatility'] = self.returns.rolling(window=window).std() * np.sqrt(252)
        
        # Rolling Sharpe ratio
        rolling_data['Rolling_Sharpe'] = self.returns.rolling(window=window).apply(
            lambda x: (x.mean() * 252 - self.risk_free_rate) / (x.std() * np.sqrt(252))
        )
        
        # Rolling max drawdown
        rolling_data['Rolling_MaxDrawdown'] = self.returns.rolling(window=window).apply(
            lambda x: ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min()
        )
        
        return rolling_data
    
    def plot_financial_analysis(self, figsize: Tuple[int, int] = (20, 16)):
        """
        Create comprehensive financial analysis visualization
        
        Args:
            figsize: Figure size tuple
        """
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle('Comprehensive Financial Analysis', fontsize=16, fontweight='bold')
        
        # Price chart
        ax1 = axes[0, 0]
        ax1.plot(self.data.index, self.data[self.close_col], linewidth=1, color='blue')
        ax1.set_title('Price Chart')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative returns
        ax2 = axes[0, 1]
        cumulative_returns = (1 + self.returns).cumprod()
        ax2.plot(cumulative_returns.index, cumulative_returns, linewidth=1, color='green')
        ax2.set_title('Cumulative Returns')
        ax2.set_ylabel('Cumulative Return')
        ax2.grid(True, alpha=0.3)
        
        # Returns distribution
        ax3 = axes[0, 2]
        ax3.hist(self.returns, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(self.returns.mean(), color='red', linestyle='--', label=f'Mean: {self.returns.mean():.4f}')
        ax3.set_title('Returns Distribution')
        ax3.set_xlabel('Daily Returns')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Drawdown chart
        ax4 = axes[1, 0]
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        ax4.fill_between(drawdown.index, drawdown, 0, alpha=0.7, color='red')
        ax4.set_title('Drawdown Analysis')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        # Rolling volatility
        ax5 = axes[1, 1]
        rolling_vol = self.returns.rolling(window=60).std() * np.sqrt(252)
        ax5.plot(rolling_vol.index, rolling_vol, linewidth=1, color='orange')
        ax5.set_title('Rolling 60-Day Volatility')
        ax5.set_ylabel('Annualized Volatility')
        ax5.grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        ax6 = axes[1, 2]
        rolling_sharpe = self.returns.rolling(window=60).apply(
            lambda x: (x.mean() * 252 - self.risk_free_rate) / (x.std() * np.sqrt(252))
        )
        ax6.plot(rolling_sharpe.index, rolling_sharpe, linewidth=1, color='brown')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.set_title('Rolling 60-Day Sharpe Ratio')
        ax6.set_ylabel('Sharpe Ratio')
        ax6.grid(True, alpha=0.3)
        
        # Returns vs risk scatter
        ax7 = axes[2, 0]
        monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_vol = self.returns.resample('M').std()
        ax7.scatter(monthly_vol, monthly_returns, alpha=0.6, color='teal')
        ax7.set_title('Risk vs Return (Monthly)')
        ax7.set_xlabel('Monthly Volatility')
        ax7.set_ylabel('Monthly Return')
        ax7.grid(True, alpha=0.3)
        
        # QQ plot
        ax8 = axes[2, 1]
        from scipy import stats
        stats.probplot(self.returns, dist="norm", plot=ax8)
        ax8.set_title('Q-Q Plot (Normal Distribution)')
        ax8.grid(True, alpha=0.3)
        
        # Volume analysis
        ax9 = axes[2, 2]
        if self.volume_col in self.data.columns:
            ax9.bar(self.data.index, self.data[self.volume_col], alpha=0.6, color='gray')
            ax9.set_title('Volume Analysis')
            ax9.set_ylabel('Volume')
        else:
            # Performance metrics summary
            metrics = self.calculate_basic_metrics()
            metric_names = list(metrics.keys())[:8]  # Show top 8 metrics
            metric_values = [metrics[name] for name in metric_names]
            
            y_pos = np.arange(len(metric_names))
            ax9.barh(y_pos, metric_values)
            ax9.set_yticks(y_pos)
            ax9.set_yticklabels(metric_names)
            ax9.set_title('Key Performance Metrics')
        
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def get_comprehensive_financial_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive financial analysis
        
        Returns:
            Dictionary containing all financial analysis
        """
        print("💰 Performing comprehensive financial analysis...")
        
        analysis = {}
        
        try:
            # Performance metrics
            analysis['performance_metrics'] = self.calculate_performance_metrics()
            
            # Distribution analysis
            analysis['distribution_analysis'] = self.analyze_returns_distribution()
            
            # Rolling metrics
            analysis['rolling_metrics'] = self.calculate_rolling_metrics()
            
            # Risk metrics
            analysis['risk_metrics'] = self.calculate_var_cvar()
            
            print("✅ Financial analysis completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in financial analysis: {str(e)}")
            
        return analysis 