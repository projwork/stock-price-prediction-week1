"""
Simple Technical Analyzer - No TA-Lib Dependencies
Provides basic technical indicators using only pandas and numpy.
This is a fallback for environments where TA-Lib cannot be installed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class SimpleTechnicalAnalyzer:
    """Simple technical analysis without external TA-Lib dependencies."""
    
    def __init__(self, data: pd.DataFrame, 
                 date_column: str = 'Date',
                 open_col: str = 'Open',
                 high_col: str = 'High', 
                 low_col: str = 'Low',
                 close_col: str = 'Close',
                 volume_col: str = 'Volume'):
        """
        Initialize with stock price data.
        
        Args:
            data: DataFrame with OHLCV data
            date_column: Name of date column
            open_col: Name of open price column
            high_col: Name of high price column
            low_col: Name of low price column
            close_col: Name of close price column
            volume_col: Name of volume column
        """
        self.data = data.copy()
        self.date_column = date_column
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare and validate the data."""
        # Ensure date column is datetime
        if self.date_column in self.data.columns:
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
            self.data = self.data.sort_values(self.date_column)
        
        # Ensure numeric columns
        numeric_cols = [self.open_col, self.high_col, self.low_col, self.close_col]
        if self.volume_col in self.data.columns:
            numeric_cols.append(self.volume_col)
            
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
    
    def simple_moving_average(self, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return self.data[self.close_col].rolling(window=period).mean()
    
    def exponential_moving_average(self, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return self.data[self.close_col].ewm(span=period).mean()
    
    def rsi(self, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            period: Period for RSI calculation
            
        Returns:
            RSI values
        """
        close = self.data[self.close_col]
        delta = close.diff()
        
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            DataFrame with MACD, Signal, and Histogram
        """
        close = self.data[self.close_col]
        
        ema_fast = close.ewm(span=fast_period).mean()
        ema_slow = close.ewm(span=slow_period).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })
    
    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            period: Period for moving average
            std_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with Upper, Middle, and Lower bands
        """
        close = self.data[self.close_col]
        
        middle_band = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return pd.DataFrame({
            'Upper': upper_band,
            'Middle': middle_band,
            'Lower': lower_band
        })
    
    def stochastic_oscillator(self, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            k_period: Period for %K calculation
            d_period: Period for %D smoothing
            
        Returns:
            DataFrame with %K and %D values
        """
        high = self.data[self.high_col]
        low = self.data[self.low_col]
        close = self.data[self.close_col]
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            '%K': k_percent,
            '%D': d_percent
        })
    
    def average_true_range(self, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            period: Period for ATR calculation
            
        Returns:
            ATR values
        """
        high = self.data[self.high_col]
        low = self.data[self.low_col]
        close = self.data[self.close_col]
        
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def price_change_analysis(self) -> pd.DataFrame:
        """Calculate price change metrics."""
        close = self.data[self.close_col]
        
        return pd.DataFrame({
            'Price_Change': close.diff(),
            'Price_Change_Pct': close.pct_change() * 100,
            'Cumulative_Return': (close.pct_change() + 1).cumprod() - 1
        })
    
    def volume_analysis(self) -> pd.DataFrame:
        """Analyze volume patterns if volume data is available."""
        if self.volume_col not in self.data.columns:
            return pd.DataFrame({'Volume_SMA_20': [np.nan] * len(self.data)})
        
        volume = self.data[self.volume_col]
        
        return pd.DataFrame({
            'Volume_SMA_20': volume.rolling(window=20).mean(),
            'Volume_Ratio': volume / volume.rolling(window=20).mean(),
            'Volume_Change_Pct': volume.pct_change() * 100
        })
    
    def get_comprehensive_analysis(self) -> Dict[str, pd.DataFrame]:
        """Get all technical indicators in one comprehensive analysis."""
        
        results = {}
        
        # Moving Averages
        ma_data = pd.DataFrame({
            'SMA_5': self.simple_moving_average(5),
            'SMA_10': self.simple_moving_average(10),
            'SMA_20': self.simple_moving_average(20),
            'SMA_50': self.simple_moving_average(50),
            'EMA_12': self.exponential_moving_average(12),
            'EMA_26': self.exponential_moving_average(26)
        })
        results['Moving_Averages'] = ma_data
        
        # Oscillators
        results['RSI'] = pd.DataFrame({'RSI': self.rsi()})
        results['MACD'] = self.macd()
        results['Stochastic'] = self.stochastic_oscillator()
        
        # Bands and Volatility
        results['Bollinger_Bands'] = self.bollinger_bands()
        results['ATR'] = pd.DataFrame({'ATR': self.average_true_range()})
        
        # Price and Volume Analysis
        results['Price_Analysis'] = self.price_change_analysis()
        results['Volume_Analysis'] = self.volume_analysis()
        
        return results
    
    def generate_simple_signals(self) -> pd.DataFrame:
        """Generate simple trading signals based on technical indicators."""
        signals = pd.DataFrame(index=self.data.index)
        
        # Simple MA crossover signals
        sma_20 = self.simple_moving_average(20)
        sma_50 = self.simple_moving_average(50)
        signals['MA_Signal'] = np.where(sma_20 > sma_50, 1, -1)
        
        # RSI overbought/oversold signals
        rsi_values = self.rsi()
        signals['RSI_Signal'] = np.where(rsi_values < 30, 1, np.where(rsi_values > 70, -1, 0))
        
        # MACD signals
        macd_data = self.macd()
        signals['MACD_Signal'] = np.where(
            macd_data['MACD'] > macd_data['Signal'], 1, -1
        )
        
        # Combined signal (simple average)
        signals['Combined_Signal'] = (
            signals['MA_Signal'] + 
            signals['RSI_Signal'] + 
            signals['MACD_Signal']
        ) / 3
        
        return signals
    
    def plot_technical_analysis(self, figsize: Tuple[int, int] = (15, 12)):
        """Plot comprehensive technical analysis charts."""
        
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Price and Moving Averages
        axes[0].plot(self.data.index, self.data[self.close_col], label='Close', linewidth=1)
        axes[0].plot(self.data.index, self.simple_moving_average(20), label='SMA 20', alpha=0.7)
        axes[0].plot(self.data.index, self.simple_moving_average(50), label='SMA 50', alpha=0.7)
        
        # Bollinger Bands
        bb = self.bollinger_bands()
        axes[0].fill_between(self.data.index, bb['Lower'], bb['Upper'], alpha=0.2, label='Bollinger Bands')
        
        axes[0].set_title('Price and Moving Averages')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        rsi_values = self.rsi()
        axes[1].plot(self.data.index, rsi_values, label='RSI (14)', color='orange')
        axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        axes[1].set_title('RSI')
        axes[1].set_ylabel('RSI')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # MACD
        macd_data = self.macd()
        axes[2].plot(self.data.index, macd_data['MACD'], label='MACD', color='blue')
        axes[2].plot(self.data.index, macd_data['Signal'], label='Signal', color='red')
        axes[2].bar(self.data.index, macd_data['Histogram'], label='Histogram', alpha=0.6, color='gray')
        axes[2].set_title('MACD')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Volume (if available)
        if self.volume_col in self.data.columns:
            axes[3].bar(self.data.index, self.data[self.volume_col], alpha=0.6, color='lightblue')
            volume_ma = self.data[self.volume_col].rolling(window=20).mean()
            axes[3].plot(self.data.index, volume_ma, color='red', label='Volume MA 20')
            axes[3].set_title('Volume')
            axes[3].legend()
        else:
            axes[3].text(0.5, 0.5, 'Volume data not available', 
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('Volume (Not Available)')
        
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the analysis."""
        close = self.data[self.close_col]
        
        return {
            'Total_Return_Pct': ((close.iloc[-1] / close.iloc[0]) - 1) * 100,
            'Volatility': close.pct_change().std() * np.sqrt(252) * 100,  # Annualized
            'Max_Drawdown': ((close / close.expanding().max()) - 1).min() * 100,
            'Current_Price': close.iloc[-1],
            'Price_Range': {
                'Min': close.min(),
                'Max': close.max(),
                'Current_vs_Min_Pct': ((close.iloc[-1] / close.min()) - 1) * 100,
                'Current_vs_Max_Pct': ((close.iloc[-1] / close.max()) - 1) * 100
            },
            'Data_Points': len(self.data),
            'Date_Range': {
                'Start': self.data[self.date_column].min() if self.date_column in self.data.columns else 'N/A',
                'End': self.data[self.date_column].max() if self.date_column in self.data.columns else 'N/A'
            }
        }


def test_simple_technical_analyzer():
    """Test function for the SimpleTechnicalAnalyzer."""
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate realistic OHLCV data
    base_price = 100
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # Random walk with some trend
        if i == 0:
            close = base_price
        else:
            close = prices[-1]['Close'] * (1 + np.random.normal(0, 0.02))
        
        # OHLC based on close
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = low + (high - low) * np.random.random()
        
        prices.append({
            'Date': dates[i],
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close
        })
        
        volumes.append(np.random.randint(100000, 1000000))
    
    # Create DataFrame
    df = pd.DataFrame(prices)
    df['Volume'] = volumes
    
    # Test analyzer
    analyzer = SimpleTechnicalAnalyzer(df)
    
    # Test individual indicators
    print("Testing Simple Technical Analyzer...")
    print(f"RSI (last 5 values): {analyzer.rsi().tail().values}")
    print(f"SMA 20 (last 5 values): {analyzer.simple_moving_average(20).tail().values}")
    
    # Test comprehensive analysis
    analysis = analyzer.get_comprehensive_analysis()
    print(f"Analysis components: {list(analysis.keys())}")
    
    # Test signals
    signals = analyzer.generate_simple_signals()
    print(f"Signal columns: {list(signals.columns)}")
    
    # Test summary
    summary = analyzer.get_summary_statistics()
    print(f"Summary: {summary}")
    
    print("✓ SimpleTechnicalAnalyzer test completed successfully!")


if __name__ == "__main__":
    test_simple_technical_analyzer() 