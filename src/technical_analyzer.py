"""
Technical Analysis Module for Financial Data
Utilizes TA-Lib for comprehensive technical indicator calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Handle TA-Lib import gracefully
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Using simplified calculations.")

class TechnicalAnalyzer:
    """
    Comprehensive Technical Analysis using TA-Lib and custom calculations
    """
    
    def __init__(self, data: pd.DataFrame, 
                 date_column: str = 'Date',
                 open_col: str = 'Open',
                 high_col: str = 'High', 
                 low_col: str = 'Low',
                 close_col: str = 'Close',
                 volume_col: str = 'Volume'):
        """
        Initialize Technical Analyzer
        
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
        
        # Prepare data
        self.prepare_data()
        
        # Extract price arrays for TA-Lib
        self.open_prices = self.data[self.open_col].values
        self.high_prices = self.data[self.high_col].values
        self.low_prices = self.data[self.low_col].values
        self.close_prices = self.data[self.close_col].values
        self.volumes = self.data[self.volume_col].values
        
    def prepare_data(self):
        """Prepare and clean the data"""
        # Convert date column to datetime
        if self.date_column in self.data.columns:
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
            self.data.set_index(self.date_column, inplace=True)
        
        # Sort by date
        self.data.sort_index(inplace=True)
        
        # Remove any rows with NaN values in OHLCV columns
        required_cols = [self.open_col, self.high_col, self.low_col, self.close_col, self.volume_col]
        self.data.dropna(subset=required_cols, inplace=True)
        
        print(f"Data prepared: {len(self.data)} records from {self.data.index.min()} to {self.data.index.max()}")
    
    def calculate_moving_averages(self, periods: List[int] = [5, 10, 20, 50, 100, 200]) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages for specified periods
        
        Args:
            periods: List of periods for moving averages
            
        Returns:
            DataFrame with moving averages
        """
        ma_data = self.data.copy()
        
        for period in periods:
            if TALIB_AVAILABLE:
                ma_data[f'SMA_{period}'] = talib.SMA(self.close_prices, timeperiod=period)
            else:
                # Fallback calculation
                ma_data[f'SMA_{period}'] = self.data[self.close_col].rolling(window=period).mean()
        
        return ma_data
    
    def calculate_exponential_moving_averages(self, periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages
        
        Args:
            periods: List of periods for EMAs
            
        Returns:
            DataFrame with EMAs
        """
        ema_data = self.data.copy()
        
        for period in periods:
            if TALIB_AVAILABLE:
                ema_data[f'EMA_{period}'] = talib.EMA(self.close_prices, timeperiod=period)
            else:
                # Fallback calculation
                ema_data[f'EMA_{period}'] = self.data[self.close_col].ewm(span=period).mean()
                
        return ema_data
    
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            period: Period for RSI calculation
            
        Returns:
            Series with RSI values
        """
        if TALIB_AVAILABLE:
            return pd.Series(talib.RSI(self.close_prices, timeperiod=period), 
                           index=self.data.index, name=f'RSI_{period}')
        else:
            # Fallback RSI calculation
            delta = self.data[self.close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.rename(f'RSI_{period}')
    
    def calculate_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            DataFrame with MACD, signal, and histogram
        """
        if TALIB_AVAILABLE:
            macd, macd_signal, macd_hist = talib.MACD(self.close_prices, 
                                                     fastperiod=fast_period,
                                                     slowperiod=slow_period, 
                                                     signalperiod=signal_period)
            return pd.DataFrame({
                'MACD': macd,
                'MACD_Signal': macd_signal,
                'MACD_Histogram': macd_hist
            }, index=self.data.index)
        else:
            # Fallback MACD calculation
            ema_fast = self.data[self.close_col].ewm(span=fast_period).mean()
            ema_slow = self.data[self.close_col].ewm(span=slow_period).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal_period).mean()
            macd_hist = macd - macd_signal
            
            return pd.DataFrame({
                'MACD': macd,
                'MACD_Signal': macd_signal,
                'MACD_Histogram': macd_hist
            })
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            period: Period for moving average
            std_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with Bollinger Bands
        """
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(self.close_prices, 
                                              timeperiod=period,
                                              nbdevup=std_dev,
                                              nbdevdn=std_dev)
            return pd.DataFrame({
                'BB_Upper': upper,
                'BB_Middle': middle,
                'BB_Lower': lower
            }, index=self.data.index)
        else:
            # Fallback calculation
            sma = self.data[self.close_col].rolling(window=period).mean()
            std = self.data[self.close_col].rolling(window=period).std()
            
            return pd.DataFrame({
                'BB_Upper': sma + (std * std_dev),
                'BB_Middle': sma,
                'BB_Lower': sma - (std * std_dev)
            })
    
    def calculate_stochastic_oscillator(self, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator
        
        Args:
            k_period: Period for %K calculation
            d_period: Period for %D smoothing
            
        Returns:
            DataFrame with %K and %D
        """
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(self.high_prices, self.low_prices, self.close_prices,
                                     fastk_period=k_period, slowk_period=3, slowd_period=d_period)
            return pd.DataFrame({
                'Stoch_K': slowk,
                'Stoch_D': slowd
            }, index=self.data.index)
        else:
            # Fallback calculation
            lowest_low = self.data[self.low_col].rolling(window=k_period).min()
            highest_high = self.data[self.high_col].rolling(window=k_period).max()
            k_percent = 100 * ((self.data[self.close_col] - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return pd.DataFrame({
                'Stoch_K': k_percent,
                'Stoch_D': d_percent
            })
    
    def calculate_atr(self, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            period: Period for ATR calculation
            
        Returns:
            Series with ATR values
        """
        if TALIB_AVAILABLE:
            atr = talib.ATR(self.high_prices, self.low_prices, self.close_prices, timeperiod=period)
            return pd.Series(atr, index=self.data.index, name=f'ATR_{period}')
        else:
            # Fallback calculation
            high_low = self.data[self.high_col] - self.data[self.low_col]
            high_close = np.abs(self.data[self.high_col] - self.data[self.close_col].shift())
            low_close = np.abs(self.data[self.low_col] - self.data[self.close_col].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr.rename(f'ATR_{period}')
    
    def calculate_volume_indicators(self) -> pd.DataFrame:
        """
        Calculate volume-based indicators
        
        Returns:
            DataFrame with volume indicators
        """
        volume_data = pd.DataFrame(index=self.data.index)
        
        # Volume Moving Averages
        volume_data['Volume_SMA_10'] = self.data[self.volume_col].rolling(window=10).mean()
        volume_data['Volume_SMA_20'] = self.data[self.volume_col].rolling(window=20).mean()
        
        # Volume Ratio
        volume_data['Volume_Ratio'] = self.data[self.volume_col] / volume_data['Volume_SMA_20']
        
        if TALIB_AVAILABLE:
            # On-Balance Volume
            volume_data['OBV'] = talib.OBV(self.close_prices, self.volumes)
            
            # Accumulation/Distribution Line
            volume_data['AD'] = talib.AD(self.high_prices, self.low_prices, self.close_prices, self.volumes)
        else:
            # Fallback OBV calculation
            obv = [0]
            for i in range(1, len(self.data)):
                if self.close_prices[i] > self.close_prices[i-1]:
                    obv.append(obv[-1] + self.volumes[i])
                elif self.close_prices[i] < self.close_prices[i-1]:
                    obv.append(obv[-1] - self.volumes[i])
                else:
                    obv.append(obv[-1])
            volume_data['OBV'] = obv
        
        return volume_data
    
    def get_comprehensive_technical_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive technical analysis with all indicators
        
        Returns:
            Dictionary containing all technical indicators
        """
        print("🔧 Calculating comprehensive technical indicators...")
        
        # Prepare comprehensive analysis
        analysis = {}
        
        try:
            # Moving Averages
            print("📈 Calculating Moving Averages...")
            analysis['moving_averages'] = self.calculate_moving_averages()
            analysis['ema'] = self.calculate_exponential_moving_averages()
            
            # Momentum Indicators
            print("📊 Calculating Momentum Indicators...")
            analysis['rsi'] = self.calculate_rsi()
            analysis['macd'] = self.calculate_macd()
            analysis['stochastic'] = self.calculate_stochastic_oscillator()
            
            # Volatility Indicators
            print("📉 Calculating Volatility Indicators...")
            analysis['bollinger_bands'] = self.calculate_bollinger_bands()
            analysis['atr'] = self.calculate_atr()
            
            # Volume Indicators
            print("📊 Calculating Volume Indicators...")
            analysis['volume'] = self.calculate_volume_indicators()
            
            # Support and Resistance Levels
            print("🎯 Calculating Support and Resistance...")
            analysis['support_resistance'] = self.calculate_support_resistance()
            
            print("✅ Technical analysis completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in technical analysis: {str(e)}")
            
        return analysis
    
    def calculate_support_resistance(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate support and resistance levels
        
        Args:
            window: Window for calculating levels
            
        Returns:
            DataFrame with support and resistance levels
        """
        sr_data = pd.DataFrame(index=self.data.index)
        
        # Rolling support and resistance
        sr_data['Support'] = self.data[self.low_col].rolling(window=window).min()
        sr_data['Resistance'] = self.data[self.high_col].rolling(window=window).max()
        
        # Pivot points
        sr_data['Pivot'] = (self.data[self.high_col] + self.data[self.low_col] + self.data[self.close_col]) / 3
        sr_data['R1'] = 2 * sr_data['Pivot'] - self.data[self.low_col]
        sr_data['S1'] = 2 * sr_data['Pivot'] - self.data[self.high_col]
        sr_data['R2'] = sr_data['Pivot'] + (self.data[self.high_col] - self.data[self.low_col])
        sr_data['S2'] = sr_data['Pivot'] - (self.data[self.high_col] - self.data[self.low_col])
        
        return sr_data
    
    def generate_trading_signals(self) -> pd.DataFrame:
        """
        Generate basic trading signals based on technical indicators
        
        Returns:
            DataFrame with trading signals
        """
        signals = pd.DataFrame(index=self.data.index)
        
        # Get indicators
        sma_20 = self.data[self.close_col].rolling(window=20).mean()
        sma_50 = self.data[self.close_col].rolling(window=50).mean()
        rsi = self.calculate_rsi()
        
        # Moving Average Crossover Signal
        signals['MA_Signal'] = np.where(sma_20 > sma_50, 1, -1)
        
        # RSI Overbought/Oversold Signal
        signals['RSI_Signal'] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
        
        # Combined Signal
        signals['Combined_Signal'] = (signals['MA_Signal'] + signals['RSI_Signal']) / 2
        
        return signals
    
    def plot_technical_analysis(self, figsize: Tuple[int, int] = (20, 16)):
        """
        Create comprehensive technical analysis visualization
        
        Args:
            figsize: Figure size tuple
        """
        # Get technical indicators
        analysis = self.get_comprehensive_technical_analysis()
        
        fig, axes = plt.subplots(4, 2, figsize=figsize)
        fig.suptitle('Comprehensive Technical Analysis', fontsize=16, fontweight='bold')
        
        # Price and Moving Averages
        ax1 = axes[0, 0]
        ax1.plot(self.data.index, self.data[self.close_col], label='Close Price', linewidth=1)
        
        # Add moving averages if available
        if 'moving_averages' in analysis:
            ma_data = analysis['moving_averages']
            for col in ma_data.columns:
                if 'SMA' in col:
                    ax1.plot(ma_data.index, ma_data[col], label=col, alpha=0.7)
        
        ax1.set_title('Price with Moving Averages')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI
        ax2 = axes[0, 1]
        if 'rsi' in analysis:
            rsi_data = analysis['rsi']
            ax2.plot(rsi_data.index, rsi_data, label='RSI', color='purple')
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
            ax2.set_ylim(0, 100)
        ax2.set_title('RSI (Relative Strength Index)')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # MACD
        ax3 = axes[1, 0]
        if 'macd' in analysis:
            macd_data = analysis['macd']
            ax3.plot(macd_data.index, macd_data['MACD'], label='MACD', color='blue')
            ax3.plot(macd_data.index, macd_data['MACD_Signal'], label='Signal', color='red')
            ax3.bar(macd_data.index, macd_data['MACD_Histogram'], label='Histogram', alpha=0.3)
        ax3.set_title('MACD')
        ax3.set_ylabel('MACD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Bollinger Bands
        ax4 = axes[1, 1]
        ax4.plot(self.data.index, self.data[self.close_col], label='Close Price', color='black')
        if 'bollinger_bands' in analysis:
            bb_data = analysis['bollinger_bands']
            ax4.plot(bb_data.index, bb_data['BB_Upper'], label='Upper Band', color='red', alpha=0.7)
            ax4.plot(bb_data.index, bb_data['BB_Middle'], label='Middle Band', color='blue', alpha=0.7)
            ax4.plot(bb_data.index, bb_data['BB_Lower'], label='Lower Band', color='green', alpha=0.7)
            ax4.fill_between(bb_data.index, bb_data['BB_Upper'], bb_data['BB_Lower'], alpha=0.1)
        ax4.set_title('Bollinger Bands')
        ax4.set_ylabel('Price ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Volume
        ax5 = axes[2, 0]
        ax5.bar(self.data.index, self.data[self.volume_col], alpha=0.6, color='gray', label='Volume')
        if 'volume' in analysis:
            vol_data = analysis['volume']
            if 'Volume_SMA_20' in vol_data.columns:
                ax5.plot(vol_data.index, vol_data['Volume_SMA_20'], label='Volume SMA 20', color='red')
        ax5.set_title('Volume Analysis')
        ax5.set_ylabel('Volume')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Stochastic Oscillator
        ax6 = axes[2, 1]
        if 'stochastic' in analysis:
            stoch_data = analysis['stochastic']
            ax6.plot(stoch_data.index, stoch_data['Stoch_K'], label='%K', color='blue')
            ax6.plot(stoch_data.index, stoch_data['Stoch_D'], label='%D', color='red')
            ax6.axhline(y=80, color='r', linestyle='--', alpha=0.7)
            ax6.axhline(y=20, color='g', linestyle='--', alpha=0.7)
            ax6.set_ylim(0, 100)
        ax6.set_title('Stochastic Oscillator')
        ax6.set_ylabel('Stochastic %')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # ATR
        ax7 = axes[3, 0]
        if 'atr' in analysis:
            atr_data = analysis['atr']
            ax7.plot(atr_data.index, atr_data, label='ATR', color='orange')
        ax7.set_title('Average True Range (ATR)')
        ax7.set_ylabel('ATR')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Trading Signals
        ax8 = axes[3, 1]
        signals = self.generate_trading_signals()
        ax8.plot(signals.index, signals['Combined_Signal'], label='Combined Signal', color='purple')
        ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax8.set_title('Trading Signals')
        ax8.set_ylabel('Signal Strength')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig 