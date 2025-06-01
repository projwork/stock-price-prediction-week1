"""
Data Loader for Financial Data
Handles loading multiple stock data files and portfolio analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import os
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

class FinancialDataLoader:
    """
    Comprehensive Data Loader for Financial Analysis
    """
    
    def __init__(self, data_directory: str = "data/yfinance_data"):
        """
        Initialize Financial Data Loader
        
        Args:
            data_directory: Directory containing the stock data files
        """
        self.data_directory = data_directory
        self.stock_data = {}
        self.tickers = []
        self.portfolio_data = None
        
    def load_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """
        Load all stock data files from the directory
        
        Returns:
            Dictionary with ticker symbols as keys and DataFrames as values
        """
        print(f"📂 Loading stock data from {self.data_directory}...")
        
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(self.data_directory, "*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {self.data_directory}")
            return {}
        
        # Load each file
        for file_path in csv_files:
            try:
                # Extract ticker symbol from filename
                filename = os.path.basename(file_path)
                ticker = filename.replace('_historical_data.csv', '').replace('.csv', '')
                
                # Load the data
                df = pd.read_csv(file_path)
                
                # Basic data preparation
                df = self.prepare_stock_data(df, ticker)
                
                if not df.empty:
                    self.stock_data[ticker] = df
                    self.tickers.append(ticker)
                    print(f"✅ Loaded {ticker}: {len(df)} records")
                else:
                    print(f"⚠️ Warning: Empty data for {ticker}")
                    
            except Exception as e:
                print(f"❌ Error loading {filename}: {str(e)}")
        
        print(f"📊 Successfully loaded {len(self.stock_data)} stocks: {', '.join(self.tickers)}")
        return self.stock_data
    
    def prepare_stock_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Prepare individual stock data
        
        Args:
            df: Raw stock DataFrame
            ticker: Stock ticker symbol
            
        Returns:
            Cleaned DataFrame
        """
        try:
            # Make a copy
            data = df.copy()
            
            # Convert Date column to datetime
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
            
            # Sort by date
            data.sort_index(inplace=True)
            
            # Remove any rows with NaN values in essential columns
            essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in essential_cols if col in data.columns]
            data.dropna(subset=available_cols, inplace=True)
            
            # Add ticker column
            data['Ticker'] = ticker
            
            # Calculate basic derived metrics
            if 'Close' in data.columns:
                data['Returns'] = data['Close'].pct_change()
                data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
                
                # Price change metrics
                data['Price_Change'] = data['Close'].diff()
                data['Price_Change_Pct'] = data['Returns'] * 100
                
                # Volatility (rolling 30-day)
                data['Volatility_30d'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)
                
            # Volume metrics
            if 'Volume' in data.columns:
                data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
                data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']
            
            return data
            
        except Exception as e:
            print(f"❌ Error preparing data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        Get data for a specific stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with stock data
        """
        if ticker in self.stock_data:
            return self.stock_data[ticker]
        else:
            print(f"❌ Ticker {ticker} not found. Available tickers: {', '.join(self.tickers)}")
            return pd.DataFrame()
    
    def create_portfolio_data(self, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Create portfolio-level data from individual stocks
        
        Args:
            weights: Dictionary of ticker weights (default: equal weights)
            
        Returns:
            DataFrame with portfolio data
        """
        if not self.stock_data:
            print("❌ No stock data loaded")
            return pd.DataFrame()
        
        print("📊 Creating portfolio data...")
        
        # Default to equal weights if not provided
        if weights is None:
            weights = {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}
        
        # Validate weights
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.001:
            print(f"⚠️ Warning: Weights sum to {total_weight:.3f}, normalizing to 1.0")
            weights = {ticker: weight / total_weight for ticker, weight in weights.items()}
        
        # Get aligned price data
        price_data = {}
        for ticker in self.tickers:
            if ticker in weights and weights[ticker] > 0:
                price_data[ticker] = self.stock_data[ticker]['Close']
        
        # Create aligned DataFrame
        portfolio_prices = pd.DataFrame(price_data)
        portfolio_prices.dropna(inplace=True)
        
        # Calculate portfolio metrics
        portfolio_data = pd.DataFrame(index=portfolio_prices.index)
        
        # Portfolio value (normalized to start at 100)
        returns_data = portfolio_prices.pct_change().dropna()
        weighted_returns = (returns_data * pd.Series(weights)).sum(axis=1)
        portfolio_data['Portfolio_Value'] = (1 + weighted_returns).cumprod() * 100
        portfolio_data['Portfolio_Returns'] = weighted_returns
        
        # Portfolio metrics
        portfolio_data['Portfolio_Volatility'] = weighted_returns.rolling(window=30).std() * np.sqrt(252)
        portfolio_data['Cumulative_Return'] = (portfolio_data['Portfolio_Value'] / 100) - 1
        
        # Individual stock contributions
        for ticker, weight in weights.items():
            if ticker in returns_data.columns:
                portfolio_data[f'{ticker}_Contribution'] = returns_data[ticker] * weight
        
        self.portfolio_data = portfolio_data
        print(f"✅ Portfolio created with {len(portfolio_data)} records")
        
        return portfolio_data
    
    def get_price_matrix(self, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get price matrix for all stocks
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            DataFrame with prices for all stocks
        """
        if not self.stock_data:
            return pd.DataFrame()
        
        price_data = {}
        for ticker in self.tickers:
            price_data[ticker] = self.stock_data[ticker]['Close']
        
        price_matrix = pd.DataFrame(price_data)
        
        # Apply date filters if provided
        if start_date:
            price_matrix = price_matrix[price_matrix.index >= start_date]
        if end_date:
            price_matrix = price_matrix[price_matrix.index <= end_date]
        
        return price_matrix.dropna()
    
    def get_returns_matrix(self, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get returns matrix for all stocks
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            DataFrame with returns for all stocks
        """
        if not self.stock_data:
            return pd.DataFrame()
        
        returns_data = {}
        for ticker in self.tickers:
            returns_data[ticker] = self.stock_data[ticker]['Returns']
        
        returns_matrix = pd.DataFrame(returns_data)
        
        # Apply date filters if provided
        if start_date:
            returns_matrix = returns_matrix[returns_matrix.index >= start_date]
        if end_date:
            returns_matrix = returns_matrix[returns_matrix.index <= end_date]
        
        return returns_matrix.dropna()
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between stocks
        
        Returns:
            Correlation matrix DataFrame
        """
        returns_matrix = self.get_returns_matrix()
        if returns_matrix.empty:
            return pd.DataFrame()
        
        correlation_matrix = returns_matrix.corr()
        return correlation_matrix
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all stocks
        
        Returns:
            DataFrame with summary statistics
        """
        if not self.stock_data:
            return pd.DataFrame()
        
        summary_stats = []
        
        for ticker in self.tickers:
            data = self.stock_data[ticker]
            
            stats = {
                'Ticker': ticker,
                'Start_Date': data.index.min(),
                'End_Date': data.index.max(),
                'Total_Days': len(data),
                'Current_Price': data['Close'].iloc[-1] if 'Close' in data.columns else np.nan,
                'Max_Price': data['Close'].max() if 'Close' in data.columns else np.nan,
                'Min_Price': data['Close'].min() if 'Close' in data.columns else np.nan,
                'Total_Return': ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) if 'Close' in data.columns else np.nan,
                'Avg_Daily_Return': data['Returns'].mean() if 'Returns' in data.columns else np.nan,
                'Daily_Volatility': data['Returns'].std() if 'Returns' in data.columns else np.nan,
                'Annualized_Volatility': data['Returns'].std() * np.sqrt(252) if 'Returns' in data.columns else np.nan,
                'Max_Drawdown': self.calculate_max_drawdown(data['Returns']) if 'Returns' in data.columns else np.nan,
                'Avg_Volume': data['Volume'].mean() if 'Volume' in data.columns else np.nan,
                'Sharpe_Ratio': self.calculate_sharpe_ratio(data['Returns']) if 'Returns' in data.columns else np.nan
            }
            
            summary_stats.append(stats)
        
        return pd.DataFrame(summary_stats)
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for a returns series"""
        try:
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.cummax()
            drawdown = (cumulative - rolling_max) / rolling_max
            return drawdown.min()
        except:
            return np.nan
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for a returns series"""
        try:
            excess_return = returns.mean() * 252 - risk_free_rate
            volatility = returns.std() * np.sqrt(252)
            if volatility == 0:
                return 0
            return excess_return / volatility
        except:
            return np.nan
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        Generate data quality report
        
        Returns:
            Dictionary with data quality metrics
        """
        if not self.stock_data:
            return {}
        
        report = {
            'total_stocks': len(self.stock_data),
            'tickers': self.tickers,
            'date_range': {},
            'data_completeness': {},
            'missing_data': {}
        }
        
        # Overall date range
        all_dates = []
        for ticker in self.tickers:
            dates = self.stock_data[ticker].index
            all_dates.extend(dates)
        
        if all_dates:
            report['date_range']['earliest'] = min(all_dates)
            report['date_range']['latest'] = max(all_dates)
            report['date_range']['total_days'] = (max(all_dates) - min(all_dates)).days
        
        # Data completeness by ticker
        for ticker in self.tickers:
            data = self.stock_data[ticker]
            
            completeness = {}
            completeness['records'] = len(data)
            completeness['date_range'] = (data.index.min(), data.index.max())
            completeness['missing_values'] = data.isnull().sum().to_dict()
            
            report['data_completeness'][ticker] = completeness
        
        return report
    
    def export_combined_data(self, output_path: str = "data/combined_stock_data.csv"):
        """
        Export combined data for all stocks
        
        Args:
            output_path: Path to save the combined data
        """
        if not self.stock_data:
            print("❌ No data to export")
            return
        
        print(f"💾 Exporting combined data to {output_path}...")
        
        combined_data = []
        
        for ticker in self.tickers:
            data = self.stock_data[ticker].copy()
            data['Ticker'] = ticker
            data.reset_index(inplace=True)
            combined_data.append(data)
        
        if combined_data:
            final_data = pd.concat(combined_data, ignore_index=True)
            final_data.to_csv(output_path, index=False)
            print(f"✅ Exported {len(final_data)} records to {output_path}")
        else:
            print("❌ No data to combine")
    
    def get_sector_allocation(self, sector_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Get sector allocation (requires sector mapping)
        
        Args:
            sector_mapping: Dictionary mapping tickers to sectors
            
        Returns:
            DataFrame with sector allocation
        """
        if sector_mapping is None:
            # Default sector mapping for common tech stocks
            sector_mapping = {
                'AAPL': 'Technology',
                'MSFT': 'Technology', 
                'GOOG': 'Technology',
                'AMZN': 'Consumer Discretionary',
                'TSLA': 'Automotive',
                'META': 'Technology',
                'NVDA': 'Technology'
            }
        
        sector_data = []
        
        for ticker in self.tickers:
            if ticker in sector_mapping:
                data = self.stock_data[ticker]
                current_price = data['Close'].iloc[-1] if 'Close' in data.columns else 0
                
                sector_data.append({
                    'Ticker': ticker,
                    'Sector': sector_mapping[ticker],
                    'Current_Price': current_price,
                    'Market_Value': current_price  # Assuming 1 share for simplicity
                })
        
        if sector_data:
            df = pd.DataFrame(sector_data)
            sector_summary = df.groupby('Sector').agg({
                'Market_Value': 'sum',
                'Ticker': 'count'
            }).rename(columns={'Ticker': 'Stock_Count'})
            
            sector_summary['Allocation_Pct'] = sector_summary['Market_Value'] / sector_summary['Market_Value'].sum() * 100
            
            return sector_summary
        
        return pd.DataFrame() 