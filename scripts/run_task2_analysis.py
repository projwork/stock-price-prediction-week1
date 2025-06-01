"""
Task 2: Quantitative Analysis using PyNance and TA-Lib
Comprehensive automated execution script for technical and financial analysis
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import json
from datetime import datetime
from pathlib import Path

# Import our modules
from data_loader import FinancialDataLoader
from technical_analyzer import TechnicalAnalyzer
from financial_metrics_analyzer import FinancialMetricsAnalyzer

warnings.filterwarnings('ignore')

def create_output_directory():
    """Create output directory for results"""
    output_dir = Path("data/results/task2_quantitative_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def analyze_individual_stock(ticker: str, data: pd.DataFrame, output_dir: Path):
    """
    Perform comprehensive analysis on individual stock
    
    Args:
        ticker: Stock ticker symbol
        data: Stock price data
        output_dir: Output directory for results
    """
    print(f"\n🔍 Analyzing {ticker}...")
    
    # Create stock-specific output directory
    stock_dir = output_dir / ticker
    stock_dir.mkdir(exist_ok=True)
    
    # Technical Analysis
    print(f"📈 Performing technical analysis for {ticker}...")
    tech_analyzer = TechnicalAnalyzer(data.copy())
    technical_analysis = tech_analyzer.get_comprehensive_technical_analysis()
    
    # Create technical analysis plot
    try:
        plt.style.use('default')
        fig = tech_analyzer.plot_technical_analysis(figsize=(24, 18))
        plt.savefig(stock_dir / f"{ticker}_technical_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Technical analysis plot saved for {ticker}")
    except Exception as e:
        print(f"⚠️ Error creating technical plot for {ticker}: {str(e)}")
    
    # Financial Metrics Analysis
    print(f"💰 Performing financial metrics analysis for {ticker}...")
    finance_analyzer = FinancialMetricsAnalyzer(data.copy())
    financial_analysis = finance_analyzer.get_comprehensive_financial_analysis()
    
    # Create financial analysis plot
    try:
        fig = finance_analyzer.plot_financial_analysis(figsize=(24, 18))
        plt.savefig(stock_dir / f"{ticker}_financial_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Financial analysis plot saved for {ticker}")
    except Exception as e:
        print(f"⚠️ Error creating financial plot for {ticker}: {str(e)}")
    
    # Combine analysis results
    stock_analysis = {
        'ticker': ticker,
        'analysis_date': datetime.now().isoformat(),
        'data_period': {
            'start': data.index.min().isoformat(),
            'end': data.index.max().isoformat(),
            'total_days': len(data)
        },
        'technical_analysis': {},
        'financial_analysis': financial_analysis
    }
    
    # Extract key technical indicators for JSON export
    try:
        if 'moving_averages' in technical_analysis:
            ma_data = technical_analysis['moving_averages']
            stock_analysis['technical_analysis']['latest_prices'] = {
                'close': float(ma_data['Close'].iloc[-1]),
                'sma_20': float(ma_data['SMA_20'].iloc[-1]) if 'SMA_20' in ma_data.columns else None,
                'sma_50': float(ma_data['SMA_50'].iloc[-1]) if 'SMA_50' in ma_data.columns else None,
                'sma_200': float(ma_data['SMA_200'].iloc[-1]) if 'SMA_200' in ma_data.columns else None
            }
        
        if 'rsi' in technical_analysis:
            rsi_data = technical_analysis['rsi']
            stock_analysis['technical_analysis']['rsi_current'] = float(rsi_data.iloc[-1])
        
        if 'macd' in technical_analysis:
            macd_data = technical_analysis['macd']
            stock_analysis['technical_analysis']['macd_current'] = {
                'macd': float(macd_data['MACD'].iloc[-1]),
                'signal': float(macd_data['MACD_Signal'].iloc[-1]),
                'histogram': float(macd_data['MACD_Histogram'].iloc[-1])
            }
        
    except Exception as e:
        print(f"⚠️ Error extracting technical indicators for {ticker}: {str(e)}")
    
    # Save analysis results
    try:
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        stock_analysis_clean = convert_numpy_types(stock_analysis)
        
        with open(stock_dir / f"{ticker}_analysis_summary.json", 'w') as f:
            json.dump(stock_analysis_clean, f, indent=2, default=str)
        print(f"✅ Analysis summary saved for {ticker}")
        
    except Exception as e:
        print(f"⚠️ Error saving analysis summary for {ticker}: {str(e)}")
    
    return stock_analysis

def create_portfolio_analysis(data_loader: FinancialDataLoader, output_dir: Path):
    """
    Create portfolio-level analysis
    
    Args:
        data_loader: Data loader with all stock data
        output_dir: Output directory for results
    """
    print("\n📊 Creating portfolio analysis...")
    
    # Create portfolio with equal weights
    portfolio_data = data_loader.create_portfolio_data()
    
    if portfolio_data.empty:
        print("❌ Unable to create portfolio data")
        return
    
    # Portfolio analysis
    portfolio_analyzer = FinancialMetricsAnalyzer(
        portfolio_data[['Portfolio_Value']].rename(columns={'Portfolio_Value': 'Close'}),
        close_col='Close'
    )
    
    portfolio_analysis = portfolio_analyzer.get_comprehensive_financial_analysis()
    
    # Create portfolio visualization
    try:
        fig = portfolio_analyzer.plot_financial_analysis(figsize=(24, 18))
        plt.suptitle('Portfolio Analysis - Equal Weighted', fontsize=20, fontweight='bold')
        plt.savefig(output_dir / "portfolio_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Portfolio analysis plot saved")
    except Exception as e:
        print(f"⚠️ Error creating portfolio plot: {str(e)}")
    
    # Correlation matrix
    correlation_matrix = data_loader.calculate_correlation_matrix()
    if not correlation_matrix.empty:
        plt.figure(figsize=(12, 10))
        import seaborn as sns
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, linewidths=0.5)
        plt.title('Stock Returns Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Correlation matrix saved")
    
    return portfolio_analysis

def generate_executive_summary(individual_analyses: dict, portfolio_analysis: dict, 
                             summary_stats: pd.DataFrame, output_dir: Path):
    """
    Generate executive summary report
    
    Args:
        individual_analyses: Dictionary of individual stock analyses
        portfolio_analysis: Portfolio analysis results
        summary_stats: Summary statistics DataFrame
        output_dir: Output directory for results
    """
    print("\n📝 Generating executive summary...")
    
    summary = {
        'task': 'Task 2 - Quantitative Analysis using PyNance and TA-Lib',
        'analysis_date': datetime.now().isoformat(),
        'overview': {
            'total_stocks_analyzed': len(individual_analyses),
            'tickers': list(individual_analyses.keys()),
            'analysis_period': {
                'start': summary_stats['Start_Date'].min().isoformat(),
                'end': summary_stats['End_Date'].max().isoformat()
            }
        },
        'key_findings': {},
        'technical_indicators_summary': {},
        'financial_metrics_summary': {},
        'portfolio_analysis': portfolio_analysis,
        'recommendations': []
    }
    
    # Calculate key findings
    try:
        # Best and worst performers
        best_performer = summary_stats.loc[summary_stats['Total_Return'].idxmax()]
        worst_performer = summary_stats.loc[summary_stats['Total_Return'].idxmin()]
        
        summary['key_findings'] = {
            'best_performer': {
                'ticker': best_performer['Ticker'],
                'total_return': float(best_performer['Total_Return']),
                'sharpe_ratio': float(best_performer['Sharpe_Ratio'])
            },
            'worst_performer': {
                'ticker': worst_performer['Ticker'],
                'total_return': float(worst_performer['Total_Return']),
                'sharpe_ratio': float(worst_performer['Sharpe_Ratio'])
            },
            'highest_volatility': {
                'ticker': summary_stats.loc[summary_stats['Annualized_Volatility'].idxmax(), 'Ticker'],
                'volatility': float(summary_stats['Annualized_Volatility'].max())
            },
            'lowest_volatility': {
                'ticker': summary_stats.loc[summary_stats['Annualized_Volatility'].idxmin(), 'Ticker'],
                'volatility': float(summary_stats['Annualized_Volatility'].min())
            }
        }
        
        # Technical indicators summary
        rsi_data = {}
        macd_signals = {}
        
        for ticker, analysis in individual_analyses.items():
            tech_analysis = analysis.get('technical_analysis', {})
            
            # RSI analysis
            rsi_current = tech_analysis.get('rsi_current')
            if rsi_current:
                if rsi_current > 70:
                    rsi_data[ticker] = 'Overbought'
                elif rsi_current < 30:
                    rsi_data[ticker] = 'Oversold'
                else:
                    rsi_data[ticker] = 'Neutral'
            
            # MACD signals
            macd_current = tech_analysis.get('macd_current', {})
            if macd_current:
                macd_val = macd_current.get('macd', 0)
                signal_val = macd_current.get('signal', 0)
                if macd_val > signal_val:
                    macd_signals[ticker] = 'Bullish'
                else:
                    macd_signals[ticker] = 'Bearish'
        
        summary['technical_indicators_summary'] = {
            'rsi_signals': rsi_data,
            'macd_signals': macd_signals
        }
        
        # Generate recommendations
        recommendations = []
        
        # High Sharpe ratio stocks
        high_sharpe_stocks = summary_stats[summary_stats['Sharpe_Ratio'] > summary_stats['Sharpe_Ratio'].quantile(0.75)]
        if not high_sharpe_stocks.empty:
            recommendations.append({
                'type': 'Risk-Adjusted Performance',
                'recommendation': f"Consider overweighting {', '.join(high_sharpe_stocks['Ticker'].tolist())} due to superior risk-adjusted returns",
                'details': 'These stocks show high Sharpe ratios indicating good risk-adjusted performance'
            })
        
        # Low volatility stocks for stability
        low_vol_stocks = summary_stats[summary_stats['Annualized_Volatility'] < summary_stats['Annualized_Volatility'].quantile(0.25)]
        if not low_vol_stocks.empty:
            recommendations.append({
                'type': 'Risk Management',
                'recommendation': f"For conservative portfolios, consider {', '.join(low_vol_stocks['Ticker'].tolist())} for lower volatility",
                'details': 'These stocks show lower volatility suitable for risk-averse investors'
            })
        
        # RSI-based recommendations
        oversold_stocks = [ticker for ticker, signal in rsi_data.items() if signal == 'Oversold']
        if oversold_stocks:
            recommendations.append({
                'type': 'Technical Analysis',
                'recommendation': f"Potential buying opportunities in {', '.join(oversold_stocks)} based on RSI oversold conditions",
                'details': 'RSI below 30 suggests potential oversold conditions'
            })
        
        summary['recommendations'] = recommendations
        
    except Exception as e:
        print(f"⚠️ Error generating key findings: {str(e)}")
    
    # Save executive summary
    try:
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        summary_clean = convert_numpy_types(summary)
        
        with open(output_dir / "executive_summary.json", 'w') as f:
            json.dump(summary_clean, f, indent=2, default=str)
        
        # Also create a readable text summary
        with open(output_dir / "executive_summary.txt", 'w') as f:
            f.write("Task 2: Quantitative Analysis using PyNance and TA-Lib\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Stocks Analyzed: {len(individual_analyses)}\n")
            f.write(f"Tickers: {', '.join(individual_analyses.keys())}\n\n")
            
            f.write("KEY FINDINGS:\n")
            f.write("-" * 30 + "\n")
            if 'key_findings' in summary:
                kf = summary['key_findings']
                f.write(f"Best Performer: {kf['best_performer']['ticker']} ({kf['best_performer']['total_return']:.2%})\n")
                f.write(f"Worst Performer: {kf['worst_performer']['ticker']} ({kf['worst_performer']['total_return']:.2%})\n")
                f.write(f"Highest Volatility: {kf['highest_volatility']['ticker']} ({kf['highest_volatility']['volatility']:.2%})\n")
                f.write(f"Lowest Volatility: {kf['lowest_volatility']['ticker']} ({kf['lowest_volatility']['volatility']:.2%})\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            for i, rec in enumerate(summary.get('recommendations', []), 1):
                f.write(f"{i}. {rec['type']}: {rec['recommendation']}\n")
                f.write(f"   Details: {rec['details']}\n\n")
        
        print("✅ Executive summary saved")
        
    except Exception as e:
        print(f"⚠️ Error saving executive summary: {str(e)}")

def main():
    """Main execution function for Task 2"""
    print("🚀 Starting Task 2: Quantitative Analysis using PyNance and TA-Lib")
    print("=" * 80)
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"📁 Output directory: {output_dir}")
    
    # Load data
    print("\n📂 Loading financial data...")
    data_loader = FinancialDataLoader("data/yfinance_data")
    stock_data = data_loader.load_all_stocks()
    
    if not stock_data:
        print("❌ No stock data loaded. Please check the data directory.")
        return
    
    # Get summary statistics
    summary_stats = data_loader.get_summary_statistics()
    print(f"📊 Loaded {len(summary_stats)} stocks for analysis")
    
    # Save summary statistics
    summary_stats.to_csv(output_dir / "summary_statistics.csv", index=False)
    print("✅ Summary statistics saved")
    
    # Analyze individual stocks
    individual_analyses = {}
    
    for ticker in data_loader.tickers:
        try:
            stock_data_individual = data_loader.get_stock_data(ticker)
            if not stock_data_individual.empty:
                analysis = analyze_individual_stock(ticker, stock_data_individual, output_dir)
                individual_analyses[ticker] = analysis
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {str(e)}")
            continue
    
    # Portfolio analysis
    try:
        portfolio_analysis = create_portfolio_analysis(data_loader, output_dir)
    except Exception as e:
        print(f"❌ Error in portfolio analysis: {str(e)}")
        portfolio_analysis = {}
    
    # Generate executive summary
    try:
        generate_executive_summary(individual_analyses, portfolio_analysis, 
                                 summary_stats, output_dir)
    except Exception as e:
        print(f"❌ Error generating executive summary: {str(e)}")
    
    # Print completion message
    print("\n" + "=" * 80)
    print("✅ Task 2 Analysis Complete!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📊 Analyzed {len(individual_analyses)} stocks")
    print("📈 Technical indicators calculated using TA-Lib")
    print("💰 Financial metrics calculated using PyNance")
    print("📋 Executive summary and visualizations generated")
    print("=" * 80)

if __name__ == "__main__":
    main() 