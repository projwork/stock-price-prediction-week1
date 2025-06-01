# Financial Analysis Project - EDA and Quantitative Analysis

## 📊 **Project Overview**

This project provides comprehensive financial analysis capabilities including both Exploratory Data Analysis (EDA) for financial news data and advanced Quantitative Analysis for stock price data using industry-standard libraries.

## 🎯 **Tasks Completed**

### ✅ **Task 1: Financial News EDA**

- **Dataset**: Financial News and Stock Price Integration Dataset (FNSPID)
- **Size**: 313MB, 1.4M+ financial news articles
- **Features**: Advanced text analysis, time series patterns, publisher analytics

### ✅ **Task 2: Quantitative Analysis** ⭐ **NEW**

- **Dataset**: Historical stock price data (OHLCV)
- **Stocks**: AAPL, AMZN, GOOG, META, MSFT, NVDA, TSLA
- **Tools**: TA-Lib for technical indicators, PyNance for financial metrics

---

## 🚀 **Quick Start**

### **Installation**

```bash
# Clone the repository
git clone <repository-url>
cd financial-analysis-project

# Install dependencies
pip install -r requirements.txt

# Additional packages for Task 2 (Quantitative Analysis)
pip install TA-Lib pynance yfinance plotly cufflinks mplfinance

# Note: TA-Lib may require additional system dependencies
# For Windows: pip install --find-links https://github.com/cgohlke/talib-build/releases TA-Lib
```

### **Quick Execution**

```bash
# Task 1: Financial News EDA
python scripts/run_comprehensive_eda.py

# Task 2: Quantitative Analysis
python scripts/run_task2_analysis.py
```

---

## 📈 **Task 2: Quantitative Analysis using PyNance and TA-Lib**

### **🎯 Objectives**

- Load and prepare historical stock price data (OHLCV)
- Apply technical indicators using **TA-Lib**
- Calculate financial metrics using **PyNance**
- Create comprehensive visualizations
- Generate trading signals and portfolio analysis

### **🔧 Technical Indicators (TA-Lib)**

- **Moving Averages**: SMA, EMA (5, 10, 20, 50, 100, 200 periods)
- **Momentum**: RSI, MACD, Stochastic Oscillator
- **Volatility**: Bollinger Bands, Average True Range (ATR)
- **Volume**: OBV, Accumulation/Distribution Line
- **Support/Resistance**: Pivot Points, Rolling Levels

### **💰 Financial Metrics (PyNance)**

- **Return Metrics**: Total return, annualized return, risk-adjusted returns
- **Risk Metrics**: Volatility, VaR, CVaR, Maximum Drawdown
- **Performance Ratios**: Sharpe, Sortino, Calmar, Information, Treynor
- **Distribution Analysis**: Skewness, kurtosis, percentile analysis
- **Portfolio Metrics**: Correlation, beta, tracking error

### **📊 Key Features**

#### **Modular Architecture**

```
src/
├── technical_analyzer.py      # TA-Lib technical indicators
├── financial_metrics_analyzer.py  # PyNance financial metrics
├── data_loader.py            # Multi-stock data loading
└── (Task 1 modules...)
```

#### **Analysis Capabilities**

- **Individual Stock Analysis**: Comprehensive technical and fundamental analysis
- **Portfolio Analysis**: Equal-weighted portfolio construction and analysis
- **Correlation Analysis**: Inter-stock relationship mapping
- **Risk Analysis**: VaR, CVaR, drawdown analysis
- **Trading Signals**: Multi-indicator signal generation

#### **Visualization Suite**

- **Technical Charts**: Price, indicators, volume analysis
- **Financial Charts**: Returns distribution, drawdown, risk metrics
- **Portfolio Charts**: Correlation heatmaps, efficient frontier
- **Comparison Charts**: Multi-stock performance comparison

### **📋 Usage Examples**

#### **Interactive Analysis (Jupyter)**

```python
# Open the comprehensive notebook
jupyter notebook notebooks/task2_eda.ipynb
```

#### **Programmatic Analysis**

```python
from src.data_loader import FinancialDataLoader
from src.technical_analyzer import TechnicalAnalyzer
from src.financial_metrics_analyzer import FinancialMetricsAnalyzer

# Load data
loader = FinancialDataLoader("data/yfinance_data")
stock_data = loader.load_all_stocks()

# Technical analysis
tech_analyzer = TechnicalAnalyzer(stock_data['AAPL'])
technical_analysis = tech_analyzer.get_comprehensive_technical_analysis()

# Financial analysis
finance_analyzer = FinancialMetricsAnalyzer(stock_data['AAPL'])
financial_analysis = finance_analyzer.get_comprehensive_financial_analysis()

# Generate plots
tech_analyzer.plot_technical_analysis()
finance_analyzer.plot_financial_analysis()
```

### **📁 Output Structure**

```
data/results/task2_quantitative_analysis/
├── executive_summary.json           # Comprehensive analysis summary
├── executive_summary.txt            # Human-readable summary
├── summary_statistics.csv           # All stocks summary stats
├── portfolio_analysis.png           # Portfolio visualization
├── correlation_matrix.png           # Stock correlation heatmap
└── [TICKER]/                        # Individual stock analysis
    ├── [TICKER]_technical_analysis.png
    ├── [TICKER]_financial_analysis.png
    └── [TICKER]_analysis_summary.json
```

### **🎯 Key Performance Indicators (KPIs)**

#### **Technical Analysis Accuracy**

- ✅ **RSI Signals**: Overbought (>70) / Oversold (<30) detection
- ✅ **MACD Signals**: Bullish/Bearish crossover identification
- ✅ **Moving Average Signals**: Trend direction analysis
- ✅ **Bollinger Bands**: Volatility breakout detection
- ✅ **Volume Confirmation**: Price movement validation

#### **Financial Metrics Completeness**

- ✅ **Return Analysis**: Total, annualized, risk-adjusted returns
- ✅ **Risk Assessment**: VaR, CVaR, maximum drawdown
- ✅ **Performance Ratios**: Sharpe, Sortino, Calmar ratios
- ✅ **Distribution Analysis**: Skewness, kurtosis, tail analysis
- ✅ **Portfolio Metrics**: Correlation, diversification benefits

#### **Data Analysis Depth**

- ✅ **Multi-timeframe Analysis**: Daily, monthly, rolling metrics
- ✅ **Cross-stock Comparison**: Relative performance analysis
- ✅ **Portfolio Construction**: Equal-weighted portfolio analysis
- ✅ **Signal Generation**: Multi-indicator trading signals

---

## 📊 **Task 1: Financial News EDA** (Previously Completed)

### **🔍 Advanced Text Analysis**

- **NLP Pipeline**: NLTK-based tokenization, lemmatization, sentiment analysis
- **Financial Keywords**: Automated detection of earnings, M&A, regulatory terms
- **Topic Modeling**: Financial phrase categorization and n-gram analysis
- **Sentiment Scoring**: Domain-specific financial sentiment analysis

### **⏰ Time Series Analysis**

- **Publication Patterns**: Intraday, daily, seasonal trend analysis
- **Market Event Detection**: Statistical spike detection algorithm
- **Trading Hours Optimization**: 9:30AM-4PM ET focus for maximum relevance
- **Seasonal Trends**: Monthly, quarterly, yearly pattern identification

### **📰 Enhanced Publisher Analysis**

- **Source Classification**: Major financial news vs. individual contributors
- **Domain Analysis**: Email and URL domain extraction and categorization
- **Content Type Mapping**: News vs. analysis vs. opinion classification
- **Credibility Scoring**: Publisher reliability assessment

---

## 🏗️ **Project Structure**

```
financial-analysis-project/
├── 📁 src/                          # Core analysis modules
│   ├── 🔧 technical_analyzer.py     # TA-Lib technical indicators
│   ├── 💰 financial_metrics_analyzer.py  # PyNance financial metrics
│   ├── 📂 data_loader.py            # Multi-stock data management
│   ├── 📝 text_analyzer.py          # NLP and text analysis
│   ├── ⏰ time_series_analyzer.py    # Temporal pattern analysis
│   ├── 📰 publisher_analyzer.py     # Publisher and source analysis
│   └── 📊 eda_analyzer.py           # Core EDA functionality
├── 📁 scripts/                      # Execution scripts
│   ├── 🚀 run_task2_analysis.py     # Task 2 automated execution
│   └── 🚀 run_comprehensive_eda.py  # Task 1 automated execution
├── 📁 notebooks/                    # Interactive analysis
│   ├── 📊 task2_eda.ipynb           # Task 2 quantitative analysis
│   └── 📊 task1_eda.ipynb           # Task 1 financial news EDA
├── 📁 data/                         # Data storage
│   ├── 📈 yfinance_data/            # Stock price data (OHLCV)
│   │   ├── AAPL_historical_data.csv
│   │   ├── AMZN_historical_data.csv
│   │   ├── GOOG_historical_data.csv
│   │   ├── META_historical_data.csv
│   │   ├── MSFT_historical_data.csv
│   │   ├── NVDA_historical_data.csv
│   │   └── TSLA_historical_data.csv
│   ├── 📰 raw_analyst_ratings.csv   # Financial news dataset (313MB)
│   └── 📁 results/                  # Analysis outputs
│       ├── 📁 task1_financial_news_eda/
│       └── 📁 task2_quantitative_analysis/
├── 📄 requirements.txt              # Python dependencies
└── 📖 README.md                     # Project documentation
```

---

## 🔧 **Dependencies**

### **Core Libraries**

```txt
pandas>=2.0.0          # Data manipulation and analysis
numpy>=1.24.0           # Numerical computing
matplotlib>=3.6.0       # Basic plotting
seaborn>=0.12.0         # Statistical visualization
scipy>=1.10.0           # Scientific computing
jupyter>=1.0.0          # Interactive notebooks
```

### **Task 2 - Quantitative Analysis**

```txt
TA-Lib>=0.4.0          # Technical analysis indicators
pynance>=0.5.0         # Financial metrics and risk analysis
yfinance>=0.2.0        # Yahoo Finance data (backup)
plotly>=5.0.0          # Interactive plots
cufflinks>=0.17.0      # Plotly integration
mplfinance>=0.12.0     # Financial plotting
```

### **Task 1 - Text Analysis**

```txt
nltk>=3.8              # Natural language processing
textblob>=0.17.0       # Sentiment analysis
wordcloud>=1.9.0       # Word cloud generation
```

---

## 🎯 **Usage Guide**

### **For Task 2: Quantitative Analysis**

#### **1. Quick Start**

```bash
# Run complete analysis
python scripts/run_task2_analysis.py
```

#### **2. Interactive Analysis**

```bash
# Open Jupyter notebook
jupyter notebook notebooks/task2_eda.ipynb
```

#### **3. Custom Analysis**

```python
# Load specific stock
from src.data_loader import FinancialDataLoader
loader = FinancialDataLoader()
aapl_data = loader.get_stock_data('AAPL')

# Technical analysis
from src.technical_analyzer import TechnicalAnalyzer
tech = TechnicalAnalyzer(aapl_data)
rsi = tech.calculate_rsi()
macd = tech.calculate_macd()

# Financial metrics
from src.financial_metrics_analyzer import FinancialMetricsAnalyzer
finance = FinancialMetricsAnalyzer(aapl_data)
sharpe = finance.calculate_sharpe_ratio()
var = finance.calculate_var_cvar()
```

### **For Task 1: Financial News EDA**

#### **1. Quick Start**

```bash
# Run complete EDA
python scripts/run_comprehensive_eda.py
```

#### **2. Interactive Analysis**

```bash
# Open Jupyter notebook
jupyter notebook notebooks/task1_eda.ipynb
```

---

## 📊 **Expected Outputs**

### **Task 2 Deliverables**

- 📈 **Technical Analysis Charts**: RSI, MACD, Bollinger Bands, volume analysis
- 💰 **Financial Metrics Dashboard**: Sharpe ratio, VaR, drawdown analysis
- 📊 **Portfolio Analysis**: Correlation matrix, risk-return scatter plots
- 🎯 **Trading Signals**: Multi-indicator buy/sell recommendations
- 📋 **Executive Summary**: JSON and text format comprehensive reports

### **Task 1 Deliverables**

- 📝 **Text Analytics Dashboard**: Keyword extraction, sentiment analysis
- ⏰ **Time Series Insights**: Publication patterns, market event detection
- 📰 **Publisher Analytics**: Source credibility, content classification
- 📊 **Interactive Visualizations**: Word clouds, trend charts, heatmaps

---

## 🏆 **Key Achievements**

### **Technical Excellence**

- ✅ **Modular Design**: Reusable, maintainable code architecture
- ✅ **Error Handling**: Graceful fallbacks for missing dependencies
- ✅ **Performance**: Optimized for large datasets (1.4M+ records)
- ✅ **Documentation**: Comprehensive inline and README documentation

### **Analytical Depth**

- ✅ **Multi-dimensional Analysis**: Technical, fundamental, and sentiment analysis
- ✅ **Risk Management**: VaR, CVaR, drawdown, correlation analysis
- ✅ **Signal Generation**: Actionable trading recommendations
- ✅ **Portfolio Optimization**: Diversification and risk-return analysis

### **Industry Standards**

- ✅ **TA-Lib Integration**: Professional-grade technical indicators
- ✅ **PyNance Integration**: Advanced financial risk metrics
- ✅ **Trading System Ready**: Production-ready signal generation
- ✅ **Institutional Quality**: Bank-grade risk and performance analysis

---

## 🔧 **Troubleshooting**

### **TA-Lib Installation Issues**

```bash
# Windows
pip install --find-links https://github.com/cgohlke/talib-build/releases TA-Lib

# macOS
brew install ta-lib
pip install TA-Lib

# Linux (Ubuntu/Debian)
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

### **NLTK Data Issues**

```python
# Download required NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### **Memory Issues with Large Datasets**

```python
# Process data in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

---

## 📞 **Support**

For questions, issues, or contributions:

1. **Documentation**: Check the comprehensive README and code comments
2. **Dependencies**: Ensure all required packages are installed
3. **Data Path**: Verify data files are in the correct directories
4. **Environment**: Use Python 3.8+ for best compatibility

---

## 🎉 **Ready for Production**

This comprehensive financial analysis framework is ready for:

- 🏦 **Trading Systems**: Real-time signal generation
- 📊 **Risk Management**: Portfolio risk assessment
- 💼 **Investment Analysis**: Multi-asset performance evaluation
- 📈 **Market Research**: Quantitative market analysis
- 🔍 **Financial Research**: Academic and professional research

**🚀 Transform your financial analysis with professional-grade tools and insights!**
