# Financial News and Stock Price Integration Dataset (FNSPID) - Comprehensive EDA

A comprehensive Exploratory Data Analysis framework for financial news data, featuring advanced text analysis, time series analysis, and publisher analysis capabilities.

## 📊 Dataset Description

The FNSPID dataset contains financial news articles with the following structure:

- **headline**: Article release headline/title
- **url**: Direct link to the full news article
- **publisher**: Author/creator of article
- **date**: Publication date and time (UTC-4 timezone)
- **stock**: Stock ticker symbol

## 🏗️ Project Structure

```
├── src/                          # Source code modules
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── eda_analyzer.py          # Basic EDA analysis
│   ├── text_analyzer.py         # Text analysis and topic modeling
│   ├── time_series_analyzer.py  # Time series analysis
│   ├── publisher_analyzer.py    # Advanced publisher analysis
│   └── utils.py                 # Utility functions
├── scripts/                      # Execution scripts
│   ├── run_eda.py               # Basic EDA script
│   └── run_comprehensive_eda.py # Complete analysis script
├── notebooks/                    # Jupyter notebooks
│   └── task1_eda.ipynb         # Interactive EDA notebook
├── results/                      # Analysis outputs
├── data/                        # Dataset files
└── requirements.txt             # Dependencies
```

## 🔍 Analysis Capabilities

### 1. Text Analysis & Topic Modeling

- **Keyword Extraction**: Identifies most common terms in headlines
- **Financial Phrase Detection**: Tracks specific financial terminology:
  - Earnings-related: earnings, EPS, profit, revenue
  - Price targets: upgrades, downgrades, analyst ratings
  - M&A activity: mergers, acquisitions, takeovers
  - Regulatory events: FDA approvals, SEC filings
  - Market sentiment: bullish, bearish, volatile
- **N-gram Analysis**: Bigrams and trigrams for phrase patterns
- **Sentiment Analysis**: Positive, negative, and neutral keyword detection

### 2. Time Series Analysis

- **Publication Frequency**: Daily, weekly, monthly patterns
- **Spike Detection**: Identifies unusual publication volumes (potential market events)
- **Intraday Patterns**: Market hours vs. pre/post-market analysis
- **Seasonal Trends**: Quarterly and monthly publication patterns
- **Trading Hours Analysis**: Optimized for US market hours (9:30 AM - 4:00 PM ET)

### 3. Advanced Publisher Analysis

- **Publisher Classification**: Categorizes publishers by type:
  - Major financial news organizations
  - Analyst firms and research houses
  - Individual contributors
  - Email-based reporters
- **Domain Analysis**: Extracts and analyzes email domains and URL sources
- **Content Type Analysis**: Maps publisher types to content categories
- **Activity Patterns**: Identifies most active publishers and their specializations

### 4. Descriptive Statistics

- **Text Length Analysis**: Headline and URL character distributions
- **Data Quality Assessment**: Missing values, duplicates, data types
- **Coverage Analysis**: Stock symbol distribution and frequency

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure data file is placed in data/ directory
# Expected file: data/raw_analyst_ratings.csv
```

### Run Analysis

#### Option 1: Interactive Notebook (Recommended)

```bash
jupyter notebook notebooks/task1_eda.ipynb
```

#### Option 2: Command Line Script

```bash
cd scripts
python run_comprehensive_eda.py
```

#### Option 3: Basic EDA Only

```bash
cd scripts
python run_eda.py
```

## 📈 Key Features for Trading Systems

### Market Event Detection

- **Spike Detection Algorithm**: Identifies days with abnormally high news volume
- **Threshold**: Configurable standard deviation-based detection
- **Event Correlation**: Links publication spikes to potential market events

### Timing Analysis

- **Market Hours Coverage**: Tracks what percentage of news breaks during trading hours
- **Pre-Market Intelligence**: Analyzes overnight and pre-market news flow
- **Minute-Level Patterns**: Identifies optimal news monitoring times

### Source Credibility

- **Publisher Reputation**: Distinguishes between major financial news and individual contributors
- **Domain Classification**: Categorizes news sources by organization type
- **Content Specialization**: Maps publishers to their content focus areas

### Sentiment Intelligence

- **Financial Sentiment Keywords**: Custom financial vocabulary for sentiment analysis
- **Event Classification**: Categorizes news by type (earnings, M&A, regulatory, etc.)
- **Impact Assessment**: Measures sentiment distribution across news volume

### Analysis Sections Include:

1. **Dataset Overview**: Volume, date range, coverage statistics
2. **Text Analysis**: Keywords, phrases, sentiment distribution
3. **Time Series Insights**: Publication patterns, spike detection
4. **Publisher Intelligence**: Source analysis and credibility metrics
5. **Trading Recommendations**: Actionable insights for automated systems

## 🔧 Customization

### Adjusting Analysis Parameters

#### Text Analysis

```python
# Modify financial keyword categories
financial_keywords = {
    'earnings': ['earnings', 'eps', 'profit', 'revenue'],
    'custom_category': ['your', 'keywords', 'here']
}
```
