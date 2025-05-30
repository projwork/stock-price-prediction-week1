"""
Script to run the comprehensive EDA analysis on the Financial News dataset.
Includes text analysis, time series analysis, and advanced publisher analysis.
"""

import sys
import os
sys.path.append('../src')

from data_loader import FinancialDataLoader
from eda_analyzer import FinancialEDAAnalyzer
from text_analyzer import FinancialTextAnalyzer
from time_series_analyzer import FinancialTimeSeriesAnalyzer
from publisher_analyzer import FinancialPublisherAnalyzer
from utils import save_analysis_results, get_data_quality_report, format_large_numbers

def main():
    """Main function to run the comprehensive EDA analysis."""
    
    # Configuration
    data_path = "../data/raw_analyst_ratings.csv"
    results_dir = "../results"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    print("Starting Comprehensive Financial News EDA Analysis...")
    print("=" * 60)
    
    # Load data
    print("1. Loading data...")
    loader = FinancialDataLoader(data_path)
    df = loader.load_data()
    print(f"   Data loaded: {df.shape[0]:,} articles, {df.shape[1]} columns")
    
    # Data quality check
    print("2. Performing data quality assessment...")
    quality_report = get_data_quality_report(df)
    save_analysis_results(quality_report, f"{results_dir}/data_quality_report.json")
    print(f"   Quality report saved")
    
    # Basic EDA
    print("3. Performing basic EDA analysis...")
    basic_analyzer = FinancialEDAAnalyzer(df)
    desc_stats = basic_analyzer.get_descriptive_statistics()
    publisher_analysis = basic_analyzer.analyze_publishers()
    trends = basic_analyzer.analyze_publication_trends()
    
    # Text Analysis
    print("4. Performing text analysis and topic modeling...")
    text_analyzer = FinancialTextAnalyzer(df, text_column='headline')
    text_results = text_analyzer.get_comprehensive_text_analysis()
    print(f"   Found {len(text_results['top_keywords'])} top keywords")
    print(f"   Analyzed {len(text_results['financial_phrases'])} financial phrase categories")
    
    # Time Series Analysis
    print("5. Performing time series analysis...")
    ts_analyzer = FinancialTimeSeriesAnalyzer(df, date_column='date')
    ts_results = ts_analyzer.get_comprehensive_time_series_analysis()
    spike_days = ts_results['spike_analysis']['total_spike_days']
    print(f"   Detected {spike_days} publication spike days")
    print(f"   Analyzed {ts_results['data_summary']['unique_days']} unique days")
    
    # Advanced Publisher Analysis
    print("6. Performing advanced publisher analysis...")
    publisher_analyzer = FinancialPublisherAnalyzer(df, publisher_column='publisher', url_column='url')
    publisher_results = publisher_analyzer.get_comprehensive_publisher_analysis()
    total_publishers = publisher_results['publisher_activity']['total_publishers']
    print(f"   Analyzed {total_publishers} unique publishers")
    
    # Compile comprehensive results
    comprehensive_results = {
        'dataset_overview': {
            'total_articles': len(df),
            'total_publishers': len(df['publisher'].unique()) if 'publisher' in df.columns else 0,
            'date_range': {
                'start': str(df['date'].min()) if 'date' in df.columns else 'N/A',
                'end': str(df['date'].max()) if 'date' in df.columns else 'N/A'
            },
            'unique_stocks': len(df['stock'].unique()) if 'stock' in df.columns else 0
        },
        'basic_eda': {
            'descriptive_statistics': desc_stats,
            'publisher_analysis': publisher_analysis.head(20).to_dict('records'),
            'publication_trends': {
                'day_of_week': trends['day_of_week_trends'].to_dict('records'),
                'hourly': trends['hourly_trends'].to_dict('records'),
                'monthly': trends['monthly_trends'].to_dict('records')
            }
        },
        'text_analysis': text_results,
        'time_series_analysis': ts_results,
        'publisher_analysis': publisher_results,
        'data_quality': quality_report
    }
    
    # Save comprehensive results
    save_analysis_results(comprehensive_results, f"{results_dir}/comprehensive_eda_results.json")
    
    # Generate executive summary
    generate_executive_summary(comprehensive_results, results_dir)
    
    print("7. Analysis completed successfully!")
    print(f"   Results saved to: {results_dir}/")
    print(f"   - comprehensive_eda_results.json")
    print(f"   - executive_summary.txt")
    print(f"   - data_quality_report.json")

def generate_executive_summary(results, results_dir):
    """Generate an executive summary of the analysis."""
    
    overview = results['dataset_overview']
    text_analysis = results['text_analysis']
    ts_analysis = results['time_series_analysis']
    pub_analysis = results['publisher_analysis']
    
    summary_text = f"""
FINANCIAL NEWS EDA ANALYSIS - EXECUTIVE SUMMARY
================================================

DATASET OVERVIEW:
- Total Articles: {format_large_numbers(overview['total_articles'])}
- Total Publishers: {format_large_numbers(overview['total_publishers'])}
- Unique Stocks: {format_large_numbers(overview['unique_stocks'])}
- Date Range: {overview['date_range']['start']} to {overview['date_range']['end']}

KEY FINDINGS:

1. TEXT ANALYSIS INSIGHTS:
   - Most Common Keyword: '{text_analysis['top_keywords'][0]['keyword']}' ({text_analysis['top_keywords'][0]['frequency']} occurrences)
   - Sentiment Distribution:
"""
    
    # Add sentiment analysis
    sentiment_dist = text_analysis['sentiment_analysis']['sentiment_distribution']
    total_sentiment = sum(sentiment_dist.values())
    for sentiment, count in sentiment_dist.items():
        pct = (count / total_sentiment) * 100 if total_sentiment > 0 else 0
        summary_text += f"     • {sentiment.title()}: {count} articles ({pct:.1f}%)\n"
    
    # Add financial phrases
    summary_text += "\n   - Top Financial Categories:\n"
    for category, data in list(text_analysis['financial_phrases'].items())[:5]:
        if data['total_mentions'] > 0:
            summary_text += f"     • {category.replace('_', ' ').title()}: {data['total_mentions']} mentions\n"
    
    # Add time series insights
    freq_data = ts_analysis['publication_frequency']['daily']
    spike_data = ts_analysis['spike_analysis']
    intraday_data = ts_analysis['intraday_patterns']
    
    summary_text += f"""
2. TIME SERIES INSIGHTS:
   - Average Daily Articles: {freq_data['avg_daily']:.1f}
   - Peak Publishing Day: {freq_data['max_daily']} articles
   - Spike Days Detected: {spike_data['total_spike_days']} ({spike_data['spike_percentage']:.1f}% of days)
   - Market Hours Articles: {intraday_data['market_hours_percentage']:.1f}% of total
   - Pre-Market Articles: {format_large_numbers(intraday_data['pre_market_articles'])}
   - After-Market Articles: {format_large_numbers(intraday_data['after_market_articles'])}
"""
    
    # Add publisher insights
    pub_activity = pub_analysis['publisher_activity']
    top_pub_type = max(pub_activity['publisher_type_distribution'].items(), key=lambda x: x[1])
    
    summary_text += f"""
3. PUBLISHER INSIGHTS:
   - Total Publishers: {format_large_numbers(pub_activity['total_publishers'])}
   - Email-based Publishers: {pub_activity['email_publisher_stats']['email_percentage']:.1f}%
   - Most Common Publisher Type: {top_pub_type[0].replace('_', ' ').title()} ({format_large_numbers(top_pub_type[1])} articles)
   - Single-Article Publishers: {format_large_numbers(pub_activity['single_article_publishers'])}
"""
    
    # Add top publishers
    summary_text += "\n   - Top 5 Publishers:\n"
    for idx, (publisher, count) in enumerate(list(pub_activity['top_publishers'].items())[:5], 1):
        percentage = (count / overview['total_articles']) * 100
        summary_text += f"     {idx}. {publisher} - {format_large_numbers(count)} articles ({percentage:.2f}%)\n"
    
    summary_text += f"""
RECOMMENDATIONS FOR TRADING SYSTEMS:
1. Focus on market hours (9 AM - 4 PM) for {intraday_data['market_hours_percentage']:.1f}% of news flow
2. Monitor spike days (threshold: {spike_data['threshold_used']:.1f} articles/day) for market events
3. Track key financial phrases: earnings, price targets, regulatory approvals
4. Consider publisher credibility - major financial news sources vs. individual contributors
5. Implement sentiment analysis for {sentiment_dist.get('positive', 0) + sentiment_dist.get('negative', 0)} sentiment-bearing articles

ANALYSIS COMPLETED: {overview['date_range']['end']}
"""
    
    # Save summary
    with open(f"{results_dir}/executive_summary.txt", 'w') as f:
        f.write(summary_text)

if __name__ == "__main__":
    main() 