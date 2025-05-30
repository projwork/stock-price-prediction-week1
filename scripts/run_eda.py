"""
Script to run the complete EDA analysis on the Financial News dataset.
"""

import sys
import os
sys.path.append('../src')

from data_loader import FinancialDataLoader
from eda_analyzer import FinancialEDAAnalyzer
from utils import save_analysis_results, get_data_quality_report

def main():
    """Main function to run the EDA analysis."""
    
    # Configuration
    data_path = "../data/raw_analyst_ratings.csv"
    results_dir = "../results"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    print("Starting Financial News EDA Analysis...")
    print("=" * 50)
    
    # Load data
    print("1. Loading data...")
    loader = FinancialDataLoader(data_path)
    df = loader.load_data()
    
    # Data quality check
    print("2. Performing data quality assessment...")
    quality_report = get_data_quality_report(df)
    save_analysis_results(quality_report, f"{results_dir}/data_quality_report.json")
    
    # Initialize analyzer
    print("3. Initializing EDA analyzer...")
    analyzer = FinancialEDAAnalyzer(df)
    
    # Perform analyses
    print("4. Performing descriptive statistics analysis...")
    desc_stats = analyzer.get_descriptive_statistics()
    
    print("5. Analyzing publishers...")
    publisher_analysis = analyzer.analyze_publishers()
    
    print("6. Analyzing publication trends...")
    trends = analyzer.analyze_publication_trends()
    
    # Compile results
    results = {
        'descriptive_statistics': desc_stats,
        'publisher_analysis': publisher_analysis.head(20).to_dict('records'),
        'publication_trends': {
            'day_of_week': trends['day_of_week_trends'].to_dict('records'),
            'hourly': trends['hourly_trends'].to_dict('records'),
            'monthly': trends['monthly_trends'].to_dict('records')
        }
    }
    
    # Save results
    save_analysis_results(results, f"{results_dir}/complete_eda_results.json")
    
    print("7. Analysis completed successfully!")
    print(f"   Results saved to: {results_dir}/")
    print(f"   - data_quality_report.json")
    print(f"   - complete_eda_results.json")

if __name__ == "__main__":
    main() 