"""
Quick test script for text analysis without NLTK dependencies
Run this to test text analysis functionality
"""

import sys
import os
sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the simple text analyzer (no NLTK required)
from simple_text_analyzer import SimpleFinancialTextAnalyzer
from data_loader import FinancialDataLoader

def test_text_analysis():
    """Test the text analysis functionality."""
    
    print("Testing Financial Text Analysis (No NLTK required)")
    print("=" * 60)
    
    # Load a small sample of data for testing
    try:
        data_path = "../data/raw_analyst_ratings.csv"
        loader = FinancialDataLoader(data_path)
        df = loader.load_data()
        
        # Use a smaller sample for quick testing
        sample_size = min(1000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        print(f"Loaded {len(df_sample)} articles for testing")
        
        # Initialize simple text analyzer
        text_analyzer = SimpleFinancialTextAnalyzer(df_sample, text_column='headline')
        print("Text analyzer initialized successfully!")
        
        # Test keyword extraction
        print("\n1. Testing keyword extraction...")
        keywords_df = text_analyzer.extract_common_keywords(15)
        print("Top 15 Keywords:")
        for idx, row in keywords_df.iterrows():
            print(f"  {idx+1:2d}. {row['keyword']:<20} | {row['frequency']:>6} ({row['percentage']:>5.2f}%)")
        
        # Test financial phrases
        print("\n2. Testing financial phrase detection...")
        financial_phrases = text_analyzer.extract_financial_phrases()
        print("Financial Phrases by Category:")
        for category, data in financial_phrases.items():
            if data['total_mentions'] > 0:
                print(f"  {category.replace('_', ' ').title()}: {data['total_mentions']} mentions")
        
        # Test sentiment analysis
        print("\n3. Testing sentiment analysis...")
        sentiment_analysis = text_analyzer.sentiment_keyword_analysis()
        sentiment_dist = sentiment_analysis['sentiment_distribution']
        total_sentiment = sum(sentiment_dist.values())
        
        print("Sentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            pct = (count / total_sentiment) * 100 if total_sentiment > 0 else 0
            print(f"  {sentiment.title()}: {count} articles ({pct:.1f}%)")
        
        # Test bigrams
        print("\n4. Testing bigram analysis...")
        bigrams_df = text_analyzer.analyze_ngrams(2, 10)
        print("Top 10 Bigrams:")
        for idx, row in bigrams_df.iterrows():
            print(f"  {idx+1:2d}. '{row['2gram']}'  | {row['frequency']} occurrences")
        
        print("\n" + "=" * 60)
        print("✅ All text analysis tests completed successfully!")
        print("You can now use the SimpleFinancialTextAnalyzer in your notebook.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    test_text_analysis() 