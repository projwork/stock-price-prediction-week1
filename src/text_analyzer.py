import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import re
from collections import Counter
import warnings
import logging

# Configure NLTK downloads first
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data with better error handling
def download_nltk_data():
    """Download required NLTK data with proper error handling."""
    required_packages = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet')
    ]
    
    for data_path, package_name in required_packages:
        try:
            nltk.data.find(data_path)
            print(f"NLTK {package_name} already downloaded")
        except LookupError:
            print(f"Downloading NLTK {package_name}...")
            try:
                nltk.download(package_name, quiet=True)
                print(f"Successfully downloaded NLTK {package_name}")
            except Exception as e:
                print(f"Failed to download NLTK {package_name}: {e}")
                print("You may need to download manually or check your internet connection")

# Download NLTK data
download_nltk_data()

# Now import NLTK components
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
except ImportError as e:
    print(f"NLTK import error: {e}")
    print("Please ensure NLTK is properly installed: pip install nltk")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FinancialTextAnalyzer:
    """Class to perform text analysis and topic modeling on financial news data."""
    
    def __init__(self, data: pd.DataFrame, text_column: str = 'headline'):
        """
        Initialize the text analyzer.
        
        Args:
            data (pd.DataFrame): The financial news dataset
            text_column (str): Column containing text to analyze
        """
        self.data = data.copy()
        self.text_column = text_column
        
        # Initialize NLTK components with error handling
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Error initializing NLTK components: {e}")
            print("Using basic preprocessing without lemmatization and stop words")
            self.lemmatizer = None
            self.stop_words = set()
        
        # Add financial stop words
        financial_stop_words = {
            'stock', 'share', 'company', 'market', 'trading', 'price',
            'nasdaq', 'nyse', 'inc', 'corp', 'ltd', 'llc', 'co'
        }
        self.stop_words.update(financial_stop_words)
        
        # Financial keywords and phrases to track
        self.financial_keywords = {
            'earnings': ['earnings', 'eps', 'profit', 'revenue', 'sales'],
            'price_targets': ['price target', 'target price', 'pt', 'upgraded', 'downgraded'],
            'mergers_acquisitions': ['merger', 'acquisition', 'buyout', 'takeover', 'deal'],
            'regulatory': ['fda', 'sec', 'regulatory', 'approval', 'compliance'],
            'financial_performance': ['beats', 'misses', 'guidance', 'outlook', 'forecast'],
            'market_sentiment': ['bullish', 'bearish', 'optimistic', 'pessimistic', 'volatile'],
            'analyst_actions': ['upgrade', 'downgrade', 'initiate', 'coverage', 'rating'],
            'corporate_actions': ['dividend', 'split', 'buyback', 'ipo', 'offering']
        }
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for analysis with fallback for NLTK issues.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            List[str]: Preprocessed tokens
        """
        if pd.isna(text):
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize with fallback
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            print(f"NLTK tokenization failed, using basic split: {e}")
            # Fallback to basic splitting
            tokens = text.split()
        
        # Remove stop words and short words with lemmatization fallback
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                if self.lemmatizer:
                    try:
                        token = self.lemmatizer.lemmatize(token)
                    except Exception:
                        pass  # Use original token if lemmatization fails
                processed_tokens.append(token)
        
        return processed_tokens
    
    def extract_common_keywords(self, top_n: int = 50) -> pd.DataFrame:
        """
        Extract most common keywords from headlines.
        
        Args:
            top_n (int): Number of top keywords to return
            
        Returns:
            pd.DataFrame: Common keywords with counts
        """
        all_tokens = []
        
        for text in self.data[self.text_column].dropna():
            tokens = self.preprocess_text(text)
            all_tokens.extend(tokens)
        
        # Count frequency
        word_freq = Counter(all_tokens)
        
        # Create DataFrame
        keywords_df = pd.DataFrame(word_freq.most_common(top_n), 
                                 columns=['keyword', 'frequency'])
        
        # Calculate percentage
        total_words = sum(word_freq.values())
        keywords_df['percentage'] = (keywords_df['frequency'] / total_words * 100).round(3)
        
        return keywords_df
    
    def extract_financial_phrases(self) -> Dict[str, Any]:
        """
        Extract specific financial phrases and keywords.
        
        Returns:
            Dict: Analysis of financial phrases
        """
        phrase_analysis = {}
        
        for category, phrases in self.financial_keywords.items():
            category_matches = []
            
            for text in self.data[self.text_column].dropna():
                text_lower = text.lower()
                matches = []
                
                for phrase in phrases:
                    if phrase in text_lower:
                        matches.append(phrase)
                
                if matches:
                    category_matches.extend(matches)
            
            if category_matches:
                phrase_counter = Counter(category_matches)
                phrase_analysis[category] = {
                    'total_mentions': len(category_matches),
                    'unique_phrases': len(phrase_counter),
                    'top_phrases': dict(phrase_counter.most_common(5)),
                    'articles_containing': len([text for text in self.data[self.text_column].dropna() 
                                              if any(phrase in text.lower() for phrase in phrases)])
                }
            else:
                phrase_analysis[category] = {
                    'total_mentions': 0,
                    'unique_phrases': 0,
                    'top_phrases': {},
                    'articles_containing': 0
                }
        
        return phrase_analysis
    
    def analyze_ngrams(self, n: int = 2, top_k: int = 20) -> pd.DataFrame:
        """
        Extract and analyze n-grams from headlines.
        
        Args:
            n (int): N-gram size (2 for bigrams, 3 for trigrams)
            top_k (int): Number of top n-grams to return
            
        Returns:
            pd.DataFrame: Top n-grams with frequencies
        """
        ngrams = []
        
        for text in self.data[self.text_column].dropna():
            tokens = self.preprocess_text(text)
            if len(tokens) >= n:
                text_ngrams = [' '.join(tokens[i:i+n]) 
                             for i in range(len(tokens) - n + 1)]
                ngrams.extend(text_ngrams)
        
        # Count frequency
        ngram_freq = Counter(ngrams)
        
        # Create DataFrame
        ngrams_df = pd.DataFrame(ngram_freq.most_common(top_k), 
                               columns=[f'{n}gram', 'frequency'])
        
        return ngrams_df
    
    def sentiment_keyword_analysis(self) -> Dict[str, Any]:
        """
        Analyze sentiment-related keywords in headlines.
        
        Returns:
            Dict: Sentiment keyword analysis
        """
        positive_words = ['up', 'rise', 'gain', 'surge', 'rally', 'boost', 'jump', 
                         'soar', 'climb', 'advance', 'strong', 'beats', 'exceeds']
        
        negative_words = ['down', 'fall', 'drop', 'decline', 'crash', 'plunge', 
                         'sink', 'tumble', 'weak', 'misses', 'disappoints', 'cuts']
        
        neutral_words = ['hold', 'maintain', 'stable', 'steady', 'unchanged', 'flat']
        
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        sentiment_details = {'positive': [], 'negative': [], 'neutral': []}
        
        for text in self.data[self.text_column].dropna():
            text_lower = text.lower()
            
            pos_matches = [word for word in positive_words if word in text_lower]
            neg_matches = [word for word in negative_words if word in text_lower]
            neu_matches = [word for word in neutral_words if word in text_lower]
            
            if pos_matches:
                sentiment_counts['positive'] += 1
                sentiment_details['positive'].extend(pos_matches)
            elif neg_matches:
                sentiment_counts['negative'] += 1
                sentiment_details['negative'].extend(neg_matches)
            elif neu_matches:
                sentiment_counts['neutral'] += 1
                sentiment_details['neutral'].extend(neu_matches)
        
        # Count word frequencies
        for sentiment in sentiment_details:
            word_counts = Counter(sentiment_details[sentiment])
            sentiment_details[sentiment] = dict(word_counts.most_common(10))
        
        return {
            'sentiment_distribution': sentiment_counts,
            'sentiment_keywords': sentiment_details
        }
    
    def plot_keyword_analysis(self, figsize: Tuple[int, int] = (16, 12)):
        """
        Plot comprehensive keyword analysis visualizations.
        
        Args:
            figsize: Figure size for the plots
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Top keywords
        keywords_df = self.extract_common_keywords(20)
        axes[0, 0].barh(range(len(keywords_df)), keywords_df['frequency'])
        axes[0, 0].set_yticks(range(len(keywords_df)))
        axes[0, 0].set_yticklabels(keywords_df['keyword'])
        axes[0, 0].set_xlabel('Frequency')
        axes[0, 0].set_title('Top 20 Keywords in Headlines')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Financial phrases
        financial_phrases = self.extract_financial_phrases()
        categories = list(financial_phrases.keys())
        mentions = [financial_phrases[cat]['total_mentions'] for cat in categories]
        
        axes[0, 1].bar(range(len(categories)), mentions)
        axes[0, 1].set_xticks(range(len(categories)))
        axes[0, 1].set_xticklabels(categories, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Total Mentions')
        axes[0, 1].set_title('Financial Phrases by Category')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bigrams
        bigrams_df = self.analyze_ngrams(2, 15)
        axes[1, 0].barh(range(len(bigrams_df)), bigrams_df['frequency'])
        axes[1, 0].set_yticks(range(len(bigrams_df)))
        axes[1, 0].set_yticklabels(bigrams_df['2gram'])
        axes[1, 0].set_xlabel('Frequency')
        axes[1, 0].set_title('Top 15 Bigrams')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sentiment analysis
        sentiment_analysis = self.sentiment_keyword_analysis()
        sentiment_dist = sentiment_analysis['sentiment_distribution']
        
        axes[1, 1].pie(sentiment_dist.values(), labels=sentiment_dist.keys(), 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Sentiment Distribution in Headlines')
        
        plt.tight_layout()
        plt.show()
    
    def get_comprehensive_text_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive text analysis results.
        
        Returns:
            Dict: Complete text analysis
        """
        return {
            'top_keywords': self.extract_common_keywords(30).to_dict('records'),
            'financial_phrases': self.extract_financial_phrases(),
            'bigrams': self.analyze_ngrams(2, 20).to_dict('records'),
            'trigrams': self.analyze_ngrams(3, 15).to_dict('records'),
            'sentiment_analysis': self.sentiment_keyword_analysis()
        } 