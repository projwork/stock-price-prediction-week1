import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import re
from collections import Counter
import warnings
import logging
from urllib.parse import urlparse

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FinancialPublisherAnalyzer:
    """Class to perform advanced publisher analysis on financial news data."""
    
    def __init__(self, data: pd.DataFrame, publisher_column: str = 'publisher', url_column: str = 'url'):
        """
        Initialize the publisher analyzer.
        
        Args:
            data (pd.DataFrame): The financial news dataset
            publisher_column (str): Column containing publisher information
            url_column (str): Column containing URL information
        """
        self.data = data.copy()
        self.publisher_column = publisher_column
        self.url_column = url_column
        self.prepare_publisher_data()
        
    def prepare_publisher_data(self):
        """Prepare data for publisher analysis."""
        try:
            # Clean publisher names
            if self.publisher_column in self.data.columns:
                self.data[self.publisher_column] = self.data[self.publisher_column].fillna('Unknown')
                self.data['publisher_clean'] = self.data[self.publisher_column].str.strip().str.lower()
            
            # Extract email domains from publisher names
            self.data['is_email'] = self.data[self.publisher_column].str.contains('@', na=False)
            self.data['email_domain'] = self.data[self.publisher_column].apply(self.extract_email_domain)
            
            # Extract domains from URLs
            if self.url_column in self.data.columns:
                self.data['url_domain'] = self.data[self.url_column].apply(self.extract_url_domain)
            
            # Classify publisher types
            self.data['publisher_type'] = self.data.apply(self.classify_publisher_type, axis=1)
            
            logger.info("Publisher data preparation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in publisher data preparation: {str(e)}")
            raise
    
    def extract_email_domain(self, publisher: str) -> str:
        """
        Extract domain from email address.
        
        Args:
            publisher (str): Publisher name/email
            
        Returns:
            str: Email domain or None
        """
        if pd.isna(publisher):
            return None
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b'
        match = re.search(email_pattern, str(publisher))
        
        return match.group(1).lower() if match else None
    
    def extract_url_domain(self, url: str) -> str:
        """
        Extract domain from URL.
        
        Args:
            url (str): URL string
            
        Returns:
            str: Domain or None
        """
        if pd.isna(url):
            return None
        
        try:
            parsed_url = urlparse(str(url))
            domain = parsed_url.netloc.lower()
            # Remove 'www.' prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return None
    
    def classify_publisher_type(self, row) -> str:
        """
        Classify publisher type based on various criteria.
        
        Args:
            row: DataFrame row
            
        Returns:
            str: Publisher type classification
        """
        publisher = str(row[self.publisher_column]).lower()
        
        # Email-based publishers
        if row['is_email']:
            return 'email_reporter'
        
        # Known financial news organizations
        financial_orgs = ['reuters', 'bloomberg', 'cnbc', 'marketwatch', 'yahoo finance', 
                         'seeking alpha', 'fool', 'benzinga', 'zacks', 'barrons',
                         'financial times', 'wsj', 'wall street journal']
        
        for org in financial_orgs:
            if org in publisher:
                return 'major_financial_news'
        
        # Analyst firms
        analyst_firms = ['analyst', 'research', 'equity', 'capital', 'securities']
        for firm in analyst_firms:
            if firm in publisher:
                return 'analyst_firm'
        
        # Individual analysts/contributors
        individual_indicators = ['@', 'contributor', 'analyst']
        for indicator in individual_indicators:
            if indicator in publisher:
                return 'individual_contributor'
        
        # General news organizations
        general_news = ['cnn', 'fox', 'nbc', 'abc', 'cbs', 'npr', 'ap news', 'associated press']
        for news in general_news:
            if news in publisher:
                return 'general_news'
        
        return 'other'
    
    def analyze_publisher_activity(self) -> Dict[str, Any]:
        """
        Analyze publisher activity patterns.
        
        Returns:
            Dict: Publisher activity analysis
        """
        activity_analysis = {}
        
        # Basic publisher statistics
        publisher_counts = self.data[self.publisher_column].value_counts()
        activity_analysis['top_publishers'] = publisher_counts.head(20).to_dict()
        activity_analysis['total_publishers'] = len(publisher_counts)
        activity_analysis['single_article_publishers'] = len(publisher_counts[publisher_counts == 1])
        
        # Publisher type distribution
        type_distribution = self.data['publisher_type'].value_counts()
        activity_analysis['publisher_type_distribution'] = type_distribution.to_dict()
        
        # Email vs non-email publishers
        email_stats = self.data['is_email'].value_counts()
        activity_analysis['email_publisher_stats'] = {
            'email_publishers': int(email_stats.get(True, 0)),
            'non_email_publishers': int(email_stats.get(False, 0)),
            'email_percentage': (email_stats.get(True, 0) / len(self.data)) * 100
        }
        
        return activity_analysis
    
    def analyze_email_domains(self) -> Dict[str, Any]:
        """
        Analyze email domains when publishers use email addresses.
        
        Returns:
            Dict: Email domain analysis
        """
        email_publishers = self.data[self.data['is_email']].copy()
        
        if len(email_publishers) == 0:
            return {'message': 'No email-based publishers found'}
        
        domain_analysis = {}
        
        # Domain frequency
        domain_counts = email_publishers['email_domain'].value_counts()
        domain_analysis['top_domains'] = domain_counts.head(15).to_dict()
        domain_analysis['total_unique_domains'] = len(domain_counts)
        
        # Classify domain types
        domain_types = email_publishers['email_domain'].apply(self.classify_email_domain)
        domain_type_counts = domain_types.value_counts()
        domain_analysis['domain_type_distribution'] = domain_type_counts.to_dict()
        
        # Organization analysis
        org_domains = email_publishers[domain_types == 'organization']['email_domain'].value_counts()
        domain_analysis['top_organization_domains'] = org_domains.head(10).to_dict()
        
        return domain_analysis
    
    def classify_email_domain(self, domain: str) -> str:
        """
        Classify email domain type.
        
        Args:
            domain (str): Email domain
            
        Returns:
            str: Domain type classification
        """
        if pd.isna(domain):
            return 'unknown'
        
        domain = domain.lower()
        
        # Personal email providers
        personal_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 
                             'aol.com', 'icloud.com', 'live.com']
        
        if domain in personal_providers:
            return 'personal'
        
        # Known news organizations
        news_domains = ['reuters.com', 'bloomberg.net', 'cnbc.com', 'marketwatch.com',
                       'wsj.com', 'ft.com', 'barrons.com', 'seekingalpha.com']
        
        if domain in news_domains:
            return 'news_organization'
        
        # Financial institutions
        financial_keywords = ['bank', 'capital', 'securities', 'invest', 'asset', 'wealth']
        if any(keyword in domain for keyword in financial_keywords):
            return 'financial_institution'
        
        # Generic organization
        return 'organization'
    
    def analyze_url_domains(self) -> Dict[str, Any]:
        """
        Analyze URL domains to understand news sources.
        
        Returns:
            Dict: URL domain analysis
        """
        if self.url_column not in self.data.columns:
            return {'message': 'URL column not found'}
        
        url_data = self.data[self.data['url_domain'].notna()].copy()
        
        if len(url_data) == 0:
            return {'message': 'No valid URLs found'}
        
        url_analysis = {}
        
        # Domain frequency
        domain_counts = url_data['url_domain'].value_counts()
        url_analysis['top_url_domains'] = domain_counts.head(20).to_dict()
        url_analysis['total_unique_url_domains'] = len(domain_counts)
        
        # Classify URL domain types
        url_data['url_domain_type'] = url_data['url_domain'].apply(self.classify_url_domain)
        domain_type_counts = url_data['url_domain_type'].value_counts()
        url_analysis['url_domain_type_distribution'] = domain_type_counts.to_dict()
        
        return url_analysis
    
    def classify_url_domain(self, domain: str) -> str:
        """
        Classify URL domain type.
        
        Args:
            domain (str): URL domain
            
        Returns:
            str: Domain type classification
        """
        if pd.isna(domain):
            return 'unknown'
        
        domain = domain.lower()
        
        # Major financial news sites
        financial_news = ['bloomberg.com', 'reuters.com', 'marketwatch.com', 'cnbc.com',
                         'yahoo.com', 'seekingalpha.com', 'fool.com', 'benzinga.com',
                         'zacks.com', 'barrons.com', 'wsj.com', 'ft.com']
        
        if any(site in domain for site in financial_news):
            return 'major_financial_news'
        
        # Social media and forums
        social_sites = ['twitter.com', 'linkedin.com', 'facebook.com', 'reddit.com']
        if any(site in domain for site in social_sites):
            return 'social_media'
        
        # Press release sites
        pr_sites = ['prnewswire.com', 'businesswire.com', 'globenewswire.com']
        if any(site in domain for site in pr_sites):
            return 'press_release'
        
        # General news
        general_news = ['cnn.com', 'fox.com', 'nbc.com', 'abc.com', 'cbs.com']
        if any(site in domain for site in general_news):
            return 'general_news'
        
        return 'other'
    
    def analyze_publisher_content_types(self) -> Dict[str, Any]:
        """
        Analyze what types of content different publishers focus on.
        
        Returns:
            Dict: Publisher content type analysis
        """
        if 'headline' not in self.data.columns:
            return {'message': 'Headline column not found for content analysis'}
        
        content_analysis = {}
        
        # Define content type keywords
        content_keywords = {
            'earnings': ['earnings', 'eps', 'profit', 'revenue', 'quarterly'],
            'analyst_ratings': ['upgrade', 'downgrade', 'rating', 'price target', 'analyst'],
            'mergers_acquisitions': ['merger', 'acquisition', 'buyout', 'takeover'],
            'regulatory': ['fda', 'sec', 'regulatory', 'approval'],
            'market_movement': ['surge', 'plunge', 'rally', 'crash', 'volatile'],
            'ipo_offerings': ['ipo', 'offering', 'public', 'debut'],
            'dividends': ['dividend', 'payout', 'yield'],
            'leadership': ['ceo', 'cfo', 'executive', 'management', 'leadership']
        }
        
        # Analyze content types by publisher
        for publisher_type in self.data['publisher_type'].unique():
            publisher_data = self.data[self.data['publisher_type'] == publisher_type]
            
            type_content = {}
            for content_type, keywords in content_keywords.items():
                matching_articles = 0
                for keyword in keywords:
                    matching_articles += publisher_data['headline'].str.contains(
                        keyword, case=False, na=False
                    ).sum()
                
                type_content[content_type] = {
                    'count': int(matching_articles),
                    'percentage': (matching_articles / len(publisher_data)) * 100 if len(publisher_data) > 0 else 0
                }
            
            content_analysis[publisher_type] = {
                'total_articles': len(publisher_data),
                'content_breakdown': type_content
            }
        
        return content_analysis
    
    def plot_publisher_analysis(self, figsize: Tuple[int, int] = (18, 14)):
        """
        Plot comprehensive publisher analysis visualizations.
        
        Args:
            figsize: Figure size for the plots
        """
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        
        # 1. Top publishers
        publisher_counts = self.data[self.publisher_column].value_counts().head(15)
        axes[0, 0].barh(range(len(publisher_counts)), publisher_counts.values)
        axes[0, 0].set_yticks(range(len(publisher_counts)))
        axes[0, 0].set_yticklabels(publisher_counts.index, fontsize=8)
        axes[0, 0].set_xlabel('Number of Articles')
        axes[0, 0].set_title('Top 15 Publishers by Article Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Publisher type distribution
        type_dist = self.data['publisher_type'].value_counts()
        axes[0, 1].pie(type_dist.values, labels=type_dist.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Publisher Type Distribution')
        
        # 3. Email vs Non-email publishers
        email_dist = self.data['is_email'].value_counts()
        email_labels = ['Non-Email', 'Email'] if False in email_dist.index else ['Email']
        axes[0, 2].pie(email_dist.values, labels=email_labels, autopct='%1.1f%%')
        axes[0, 2].set_title('Email vs Non-Email Publishers')
        
        # 4. Top email domains
        if self.data['is_email'].any():
            email_domains = self.data[self.data['is_email']]['email_domain'].value_counts().head(10)
            if len(email_domains) > 0:
                axes[1, 0].barh(range(len(email_domains)), email_domains.values)
                axes[1, 0].set_yticks(range(len(email_domains)))
                axes[1, 0].set_yticklabels(email_domains.index, fontsize=8)
                axes[1, 0].set_xlabel('Number of Articles')
                axes[1, 0].set_title('Top 10 Email Domains')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No email domains found', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Email Domains')
        else:
            axes[1, 0].text(0.5, 0.5, 'No email publishers found', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Email Domains')
        
        # 5. Top URL domains
        if self.url_column in self.data.columns and self.data['url_domain'].notna().any():
            url_domains = self.data['url_domain'].value_counts().head(10)
            axes[1, 1].barh(range(len(url_domains)), url_domains.values)
            axes[1, 1].set_yticks(range(len(url_domains)))
            axes[1, 1].set_yticklabels(url_domains.index, fontsize=8)
            axes[1, 1].set_xlabel('Number of Articles')
            axes[1, 1].set_title('Top 10 URL Domains')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No URL data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('URL Domains')
        
        # 6. Articles per publisher distribution
        articles_per_publisher = self.data[self.publisher_column].value_counts()
        axes[1, 2].hist(articles_per_publisher.values, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 2].set_xlabel('Articles per Publisher')
        axes[1, 2].set_ylabel('Number of Publishers')
        axes[1, 2].set_title('Distribution of Articles per Publisher')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7-9. Publisher type content analysis
        content_analysis = self.analyze_publisher_content_types()
        
        for idx, (pub_type, data) in enumerate(list(content_analysis.items())[:3]):
            if isinstance(data, dict) and 'content_breakdown' in data:
                content_types = list(data['content_breakdown'].keys())
                percentages = [data['content_breakdown'][ct]['percentage'] for ct in content_types]
                
                ax = axes[2, idx]
                ax.bar(range(len(content_types)), percentages)
                ax.set_xticks(range(len(content_types)))
                ax.set_xticklabels(content_types, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Percentage')
                ax.set_title(f'Content Types: {pub_type.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_comprehensive_publisher_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive publisher analysis results.
        
        Returns:
            Dict: Complete publisher analysis
        """
        return {
            'publisher_activity': self.analyze_publisher_activity(),
            'email_domain_analysis': self.analyze_email_domains(),
            'url_domain_analysis': self.analyze_url_domains(),
            'content_type_analysis': self.analyze_publisher_content_types()
        } 