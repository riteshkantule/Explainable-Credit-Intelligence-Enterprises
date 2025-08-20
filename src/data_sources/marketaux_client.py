"""
MarketAux API Client for CredTech Hackathon  
Handles financial news data with sentiment analysis
"""

import requests
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class MarketAuxClient:
    """MarketAux API client for fetching financial news and sentiment data"""
    
    def __init__(self, api_token: str = None):
        self.api_token = api_token or os.getenv('MARKETAUX_API_TOKEN')
        self.base_url = "https://api.marketaux.com/v1/news"
        self.session = requests.Session()
        
        # Rate limiting for free tier
        self.last_request_time = 0
        self.min_request_interval = 1  # 1 second between requests
        
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request with rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        params['api_token'] = self.api_token
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            self.last_request_time = time.time()
            
            data = response.json()
            
            if 'error' in data:
                raise ValueError(f"API Error: {data['error']}")
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def get_news_for_symbols(self, symbols: Union[str, List[str]], 
                           days_back: int = 7, limit: int = 50,
                           language: str = 'en') -> pd.DataFrame:
        """Get financial news for specific stock symbols"""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        symbol_str = ','.join(symbols)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        params = {
            'symbols': symbol_str,
            'filter_entities': 'true',
            'language': language,
            'published_after': start_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'published_before': end_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'limit': limit
        }
        
        logger.info(f"Fetching news for {symbol_str} ({days_back} days back)")
        data = self._make_request('all', params)
        
        if 'data' not in data or not data['data']:
            logger.warning(f"No news found for {symbol_str}")
            return pd.DataFrame()
        
        articles = data['data']
        df = pd.DataFrame(articles)
        
        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
            
            # Extract entities information
            df['entity_symbols'] = df['entities'].apply(
                lambda x: [entity['symbol'] for entity in x if 'symbol' in entity] if x else []
            )
            
            # Clean sentiment data
            df['sentiment_score'] = df['sentiment'].apply(self._convert_sentiment_to_score)
            
            # Add derived features
            df['title_length'] = df['title'].str.len()
            df['description_length'] = df['description'].fillna('').str.len()
            df['has_image'] = df['image'].notna()
            
            df.sort_values('published_at', ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
        
        logger.info(f"Retrieved {len(df)} articles for {symbol_str}")
        return df
    
    def _convert_sentiment_to_score(self, sentiment: str) -> float:
        """Convert sentiment string to numerical score"""
        sentiment_mapping = {
            'positive': 1.0,
            'negative': 0.0,
            'neutral': 0.5
        }
        return sentiment_mapping.get(sentiment.lower() if sentiment else 'neutral', 0.5)
    
    def get_sentiment_summary(self, symbols: Union[str, List[str]], 
                            days_back: int = 7) -> Dict[str, Dict]:
        """Get sentiment summary for symbols"""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        sentiment_summary = {}
        
        for symbol in symbols:
            news_df = self.get_news_for_symbols(symbol, days_back=days_back)
            
            if news_df.empty:
                sentiment_summary[symbol] = {
                    'article_count': 0,
                    'avg_sentiment': 0.5,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'sentiment_trend': 'neutral'
                }
                continue
            
            sentiment_counts = news_df['sentiment'].value_counts()
            avg_sentiment = news_df['sentiment_score'].mean()
            
            # Calculate trend
            mid_point = len(news_df) // 2
            if mid_point > 0:
                recent_sentiment = news_df.iloc[:mid_point]['sentiment_score'].mean()
                older_sentiment = news_df.iloc[mid_point:]['sentiment_score'].mean()
                trend = 'improving' if recent_sentiment > older_sentiment else 'declining' if recent_sentiment < older_sentiment else 'stable'
            else:
                trend = 'neutral'
            
            sentiment_summary[symbol] = {
                'article_count': len(news_df),
                'avg_sentiment': avg_sentiment,
                'positive_count': sentiment_counts.get('positive', 0),
                'negative_count': sentiment_counts.get('negative', 0), 
                'neutral_count': sentiment_counts.get('neutral', 0),
                'sentiment_trend': trend,
                'latest_article_date': news_df['published_at'].max().isoformat() if not news_df.empty else None
            }
        
        return sentiment_summary
    
    def get_trending_news(self, limit: int = 20, language: str = 'en') -> pd.DataFrame:
        """Get trending financial news"""
        params = {
            'filter_entities': 'true',
            'language': language,
            'limit': limit,
            'sort': 'published_desc'
        }
        
        logger.info(f"Fetching {limit} trending financial news articles")
        data = self._make_request('all', params)
        
        if 'data' not in data or not data['data']:
            logger.warning("No trending news found")
            return pd.DataFrame()
        
        articles = data['data']
        df = pd.DataFrame(articles)
        
        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
            df['sentiment_score'] = df['sentiment'].apply(self._convert_sentiment_to_score)
            df.sort_values('published_at', ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
        
        return df
    
    def get_news_by_keywords(self, keywords: Union[str, List[str]], 
                           days_back: int = 7, limit: int = 50) -> pd.DataFrame:
        """Search news by keywords"""
        if isinstance(keywords, list):
            search_query = ' AND '.join(keywords)
        else:
            search_query = keywords
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        params = {
            'search': search_query,
            'filter_entities': 'true',
            'language': 'en',
            'published_after': start_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'published_before': end_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'limit': limit
        }
        
        logger.info(f"Searching news for keywords: {search_query}")
        data = self._make_request('all', params)
        
        if 'data' not in data or not data['data']:
            logger.warning(f"No news found for keywords: {search_query}")
            return pd.DataFrame()
        
        articles = data['data']
        df = pd.DataFrame(articles)
        
        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
            df['sentiment_score'] = df['sentiment'].apply(self._convert_sentiment_to_score)
            df.sort_values('published_at', ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
        
        return df
    
    def analyze_company_sentiment_trend(self, symbol: str, days_back: int = 30) -> Dict:
        """Analyze sentiment trend over time for a company"""
        news_df = self.get_news_for_symbols(symbol, days_back=days_back, limit=100)
        
        if news_df.empty:
            return {
                'symbol': symbol,
                'trend': 'no_data',
                'current_sentiment': 0.5,
                'sentiment_change': 0.0,
                'volatility': 0.0,
                'article_frequency': 0.0
            }
        
        # Group by date and calculate daily sentiment
        news_df['date'] = news_df['published_at'].dt.date
        daily_sentiment = news_df.groupby('date')['sentiment_score'].mean().reset_index()
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        daily_sentiment.sort_values('date', inplace=True)
        
        # Calculate trend metrics
        current_sentiment = daily_sentiment['sentiment_score'].iloc[-7:].mean()  # Last week average
        previous_sentiment = daily_sentiment['sentiment_score'].iloc[-14:-7].mean() if len(daily_sentiment) > 7 else 0.5
        
        sentiment_change = current_sentiment - previous_sentiment
        volatility = daily_sentiment['sentiment_score'].std()
        article_frequency = len(news_df) / days_back
        
        # Determine trend
        if sentiment_change > 0.1:
            trend = 'improving'
        elif sentiment_change < -0.1:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'symbol': symbol,
            'trend': trend,
            'current_sentiment': current_sentiment,
            'sentiment_change': sentiment_change,
            'volatility': volatility,
            'article_frequency': article_frequency,
            'total_articles': len(news_df),
            'date_range': f"{daily_sentiment['date'].min().date()} to {daily_sentiment['date'].max().date()}"
        }

def test_marketaux_client():
    """Test function for MarketAux client"""
    API_TOKEN = os.getenv('MARKETAUX_API_TOKEN')
    
    if not API_TOKEN:
        print("❌ MARKETAUX_API_TOKEN not found in environment variables")
        return
    
    client = MarketAuxClient(API_TOKEN)
    symbols = ['AAPL', 'MSFT']
    
    print(f"Testing MarketAux client with {symbols}")
    print("=" * 50)
    
    # Get news for symbols
    news_df = client.get_news_for_symbols(symbols, days_back=3, limit=10)
    if not news_df.empty:
        print(f"✅ News articles: {len(news_df)}")
        print("Latest articles:")
        for idx, row in news_df.head(3).iterrows():
            print(f"- {row['title'][:80]}...")
    
    # Get sentiment summary
    sentiment_summary = client.get_sentiment_summary(symbols, days_back=7)
    print("\nSentiment Summary:")
    for symbol, stats in sentiment_summary.items():
        print(f"{symbol}: {stats['article_count']} articles, "
              f"Sentiment: {stats['avg_sentiment']:.2f}")

if __name__ == "__main__":
    test_marketaux_client()