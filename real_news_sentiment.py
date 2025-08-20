"""
CredTech: Complete Real-Time Intelligence Platform
FIXED VERSION - Works with .env file and handles backslash variables
Beautiful UI + All plots + Real API integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import requests
import yfinance as yf
from textblob import TextBlob
import os
import json
import time
import ast
from datetime import datetime, timedelta
from pathlib import Path
import math

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    ENV_LOADED = True
except ImportError:
    ENV_LOADED = False

# Page configuration
st.set_page_config(
    page_title="CredTech - Advanced Credit Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# RESTORED BEAUTIFUL CSS with modern design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.main-header {
    font-family: 'Inter', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    text-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.sub-header {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    text-align: center;
    color: #6c757d;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.news-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #007bff;
    margin: 1rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.rating-A, .rating-A-plus { 
    color: #28a745; 
    font-weight: bold; 
    font-size: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    background: linear-gradient(45deg, #28a745, #20c997);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.rating-A-minus { 
    color: #28a745; 
    font-weight: bold; 
    font-size: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.rating-B-plus, .rating-B { 
    color: #ffc107; 
    font-weight: bold; 
    font-size: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.rating-B-minus, .rating-C { 
    color: #fd7e14; 
    font-weight: bold; 
    font-size: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.rating-D { 
    color: #dc3545; 
    font-weight: bold; 
    font-size: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.causality-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}

.factor-impact {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
    border-left: 5px solid;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}

.positive-impact {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-left-color: #28a745;
    color: #155724;
}

.negative-impact {
    background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
    border-left-color: #dc3545;
    color: #721c24;
}

.neutral-impact {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border-left-color: #ffc107;
    color: #856404;
}

.data-source-badge {
    display: inline-block;
    background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.correlation-legend {
    font-size: 0.9rem;
    color: #6c757d;
    margin-top: 1rem;
}

.news-sentiment-positive {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #28a745;
}

.news-sentiment-negative {
    background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #dc3545;
}

.news-sentiment-neutral {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #ffc107;
}

.interactive-widget {
    background: linear-gradient(135deg, #a8e6cf 0%, #88d8a3 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
}

/* Enhanced tab styling with better spacing */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 0.8rem;
    border-radius: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin: 1rem 0;
}

.stTabs [data-baseweb="tab"] {
    height: 65px;
    padding: 0 28px;
    background: transparent;
    border-radius: 12px;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    transition: all 0.3s ease;
    font-size: 0.95rem;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(102, 126, 234, 0.1);
    transform: translateY(-2px);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
}

/* Better spacing for charts */
.chart-container {
    margin: 2rem 0;
    padding: 1.5rem;
    background: white;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

/* Sidebar enhancements */
.sidebar-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    text-align: center;
    font-weight: 600;
}

.api-key-success {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #28a745;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}

.api-key-warning {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #ffc107;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

class RealTimeDataCollector:
    """Complete real-time data collection system with ENHANCED .env file support"""
    
    def __init__(self):
        # ENHANCED: Better API key handling for .env files with backslashes
        self.alpha_vantage_key = self._get_api_key_enhanced('ALPHA_VANTAGE_API_KEY')
        self.fmp_key = self._get_api_key_enhanced('FMP_API_KEY')
        self.eodhd_key = self._get_api_key_enhanced('EODHD_API_KEY')
        self.marketaux_key = self._get_api_key_enhanced('MARKETAUX_API_TOKEN')
        self.rapidapi_key = self._get_api_key_enhanced('RAPIDAPI_KEY')
        
        # API endpoints
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.fmp_url = "https://financialmodelingprep.com/api/v3"
        self.eodhd_url = "https://eodhd.com/api"
        
        # Request tracking
        self.requests_made = 0
        self.last_alpha_vantage_time = 0
        
    def _get_api_key_enhanced(self, key_name):
        """ENHANCED: Get API key handling both regular and backslash-escaped variable names"""
        
        # Try multiple variations of the key name
        key_variations = [
            key_name,                    # Normal: ALPHA_VANTAGE_API_KEY
            key_name.replace('_', r'\_'),  # With backslashes: ALPHA\_VANTAGE\_API\_KEY
        ]
        
        for variation in key_variations:
            # Try environment variables first
            env_key = os.getenv(variation)
            if env_key and env_key.strip():
                return env_key.strip()
                
            # Try Streamlit secrets
            try:
                if hasattr(st, 'secrets') and variation in st.secrets:
                    return st.secrets[variation]
            except Exception:
                pass
        
        return None
    
    def get_api_status(self):
        """Get comprehensive API status"""
        status = {
            'alpha_vantage': bool(self.alpha_vantage_key),
            'fmp': bool(self.fmp_key),
            'eodhd': bool(self.eodhd_key),
            'marketaux': bool(self.marketaux_key),
            'rapidapi': bool(self.rapidapi_key),
            'yahoo_finance': True  # Always available
        }
        
        return status
    
    def get_yahoo_finance_news(self, symbol):
        """Get real news from Yahoo Finance"""
        try:
            st.write(f"   📰 Fetching Yahoo Finance news for {symbol}...")
            
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                st.write(f"   ⚠️ No news found for {symbol}")
                return None
            
            articles = []
            for article in news[:10]:
                try:
                    article_data = {
                        'title': article.get('title', ''),
                        'summary': article.get('summary', ''),
                        'publisher': article.get('publisher', ''),
                        'publish_time': datetime.fromtimestamp(article.get('providerPublishTime', 0)),
                        'url': article.get('link', ''),
                        'type': article.get('type', 'NEWS')
                    }
                    
                    # Calculate sentiment using TextBlob
                    text_content = f"{article_data['title']} {article_data['summary']}"
                    sentiment = TextBlob(text_content).sentiment
                    article_data['sentiment_polarity'] = sentiment.polarity
                    article_data['sentiment_subjectivity'] = sentiment.subjectivity
                    
                    articles.append(article_data)
                    
                except Exception as e:
                    continue
            
            if articles:
                avg_sentiment = np.mean([a['sentiment_polarity'] for a in articles])
                normalized_sentiment = (avg_sentiment + 1) / 2
                
                st.write(f"   ✅ Yahoo Finance: {len(articles)} articles, sentiment: {normalized_sentiment:.3f}")
                
                return {
                    'articles': articles,
                    'sentiment_score': normalized_sentiment,
                    'sentiment_raw': avg_sentiment,
                    'article_count': len(articles),
                    'sentiment_source': 'yahoo_finance_textblob',
                    'data_quality': 'high'
                }
                
        except Exception as e:
            st.write(f"   ⚠️ Yahoo Finance news error: {e}")
            return None
    
    def get_fmp_news(self, symbol):
        """Get news from Financial Modeling Prep"""
        if not self.fmp_key:
            st.write(f"   ⚠️ FMP API key not available")
            return None
            
        try:
            st.write(f"   📰 Fetching FMP news for {symbol}...")
            
            url = f"{self.fmp_url}/stock_news?tickers={symbol}&limit=20&apikey={self.fmp_key}"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and isinstance(data, list):
                    articles = []
                    sentiments = []
                    
                    for article in data[:10]:
                        try:
                            title = article.get('title', '')
                            text = article.get('text', article.get('summary', ''))
                            
                            content = f"{title} {text}"
                            sentiment = TextBlob(content).sentiment
                            sentiment_score = (sentiment.polarity + 1) / 2
                            
                            articles.append({
                                'title': title,
                                'summary': text[:200] + '...' if len(text) > 200 else text,
                                'url': article.get('url', ''),
                                'published_date': article.get('publishedDate', ''),
                                'sentiment_polarity': sentiment.polarity,
                                'sentiment_score': sentiment_score
                            })
                            
                            sentiments.append(sentiment_score)
                            
                        except Exception as e:
                            continue
                    
                    if sentiments:
                        avg_sentiment = np.mean(sentiments)
                        st.write(f"   ✅ FMP: {len(articles)} articles, sentiment: {avg_sentiment:.3f}")
                        
                        return {
                            'articles': articles,
                            'sentiment_score': avg_sentiment,
                            'article_count': len(articles),
                            'sentiment_source': 'fmp_textblob',
                            'data_quality': 'high'
                        }
            elif response.status_code == 429:
                st.write(f"   ⚠️ FMP: Rate limit exceeded")
            elif response.status_code == 401:
                st.write(f"   ⚠️ FMP: Invalid API key")
                        
        except Exception as e:
            st.write(f"   ⚠️ FMP news error: {e}")
        
        return None
    
    def get_eodhd_news_sentiment(self, symbol):
        """Get news with built-in sentiment from EODHD"""
        if not self.eodhd_key:
            st.write(f"   ⚠️ EODHD API key not available")
            return None
            
        try:
            st.write(f"   📰 Fetching EODHD news & sentiment for {symbol}...")
            
            news_url = f"{self.eodhd_url}/news?s={symbol}.US&limit=20&api_token={self.eodhd_key}"
            sentiment_url = f"{self.eodhd_url}/sentiments?s={symbol}.US&api_token={self.eodhd_key}"
            
            # Get news
            news_response = requests.get(news_url, timeout=15)
            
            if news_response.status_code == 200:
                news_data = news_response.json()
            elif news_response.status_code == 401:
                st.write(f"   ⚠️ EODHD: Invalid API key")
                return None
            elif news_response.status_code == 429:
                st.write(f"   ⚠️ EODHD: Rate limit exceeded")
                return None
            else:
                news_data = None
            
            time.sleep(0.5)
            
            # Get sentiment
            sentiment_response = requests.get(sentiment_url, timeout=15)
            sentiment_data = sentiment_response.json() if sentiment_response.status_code == 200 else None
            
            articles = []
            if news_data and isinstance(news_data, list):
                for article in news_data[:10]:
                    articles.append({
                        'title': article.get('title', ''),
                        'content': article.get('content', ''),
                        'url': article.get('link', ''),
                        'date': article.get('date', ''),
                        'symbols': article.get('symbols', [])
                    })
            
            # Process sentiment data
            sentiment_score = 0.5
            if sentiment_data and f"{symbol}.US" in sentiment_data:
                sentiment_entries = sentiment_data[f"{symbol}.US"]
                if sentiment_entries:
                    latest_sentiment = sentiment_entries[0]['normalized']
                    sentiment_score = (latest_sentiment + 1) / 2
            
            st.write(f"   ✅ EODHD: {len(articles)} articles, sentiment: {sentiment_score:.3f}")
            
            return {
                'articles': articles,
                'sentiment_score': sentiment_score,
                'article_count': len(articles),
                'sentiment_source': 'eodhd_native',
                'data_quality': 'professional'
            }
                
        except Exception as e:
            st.write(f"   ⚠️ EODHD error: {e}")
        
        return None
    
    def get_aggregated_news_sentiment(self, symbol):
        """Aggregate sentiment from multiple news sources"""
        st.write(f"   📊 Aggregating news sentiment from multiple sources...")
        
        all_sources = []
        
        # 1. Yahoo Finance News (always try first)
        yahoo_result = self.get_yahoo_finance_news(symbol)
        if yahoo_result:
            all_sources.append(yahoo_result)
        
        # 2. Financial Modeling Prep (if API key available)
        fmp_result = self.get_fmp_news(symbol)
        if fmp_result:
            all_sources.append(fmp_result)
        
        # 3. EODHD (if API key available)
        eodhd_result = self.get_eodhd_news_sentiment(symbol)
        if eodhd_result:
            all_sources.append(eodhd_result)
        
        # Aggregate results
        if all_sources:
            weights = []
            sentiments = []
            total_articles = 0
            sources_used = []
            
            for source in all_sources:
                weight = source['article_count'] * (1.2 if source['data_quality'] == 'professional' else 1.0)
                weights.append(weight)
                sentiments.append(source['sentiment_score'])
                total_articles += source['article_count']
                sources_used.append(source['sentiment_source'])
            
            weighted_sentiment = np.average(sentiments, weights=weights) if weights else np.mean(sentiments)
            
            # Combine articles
            all_articles = []
            for source in all_sources:
                all_articles.extend(source.get('articles', []))
            
            result = {
                'sentiment_score': weighted_sentiment,
                'article_count': total_articles,
                'sentiment_sources': sources_used,
                'all_articles': all_articles[:20],
                'data_quality': 'aggregated_multi_source',
                'source_breakdown': {
                    source['sentiment_source']: {
                        'sentiment': source['sentiment_score'],
                        'articles': source['article_count']
                    } for source in all_sources
                }
            }
            
            st.write(f"   ✅ Aggregated sentiment: {weighted_sentiment:.3f} from {len(all_sources)} sources")
            return result
        
        # Fallback
        st.write(f"   ⚠️ No news sources available, using neutral sentiment")
        return {
            'sentiment_score': 0.5,
            'article_count': 0,
            'sentiment_sources': ['fallback'],
            'data_quality': 'fallback'
        }
    
    def get_stock_price_yahoo(self, symbol):
        """Get stock price from Yahoo Finance"""
        try:
            st.write(f"   📈 Fetching stock price from Yahoo Finance...")
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")
            
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                price_change = current_price - previous_close
                
                returns = hist['Close'].pct_change().dropna()
                volatility = float(returns.std()) if len(returns) > 1 else 0.02
                
                result = {
                    'latest_price': current_price,
                    'previous_close': previous_close,
                    'price_change': price_change,
                    'price_change_pct': (price_change / previous_close * 100) if previous_close > 0 else 0,
                    'volatility': volatility,
                    'volume': float(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                    'week_52_high': info.get('fiftyTwoWeekHigh', current_price * 1.2),
                    'week_52_low': info.get('fiftyTwoWeekLow', current_price * 0.8),
                    'source': 'yahoo_finance'
                }
                
                st.write(f"   ✅ Yahoo Finance: ${current_price:.2f} ({price_change:+.2f})")
                return result
                
        except Exception as e:
            st.write(f"   ❌ Yahoo Finance price error: {e}")
        
        return None
    
    def get_company_overview_alpha_vantage(self, symbol):
        """Get company overview from Alpha Vantage"""
        if not self.alpha_vantage_key:
            st.write(f"   ⚠️ Alpha Vantage API key not found")
            return None
            
        st.write(f"   📊 Fetching company overview from Alpha Vantage...")
        
        # Rate limiting for Alpha Vantage
        current_time = time.time()
        if current_time - self.last_alpha_vantage_time < 15:
            sleep_time = 15 - (current_time - self.last_alpha_vantage_time)
            st.write(f"   ⏳ Alpha Vantage rate limit: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(self.alpha_vantage_url, params=params, timeout=30)
            self.last_alpha_vantage_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Symbol' in data:
                    overview = {
                        'company_name': data.get('Name', symbol),
                        'sector': data.get('Sector', 'Unknown'),
                        'industry': data.get('Industry', 'Unknown'),
                        'description': data.get('Description', '')[:300] + '...',
                        'market_cap': self._safe_float(data.get('MarketCapitalization')),
                        'pe_ratio': self._safe_float(data.get('PERatio')),
                        'beta': self._safe_float(data.get('Beta'), 1.0),
                        'dividend_yield': self._safe_float(data.get('DividendYield')),
                        'profit_margin': self._safe_float(data.get('ProfitMargin')),
                        'roa': self._safe_float(data.get('ReturnOnAssetsTTM')),
                        'roe': self._safe_float(data.get('ReturnOnEquityTTM')),
                        'revenue_ttm': self._safe_float(data.get('RevenueTTM')),
                        'eps': self._safe_float(data.get('EPS')),
                        'source': 'alpha_vantage'
                    }
                    
                    st.write(f"   ✅ Alpha Vantage overview: {overview['company_name']}")
                    return overview
                elif 'Note' in data:
                    st.write(f"   ⚠️ Alpha Vantage: Rate limit hit - {data['Note']}")
                elif 'Error Message' in data:
                    st.write(f"   ⚠️ Alpha Vantage: {data['Error Message']}")
                else:
                    st.write(f"   ⚠️ Alpha Vantage: Invalid response format")
                    
        except Exception as e:
            st.write(f"   ❌ Alpha Vantage overview error: {e}")
        
        return None
    
    def _safe_float(self, value, default=0.0):
        """Safely convert to float"""
        if value in [None, 'None', '-', 'N/A', '', 'null']:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def collect_company_data(self, symbol, default_name):
        """Collect comprehensive real-time data"""
        st.write(f"\n🏢 **COLLECTING REAL-TIME DATA: {symbol}**")
        
        company_data = {
            'symbol': symbol,
            'company_name': default_name,
            'collection_date': datetime.now().isoformat(),
            'data_sources_used': []
        }
        
        # 1. Stock Price Data (Yahoo Finance)
        stock_data = self.get_stock_price_yahoo(symbol)
        if stock_data:
            company_data.update(stock_data)
            company_data['data_sources_used'].append('yahoo_finance_price')
        else:
            # Fallback defaults
            company_data.update({
                'latest_price': 150.0,
                'volatility': 0.025,
                'price_change': 0.0,
                'price_change_pct': 0.0,
                'source': 'default'
            })
        
        # 2. Company Overview (Alpha Vantage)
        overview = self.get_company_overview_alpha_vantage(symbol)
        if overview:
            company_data.update(overview)
            company_data['data_sources_used'].append('alpha_vantage_overview')
        else:
            # Fallback defaults
            company_data.update({
                'company_name': default_name,
                'sector': 'Technology',
                'market_cap': 2e12,
                'pe_ratio': 25.0,
                'profit_margin': 0.20,
                'roa': 0.15
            })
        
        # 3. Real News Sentiment
        news_sentiment = self.get_aggregated_news_sentiment(symbol)
        if news_sentiment:
            company_data.update({
                'news_sentiment_score': news_sentiment['sentiment_score'],
                'news_article_count': news_sentiment['article_count'],
                'news_sentiment_sources': news_sentiment['sentiment_sources'],
                'news_articles': news_sentiment.get('all_articles', [])[:10],
                'sentiment_breakdown': news_sentiment.get('source_breakdown', {}),
                'news_data_quality': news_sentiment['data_quality']
            })
            company_data['data_sources_used'].extend(news_sentiment['sentiment_sources'])
        else:
            company_data.update({
                'news_sentiment_score': 0.5,
                'news_article_count': 0,
                'news_sentiment_sources': ['fallback'],
                'news_data_quality': 'fallback'
            })
        
        return company_data

def calculate_enhanced_credit_score_v2(data):
    """Enhanced credit scoring with real news sentiment"""
    try:
        # 1. Financial Strength (35% weight)
        roa = data.get('roa', 0.05)
        profit_margin = data.get('profit_margin', 0.05)
        
        roa_score = max(0, min(1, (roa + 0.1) / 0.3))
        margin_score = max(0, min(1, (profit_margin + 0.1) / 0.4))
        financial_strength = (roa_score * 0.6 + margin_score * 0.4)
        
        # 2. Market Performance (25% weight)
        price_change_pct = data.get('price_change_pct', 0)
        volatility = data.get('volatility', 0.025)
        beta = data.get('beta', 1.0)
        
        performance_score = 0.5 + (price_change_pct / 20)
        performance_score = max(0, min(1, performance_score))
        
        stability_score = max(0, min(1, 1 - volatility * 20))
        beta_score = max(0, min(1, 2 - abs(beta)))
        
        market_performance = (performance_score * 0.5 + stability_score * 0.3 + beta_score * 0.2)
        
        # 3. Real News Sentiment (25% weight)
        news_sentiment = data.get('news_sentiment_score', 0.5)
        article_count = data.get('news_article_count', 0)
        
        news_quality = data.get('news_data_quality', 'fallback')
        quality_multiplier = {
            'professional': 1.2,
            'aggregated_multi_source': 1.1,
            'high': 1.0,
            'fallback': 0.8
        }.get(news_quality, 1.0)
        
        volume_weight = min(1, article_count / 15)
        sentiment_weighted = (news_sentiment * volume_weight + 0.5 * (1 - volume_weight)) * quality_multiplier
        sentiment_weighted = max(0, min(1, sentiment_weighted))
        
        # 4. Valuation Health (15% weight)
        pe_ratio = data.get('pe_ratio', 20)
        pe_health = max(0, min(1, 1 - abs(pe_ratio - 18) / 30)) if pe_ratio > 0 else 0.3
        
        # Composite Score
        credit_score = (
            financial_strength * 0.35 +
            market_performance * 0.25 +
            sentiment_weighted * 0.25 +
            pe_health * 0.15
        )
        
        credit_score = max(0, min(1, credit_score))
        
        # Rating Scale
        if credit_score >= 0.90:
            rating = 'A+'
        elif credit_score >= 0.80:
            rating = 'A'
        elif credit_score >= 0.70:
            rating = 'A-'
        elif credit_score >= 0.65:
            rating = 'B+'
        elif credit_score >= 0.55:
            rating = 'B'
        elif credit_score >= 0.45:
            rating = 'B-'
        elif credit_score >= 0.35:
            rating = 'C'
        else:
            rating = 'D'
        
        # Add calculated features
        data.update({
            'financial_strength': round(financial_strength, 4),
            'market_performance': round(market_performance, 4),
            'sentiment_weighted': round(sentiment_weighted, 4),
            'pe_health': round(pe_health, 4),
            'credit_score_raw': round(credit_score, 4),
            'credit_rating': rating,
            'scoring_methodology': 'real_news_sentiment_v2'
        })
        
        return data
        
    except Exception as e:
        st.error(f"Credit scoring error: {e}")
        # Fallback
        data.update({
            'financial_strength': 0.5,
            'market_performance': 0.5,
            'sentiment_weighted': 0.5,
            'pe_health': 0.5,
            'credit_score_raw': 0.5,
            'credit_rating': 'C',
            'scoring_methodology': 'fallback'
        })
        return data

# Data collection and caching
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_cached_data():
    """Get cached data or return None if cache is empty"""
    return None

def collect_fresh_data():
    """Collect fresh data from APIs"""
    
    # Clear any existing cache
    st.cache_data.clear()
    
    collector = RealTimeDataCollector()
    
    # Target companies
    companies = [
        ('AAPL', 'Apple Inc'),
        ('MSFT', 'Microsoft Corporation'),
        ('GOOGL', 'Alphabet Inc')
    ]
    
    all_data = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (symbol, name) in enumerate(companies):
        try:
            status_text.text(f'Collecting data for {symbol} ({i+1}/{len(companies)})...')
            progress_bar.progress((i) / len(companies))
            
            # Collect data
            company_data = collector.collect_company_data(symbol, name)
            
            # Calculate credit score
            company_data = calculate_enhanced_credit_score_v2(company_data)
            
            all_data.append(company_data)
            
            st.success(f"✅ {symbol}: {company_data['credit_rating']} rating ({company_data['credit_score_raw']:.3f})")
            
            # Brief pause between companies (except for last one)
            if i < len(companies) - 1:
                time.sleep(2)
                
        except Exception as e:
            st.error(f"❌ Error processing {symbol}: {e}")
            
        progress_bar.progress((i + 1) / len(companies))
    
    progress_bar.progress(1.0)
    status_text.text('✅ Data collection completed!')
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    if not df.empty:
        # Display summary
        st.success(f"🎉 Successfully collected data for {len(df)} companies!")
        
        # Show rankings
        df_sorted = df.sort_values('credit_score_raw', ascending=False)
        st.write("**🏆 Fresh Rankings:**")
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            st.write(f"   {i+1}. {row['symbol']} - {row['credit_rating']} ({row['credit_score_raw']:.3f})")
    
    time.sleep(2)  # Brief pause before clearing status
    progress_bar.empty()
    status_text.empty()
    
    return df

def create_sample_data():
    """Fallback sample data"""
    sample_companies = [
        {
            'symbol': 'AAPL',
            'company_name': 'Apple Inc',
            'latest_price': 229.75,
            'price_change_pct': -0.49,
            'market_cap': 3436888195000,
            'pe_ratio': 35.1,
            'beta': 1.29,
            'profit_margin': 0.256,
            'roa': 0.184,
            'volatility': 0.0234,
            'news_sentiment_score': 0.702,
            'news_article_count': 20,
            'news_sentiment_sources': ['yahoo_finance_textblob'],
            'financial_strength': 0.763,
            'market_performance': 0.718,
            'sentiment_weighted': 0.702,
            'pe_health': 0.481,
            'credit_score_raw': 0.762,
            'credit_rating': 'A-',
            'news_articles': []
        },
        {
            'symbol': 'MSFT',
            'company_name': 'Microsoft Corporation',
            'latest_price': 509.51,
            'price_change_pct': -1.47,
            'market_cap': 3866511802000,
            'pe_ratio': 38.2,
            'beta': 0.91,
            'profit_margin': 0.362,
            'roa': 0.186,
            'volatility': 0.0189,
            'news_sentiment_score': 0.704,
            'news_article_count': 20,
            'news_sentiment_sources': ['yahoo_finance_textblob'],
            'financial_strength': 0.799,
            'market_performance': 0.664,
            'sentiment_weighted': 0.704,
            'pe_health': 0.333,
            'credit_score_raw': 0.718,
            'credit_rating': 'A-',
            'news_articles': []
        },
        {
            'symbol': 'GOOGL',
            'company_name': 'Alphabet Inc',
            'latest_price': 201.32,
            'price_change_pct': -1.07,
            'market_cap': 2471451427000,
            'pe_ratio': 21.7,
            'beta': 1.05,
            'profit_margin': 0.274,
            'roa': 0.137,
            'volatility': 0.0267,
            'news_sentiment_score': 0.730,
            'news_article_count': 20,
            'news_sentiment_sources': ['yahoo_finance_textblob'],
            'financial_strength': 0.741,
            'market_performance': 0.712,
            'sentiment_weighted': 0.730,
            'pe_health': 0.780,
            'credit_score_raw': 0.829,
            'credit_rating': 'A',
            'news_articles': []
        }
    ]
    
    return pd.DataFrame(sample_companies)

# RESTORED ALL CHART FUNCTIONS

def safe_eval_list(value):
    """Safely convert string representation of list to actual list"""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            if ',' in value:
                return [item.strip().strip("'\"") for item in value.split(',')]
            else:
                return [value.strip("[]'\"")]
    elif isinstance(value, list):
        return value
    else:
        return [str(value)] if value else []

def get_rating_color_class(rating):
    """Get CSS class for rating color"""
    rating_clean = rating.replace('+', '-plus').replace('-', '-minus')
    return f"rating-{rating_clean}"

def create_waterfall_chart(company_data):
    """Create waterfall chart showing credit score buildup"""
    symbol = company_data['symbol']
    
    # Get component scores
    financial = company_data.get('financial_strength', 0.5) * 0.35
    market = company_data.get('market_performance', 0.5) * 0.25
    sentiment = company_data.get('sentiment_weighted', 0.5) * 0.25
    valuation = company_data.get('pe_health', 0.5) * 0.15
    
    # Create waterfall data
    categories = ['Start', 'Financial<br>Strength<br>(35%)', 'Market<br>Performance<br>(25%)', 
                 'News<br>Sentiment<br>(25%)', 'Valuation<br>Health<br>(15%)', 'Final Score']
    
    values = [0, financial, market, sentiment, valuation, 
              financial + market + sentiment + valuation]
    
    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Credit Score Components",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        x=categories,
        textposition="outside",
        text=[f"{v:.3f}" for v in values],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "rgba(40, 167, 69, 0.8)"}},
        decreasing={"marker": {"color": "rgba(220, 53, 69, 0.8)"}},
        totals={"marker": {"color": "rgba(102, 126, 234, 0.8)"}}
    ))
    
    fig.update_layout(
        title=f"Credit Score Waterfall Analysis - {symbol}",
        height=550,
        font=dict(family="Inter, sans-serif", size=12),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', range=[0, 1])
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap between factors with better layout"""
    
    # Select relevant columns for correlation
    corr_columns = ['financial_strength', 'market_performance', 'sentiment_weighted', 
                   'pe_health', 'credit_score_raw', 'latest_price', 'market_cap']
    
    # Filter available columns
    available_columns = [col for col in corr_columns if col in df.columns]
    
    if len(available_columns) < 3:
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[available_columns].corr()
    
    # Create heatmap with better formatting
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=[col.replace('_', '<br>').title() for col in corr_matrix.columns],
        y=[col.replace('_', '<br>').title() for col in corr_matrix.index],
        annotation_text=corr_matrix.round(2).values,
        showscale=True,
        colorscale='RdYlGn',
        font_colors=['white', 'black']
    )
    
    fig.update_layout(
        title="Factor Correlation Matrix - Understanding Relationships",
        height=650,
        width=800,
        font=dict(family="Inter, sans-serif", size=11),
        margin=dict(l=120, r=50, t=80, b=100)
    )
    
    return fig

def create_radar_chart(company_data):
    """Create radar chart for company factors"""
    symbol = company_data['symbol']
    
    # Get normalized factors (0-1 scale)
    factors = {
        'Financial<br>Strength': company_data.get('financial_strength', 0.5),
        'Market<br>Performance': company_data.get('market_performance', 0.5),
        'News<br>Sentiment': company_data.get('sentiment_weighted', 0.5),
        'Valuation<br>Health': company_data.get('pe_health', 0.5),
        'Profitability': min(1.0, company_data.get('roa', 0.05) * 10),  # Scale and cap ROA
        'Market<br>Stability': max(0, 1 - min(1, company_data.get('volatility', 0.02) * 20))  # Inverse volatility
    }
    
    categories = list(factors.keys())
    values = list(factors.values())
    
    # Add first value to end to close the radar chart
    values += [values[0]]
    categories += [categories[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=symbol,
        line=dict(color='rgba(102, 126, 234, 0.8)', width=3),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=True,
                ticks="outside",
                tickmode='linear',
                dtick=0.2
            )),
        showlegend=False,
        title=f"Multi-Factor Analysis - {symbol}",
        height=550,
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_news_sentiment_timeline(company_data):
    """Create news sentiment impact visualization"""
    symbol = company_data['symbol']
    
    # Get news data
    news_sentiment = company_data.get('news_sentiment_score', 0.5)
    article_count = company_data.get('news_article_count', 0)
    
    # Create impact visualization
    fig = go.Figure()
    
    # Sentiment impact bar
    fig.add_trace(go.Bar(
        name='News Sentiment Impact',
        x=[f'Current Sentiment<br>({article_count} articles)'],
        y=[news_sentiment],
        marker=dict(
            color=f'rgba({int(255*(1-news_sentiment))}, {int(255*news_sentiment)}, 100, 0.8)',
            line=dict(color='rgba(0,0,0,0.8)', width=2)
        ),
        text=[f'{news_sentiment:.3f}'],
        textposition='auto',
        width=[0.6]
    ))
    
    # Add neutral line
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                  annotation_text="Neutral Sentiment (0.5)")
    
    fig.update_layout(
        title=f"News Sentiment Analysis - {symbol}",
        yaxis_title="Sentiment Score (0-1)",
        height=450,
        showlegend=False,
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_multi_company_comparison(df):
    """Create comprehensive multi-company comparison with PERFECT spacing to avoid legend overlap"""
    
    # Chart 1: Credit Scores vs Market Cap
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=df['market_cap'] / 1e12,
            y=df['credit_score_raw'],
            mode='markers+text',
            text=df['symbol'],
            textposition="top center",
            marker=dict(
                size=np.sqrt(df['latest_price']) * 3,
                color=df['news_sentiment_score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="News Sentiment", len=0.5)
            ),
            name="Companies",
            hovertemplate='<b>%{text}</b><br>Credit Score: %{y:.3f}<br>Market Cap: $%{x:.2f}T<extra></extra>'
        )
    )
    fig1.update_layout(
        title="Credit Scores vs Market Cap",
        xaxis_title="Market Cap (Trillions $)",
        yaxis_title="Credit Score",
        height=400,
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=60, r=100, t=60, b=60)
    )
    
    # Chart 2: Risk-Return Profile  
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=df['volatility'] * 100,
            y=df.get('price_change_pct', [0] * len(df)),
            mode='markers+text',
            text=df['symbol'],
            textposition="top center",
            marker=dict(
                size=20,
                color=df['credit_score_raw'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Credit Score", len=0.5)
            ),
            name="Risk-Return",
            hovertemplate='<b>%{text}</b><br>Volatility: %{x:.2f}%<br>Price Change: %{y:.2f}%<extra></extra>'
        )
    )
    fig2.update_layout(
        title="Risk-Return Profile",
        xaxis_title="Volatility (%)",
        yaxis_title="Price Change (%)",
        height=400,
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=60, r=100, t=60, b=60)
    )
    
    # Chart 3: News Sentiment Impact
    fig3 = go.Figure()
    fig3.add_trace(
        go.Bar(
            x=df['symbol'],
            y=df['news_sentiment_score'],
            marker=dict(
                color=df['news_sentiment_score'],
                colorscale='RdYlGn',
                showscale=False
            ),
            text=[f"{count} articles" for count in df['news_article_count']],
            textposition='auto',
            name="Sentiment Score",
            hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.3f}<extra></extra>'
        )
    )
    fig3.update_layout(
        title="News Sentiment Impact",
        xaxis_title="Company",
        yaxis_title="Sentiment Score",
        height=400,
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=60, r=60, t=60, b=60)
    )
    
    # Chart 4: Financial Performance
    fig4 = go.Figure()
    fig4.add_trace(
        go.Bar(
            x=df['symbol'],
            y=df['roa'] * 100,
            name="ROA %",
            marker_color='rgba(102, 126, 234, 0.8)',
            offsetgroup=1,
            hovertemplate='<b>%{x}</b><br>ROA: %{y:.1f}%<extra></extra>'
        )
    )
    fig4.add_trace(
        go.Bar(
            x=df['symbol'],
            y=df['profit_margin'] * 100,
            name="Profit Margin %",
            marker_color='rgba(255, 127, 14, 0.8)',
            offsetgroup=2,
            hovertemplate='<b>%{x}</b><br>Profit Margin: %{y:.1f}%<extra></extra>'
        )
    )
    fig4.update_layout(
        title="Financial Performance",
        xaxis_title="Company",
        yaxis_title="Percentage (%)",
        height=400,
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig1, fig2, fig3, fig4

def create_causality_explorer(company_data):
    """Interactive causality explorer"""
    symbol = company_data['symbol']
    
    # Create causality chain
    causality_chain = {
        'Financial Performance': {
            'factors': ['ROA', 'Profit Margin', 'Revenue Growth'],
            'impact': company_data.get('financial_strength', 0.5),
            'weight': 0.35,
            'description': 'Core profitability and operational efficiency metrics'
        },
        'Market Dynamics': {
            'factors': ['Stock Performance', 'Volatility', 'Beta'],
            'impact': company_data.get('market_performance', 0.5),
            'weight': 0.25,
            'description': 'Market perception and trading characteristics'
        },
        'News & Sentiment': {
            'factors': ['Media Coverage', 'Sentiment Analysis', 'News Volume'],
            'impact': company_data.get('sentiment_weighted', 0.5),
            'weight': 0.25,
            'description': 'Real-time news sentiment and market perception'
        },
        'Valuation Metrics': {
            'factors': ['P/E Ratio', 'Price-to-Book', 'Market Valuation'],
            'impact': company_data.get('pe_health', 0.5),
            'weight': 0.15,
            'description': 'Valuation health and pricing efficiency'
        }
    }
    
    return causality_chain

def display_news_headlines(company_data):
    """Display real news headlines with sentiment"""
    articles = company_data.get('news_articles', [])
    
    # Handle string representation of articles list
    if isinstance(articles, str):
        try:
            articles = ast.literal_eval(articles)
        except (ValueError, SyntaxError):
            articles = []
    
    if not articles or len(articles) == 0:
        st.info("📰 No recent news articles available (click Refresh Data to get live news)")
        return
    
    st.subheader("📰 Recent News Headlines")
    
    for i, article in enumerate(articles[:5]):  # Show top 5
        if isinstance(article, dict):
            title = article.get('title', 'No title available')
            sentiment = article.get('sentiment_polarity', 0)
            url = article.get('url', '#')
            
            # Determine sentiment class
            if sentiment > 0.1:
                sentiment_class = "news-sentiment-positive"
                sentiment_icon = "😊"
                sentiment_text = f"Positive ({sentiment:.2f})"
            elif sentiment < -0.1:
                sentiment_class = "news-sentiment-negative"
                sentiment_icon = "😟"
                sentiment_text = f"Negative ({sentiment:.2f})"
            else:
                sentiment_class = "news-sentiment-neutral"
                sentiment_icon = "😐"
                sentiment_text = f"Neutral ({sentiment:.2f})"
            
            st.markdown(f"""
            <div class="{sentiment_class}">
                <h5>{sentiment_icon} {title}</h5>
                <p><strong>Sentiment:</strong> {sentiment_text}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"📰 Article {i+1}: {str(article)[:100]}...")

def main():
    """Main dashboard application - ENHANCED with .env support"""
    
    # Initialize the data collector to check API status
    collector = RealTimeDataCollector()
    api_status = collector.get_api_status()
    
    # Header
    st.markdown('<h1 class="main-header">🏦 CredTech Advanced Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-Time Credit Intelligence with Causality Analysis & News Sentiment</p>', unsafe_allow_html=True)
    
    # Initialize session state for data
    if 'company_data' not in st.session_state:
        st.session_state.company_data = None
        st.session_state.last_refresh = None
    
    # Sidebar with Company Selector
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🎛️ Dashboard Controls</div>', unsafe_allow_html=True)
        
        # Company selector
        if st.session_state.company_data is not None:
            st.subheader("🏢 Select Company")
            company_options = st.session_state.company_data['symbol'].tolist()
            selected_company = st.selectbox("Choose company for detailed analysis:", company_options)
        else:
            selected_company = None
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.info("🚀 Starting real-time data collection...")
            
            with st.spinner("Collecting live data from APIs..."):
                fresh_data = collect_fresh_data()
                
                if fresh_data is not None and not fresh_data.empty:
                    st.session_state.company_data = fresh_data
                    st.session_state.last_refresh = datetime.now()
                    st.success("✅ Fresh data collected successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to collect fresh data. Using sample data.")
                    st.session_state.company_data = create_sample_data()
        
        st.markdown("---")
        
        # ENHANCED API Status with detailed feedback
        st.subheader("🔑 API Status")
        
        # Show environment loading status
        if ENV_LOADED:
            st.markdown('<div class="api-key-success">✅ .env file loaded successfully!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="api-key-warning">⚠️ python-dotenv not installed. Install: pip install python-dotenv</div>', unsafe_allow_html=True)
        
        # Individual API status
        for api_name, status in api_status.items():
            api_display = api_name.replace('_', ' ').title()
            status_icon = "✅" if status else "❌"
            st.write(f"• {api_display}: {status_icon}")
        
        # Show specific API keys found
        if api_status['alpha_vantage']:
            st.markdown('<div class="api-key-success">🎯 Alpha Vantage: Ready for financial data</div>', unsafe_allow_html=True)
        
        if api_status['eodhd']:
            st.markdown('<div class="api-key-success">📰 EODHD: Professional sentiment available</div>', unsafe_allow_html=True)
            
        if api_status['fmp']:
            st.markdown('<div class="api-key-success">📊 FMP: Enhanced news feeds ready</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model info
        st.subheader("🤖 Enhanced Model v2.0")
        
        # Dynamic status based on available APIs
        available_sources = []
        if api_status['yahoo_finance']:
            available_sources.append("Yahoo Finance (Stock Data & News)")
        if api_status['alpha_vantage']:
            available_sources.append("Alpha Vantage (Fundamentals)")
        if api_status['eodhd']:
            available_sources.append("EODHD (Professional Sentiment)")
        if api_status['fmp']:
            available_sources.append("FMP (Enhanced News)")
        
        st.info(f"""
        **Active Data Sources:**
        {chr(10).join(f'• {source} ✅' for source in available_sources)}
        
        **AI Features:**
        • TextBlob NLP sentiment analysis
        • Multi-source data fusion
        • Real-time credit scoring
        • Causality analysis
        
        **Weighting:**
        • Financial Strength: 35%
        • Market Performance: 25% 
        • News Sentiment: 25%
        • Valuation Health: 15%
        """)
    
    # Load sample data button for first time
    if st.session_state.company_data is None:
        st.markdown("""
        ### 🎯 Welcome to CredTech Real-Time Intelligence Platform
        
        **🔋 Your API Configuration Status:**
        """)
        
        # Show current configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Working APIs:**")
            for api_name, status in api_status.items():
                if status:
                    api_display = api_name.replace('_', ' ').title()
                    st.write(f"• {api_display}")
        
        with col2:
            st.markdown("**❌ Missing APIs:**")
            for api_name, status in api_status.items():
                if not status and api_name != 'yahoo_finance':
                    api_display = api_name.replace('_', ' ').title()
                    st.write(f"• {api_display}")
        
        # Show env file status
        if ENV_LOADED:
            st.success("✅ Your .env file is loaded! The app detected your API keys.")
        else:
            st.info("💡 Install python-dotenv to auto-load .env files: `pip install python-dotenv`")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 **START WITH FRESH DATA**", key="start_fresh"):
                st.info("Click 'Refresh Data' in the sidebar to begin!")
        
        with col2:
            if st.button("📋 **TRY WITH SAMPLE DATA**", key="try_sample"):
                st.session_state.company_data = create_sample_data()
                st.session_state.last_refresh = datetime.now()
                st.success("📊 Sample data loaded!")
                st.rerun()
        
        return
    
    # Load data
    df = st.session_state.company_data
    
    # Main Navigation Tabs (RESTORED BEAUTIFUL HORIZONTAL TABS)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🔍 Causality", 
        "📰 News Impact", 
        "🔗 Correlations",
        "⚖️ Compare"
    ])
    
    with tab1:
        st.header("📈 Comprehensive Portfolio Overview")
        
        # Key metrics row with better spacing
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_companies = len(df)
            st.metric("🏢 Companies", total_companies)
        
        with col2:
            avg_score = df['credit_score_raw'].mean()
            st.metric("⭐ Avg Credit Score", f"{avg_score:.3f}")
        
        with col3:
            avg_sentiment = df['news_sentiment_score'].mean()
            st.metric("📰 Avg News Sentiment", f"{avg_sentiment:.3f}")
        
        with col4:
            total_articles = df['news_article_count'].sum()
            st.metric("📊 Total News Articles", int(total_articles))
        
        # Multi-company comparison with PERFECT spacing
        st.header("🔍 Comprehensive Multi-Company Analysis")
        
        # Create individual charts with proper spacing
        fig1, fig2, fig3, fig4 = create_multi_company_comparison(df)
        
        # Display charts in a 2x2 grid with perfect spacing
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced rankings table
        st.header("🏆 Enhanced Company Rankings")
        
        display_df = df[['symbol', 'company_name', 'credit_rating', 'credit_score_raw',
                        'latest_price', 'news_sentiment_score', 'news_article_count']].copy()
        
        display_df = display_df.sort_values('credit_score_raw', ascending=False)
        display_df['latest_price'] = display_df['latest_price'].apply(lambda x: f"${x:.2f}")
        display_df['news_sentiment_score'] = display_df['news_sentiment_score'].apply(lambda x: f"{x:.3f}")
        
        display_df.columns = ['Symbol', 'Company', 'Rating', 'Score', 'Price', 'Sentiment', 'Articles']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with tab2:
        if selected_company:
            # Get selected company data
            company_data = df[df['symbol'] == selected_company].iloc[0]
            
            st.header(f"🔍 Credit Rating Causality Analysis - {selected_company}")
            
            # Company profile card
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"🏢 {company_data['company_name']} ({selected_company})")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("💰 Stock Price", f"${company_data['latest_price']:.2f}")
                with col_b:
                    price_change = company_data.get('price_change_pct', 0)
                    st.metric("📈 Price Change", f"{price_change:+.2f}%")
                with col_c:
                    st.metric("📰 News Sentiment", f"{company_data['news_sentiment_score']:.3f}")
                with col_d:
                    st.metric("📊 News Articles", int(company_data['news_article_count']))
            
            with col2:
                rating = company_data['credit_rating']
                score = company_data['credit_score_raw']
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Credit Assessment</h3>
                    <div class="{get_rating_color_class(rating)}">{rating}</div>
                    <p>Score: {score:.3f}/1.000</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Waterfall chart
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.header("🌊 Credit Score Waterfall Analysis")
            waterfall_chart = create_waterfall_chart(company_data)
            st.plotly_chart(waterfall_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Causality explorer
            st.header("🔗 Interactive Causality Explorer")
            causality_data = create_causality_explorer(company_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Radar chart
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                radar_chart = create_radar_chart(company_data)
                st.plotly_chart(radar_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Factor breakdown
                st.subheader("📊 Factor Impact Analysis")
                
                for category, data in causality_data.items():
                    impact = data['impact']
                    weight = data['weight']
                    contribution = impact * weight
                    
                    if impact > 0.7:
                        impact_class = "positive-impact"
                        icon = "🟢"
                    elif impact < 0.3:
                        impact_class = "negative-impact"
                        icon = "🔴"
                    else:
                        impact_class = "neutral-impact"
                        icon = "🟡"
                    
                    st.markdown(f"""
                    <div class="factor-impact {impact_class}">
                        <h5>{icon} {category}</h5>
                        <p><strong>Impact Score:</strong> {impact:.3f} (Weight: {weight*100:.0f}%)</p>
                        <p><strong>Contribution:</strong> {contribution:.3f} to final score</p>
                        <p><strong>Key Factors:</strong> {', '.join(data['factors'])}</p>
                        <p>{data['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab3:
        if selected_company:
            company_data = df[df['symbol'] == selected_company].iloc[0]
            
            st.header(f"📰 News Sentiment Impact Analysis - {selected_company}")
            
            # News sentiment visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                news_chart = create_news_sentiment_timeline(company_data)
                st.plotly_chart(news_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # News sources breakdown
                sources = safe_eval_list(company_data.get('news_sentiment_sources', []))
                
                st.subheader("📊 News Sources")
                for source in sources:
                    source_name = source.replace('_', ' ').title()
                    st.markdown(f'<span class="data-source-badge">{source_name}</span>', 
                              unsafe_allow_html=True)
                
                # Sentiment breakdown
                st.subheader("📈 Sentiment Analysis")
                sentiment_score = company_data['news_sentiment_score']
                
                if sentiment_score > 0.6:
                    sentiment_color = "positive-impact"
                    sentiment_desc = "Positive sentiment indicates strong market confidence"
                elif sentiment_score < 0.4:
                    sentiment_color = "negative-impact"
                    sentiment_desc = "Negative sentiment suggests market concerns"
                else:
                    sentiment_color = "neutral-impact"
                    sentiment_desc = "Neutral sentiment reflects balanced market view"
                
                st.markdown(f"""
                <div class="factor-impact {sentiment_color}">
                    <h5>Overall News Sentiment</h5>
                    <p><strong>Score:</strong> {sentiment_score:.3f}/1.000</p>
                    <p><strong>Articles:</strong> {company_data['news_article_count']}</p>
                    <p>{sentiment_desc}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display news headlines
            display_news_headlines(company_data)
    
    with tab4:
        st.header("🔗 Factor Correlation Analysis")
        
        # Correlation heatmap
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        corr_chart = create_correlation_heatmap(df)
        if corr_chart:
            st.plotly_chart(corr_chart, use_container_width=True)
            
            st.markdown("""
            <div class="correlation-legend">
                <h5>📋 How to Read the Correlation Matrix:</h5>
                <ul>
                    <li><strong>+1.0:</strong> Perfect positive correlation (green)</li>
                    <li><strong>0.0:</strong> No correlation (yellow)</li>
                    <li><strong>-1.0:</strong> Perfect negative correlation (red)</li>
                </ul>
                <p>Strong correlations (|r| > 0.7) indicate factors that move together, 
                which can help identify causal relationships in credit scoring.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Not enough data for correlation analysis")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Factor importance table
        st.subheader("📊 Factor Statistics Across All Companies")
        
        factor_columns = ['financial_strength', 'market_performance', 'sentiment_weighted', 'pe_health']
        available_factors = [col for col in factor_columns if col in df.columns]
        
        if available_factors:
            stats_df = df[available_factors].describe()
            st.dataframe(stats_df.round(4), use_container_width=True)
    
    with tab5:
        st.header("⚖️ Multi-Company Comparative Analysis")
        
        # Select multiple companies for comparison
        selected_companies = st.multiselect(
            "🏢 Select Companies to Compare", 
            df['symbol'].tolist(), 
            default=df['symbol'].tolist()[:2]
        )
        
        if len(selected_companies) >= 2:
            comparison_df = df[df['symbol'].isin(selected_companies)]
            
            # Create comparison charts with perfect spacing
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Score comparison
                fig = px.bar(
                    comparison_df, 
                    x='symbol', 
                    y='credit_score_raw',
                    color='credit_rating',
                    title="Credit Score Comparison",
                    text='credit_score_raw'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(height=450, margin=dict(l=50, r=50, t=80, b=50))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # News sentiment comparison
                fig = px.bar(
                    comparison_df,
                    x='symbol',
                    y='news_sentiment_score',
                    color='news_article_count',
                    title="News Sentiment Comparison",
                    text='news_sentiment_score'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(height=450, margin=dict(l=50, r=50, t=80, b=50))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed comparison table
            st.subheader("📋 Detailed Factor Comparison")
            
            comparison_metrics = comparison_df[['symbol', 'company_name', 'credit_rating', 
                                              'credit_score_raw', 'financial_strength', 
                                              'market_performance', 'sentiment_weighted',
                                              'pe_health', 'news_sentiment_score']].round(3)
            
            st.dataframe(comparison_metrics, use_container_width=True, hide_index=True)
            
        else:
            st.info("Please select at least 2 companies for comparison")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; font-family: Inter, sans-serif;'>
        <p>🏆 <strong>CredTech Advanced Analytics Platform</strong> - Hackathon 2025</p>
        <p>Real-Time News Sentiment • Multi-Source Intelligence • Causality Analysis • Interactive Insights</p>
        <p>Built with ❤️ using Streamlit • Alpha Vantage • Yahoo Finance • EODHD • Financial Modeling Prep</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()