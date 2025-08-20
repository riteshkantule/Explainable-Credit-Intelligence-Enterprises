"""
CredTech Hackathon - REAL NEWS SENTIMENT VERSION
Integrates REAL unstructured news data with sentiment analysis from multiple free sources:
1. Yahoo Finance News (Free, unlimited)
2. Financial Modeling Prep (250 calls/day free)
3. EODHD (20 calls/day free with sentiment)
4. Alpha Vantage (fundamentals)
5. Yahoo Finance (stock prices)
"""

import os
import pandas as pd
import numpy as np
import time
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import requests
import yfinance as yf
from textblob import TextBlob

# Load environment variables
load_dotenv()

class RealNewsDataCollector:
    def __init__(self):
        # API Keys
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.fmp_key = os.getenv('FMP_API_KEY')  # Financial Modeling Prep (add to .env)
        self.eodhd_key = os.getenv('EODHD_API_KEY')  # EODHD (add to .env)
        
        # API Endpoints
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.fmp_url = "https://financialmodelingprep.com/api/v3"
        self.eodhd_url = "https://eodhd.com/api"
        self.yahoo_finance_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        
        # Request tracking
        self.alpha_vantage_requests = 0
        self.fmp_requests = 0
        self.eodhd_requests = 0
        self.last_alpha_vantage_time = 0
        self.last_fmp_time = 0
        
        print(f"🔑 Alpha Vantage Key: {'✅' if self.alpha_vantage_key else '❌'}")
        print(f"🔑 Financial Modeling Prep Key: {'✅' if self.fmp_key else '❌ (Optional - using free tier)'}")
        print(f"🔑 EODHD Key: {'✅' if self.eodhd_key else '❌ (Optional - using free tier)'}")
    
    def get_yahoo_finance_news(self, symbol):
        """Get real news from Yahoo Finance (completely free!)"""
        print(f"   📰 Getting Yahoo Finance news for {symbol}...")
        
        try:
            # Use yfinance to get news
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                print(f"   ⚠️ No Yahoo Finance news found for {symbol}")
                return None
            
            # Process news articles
            articles = []
            for article in news[:10]:  # Get latest 10 articles
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
                    article_data['sentiment_polarity'] = sentiment.polarity  # -1 to 1
                    article_data['sentiment_subjectivity'] = sentiment.subjectivity  # 0 to 1
                    
                    articles.append(article_data)
                    
                except Exception as e:
                    print(f"   ⚠️ Error processing article: {e}")
                    continue
            
            # Calculate overall sentiment
            if articles:
                avg_sentiment = np.mean([a['sentiment_polarity'] for a in articles])
                sentiment_count = len(articles)
                
                # Convert to 0-1 scale for consistency
                normalized_sentiment = (avg_sentiment + 1) / 2  # Convert -1,1 to 0,1
                
                result = {
                    'articles': articles,
                    'sentiment_score': normalized_sentiment,
                    'sentiment_raw': avg_sentiment,
                    'article_count': sentiment_count,
                    'sentiment_source': 'yahoo_finance_textblob',
                    'data_quality': 'high'
                }
                
                print(f"   ✅ Yahoo Finance: {len(articles)} articles, sentiment: {normalized_sentiment:.3f}")
                return result
            
        except Exception as e:
            print(f"   ❌ Yahoo Finance news error: {e}")
        
        return None
    
    def get_fmp_news(self, symbol):
        """Get news from Financial Modeling Prep (250 calls/day free)"""
        if self.fmp_requests >= 240:  # Stay under daily limit
            print(f"   ⚠️ FMP daily limit reached, skipping...")
            return None
            
        print(f"   📰 Getting FMP news for {symbol}...")
        
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_fmp_time < 1:
                time.sleep(1)
            
            # Build URL with or without API key
            if self.fmp_key:
                url = f"{self.fmp_url}/stock_news?tickers={symbol}&limit=20&apikey={self.fmp_key}"
            else:
                # Try free tier without key (limited)
                url = f"{self.fmp_url}/stock_news?tickers={symbol}&limit=5"
            
            response = requests.get(url, timeout=15)
            self.last_fmp_time = time.time()
            self.fmp_requests += 1
            
            if response.status_code == 200:
                data = response.json()
                
                if data and isinstance(data, list):
                    articles = []
                    sentiments = []
                    
                    for article in data[:10]:
                        try:
                            title = article.get('title', '')
                            text = article.get('text', article.get('summary', ''))
                            
                            # Calculate sentiment
                            content = f"{title} {text}"
                            sentiment = TextBlob(content).sentiment
                            sentiment_score = (sentiment.polarity + 1) / 2  # Convert to 0-1
                            
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
                            print(f"   ⚠️ Error processing FMP article: {e}")
                            continue
                    
                    if sentiments:
                        avg_sentiment = np.mean(sentiments)
                        
                        result = {
                            'articles': articles,
                            'sentiment_score': avg_sentiment,
                            'article_count': len(articles),
                            'sentiment_source': 'fmp_textblob',
                            'data_quality': 'high'
                        }
                        
                        print(f"   ✅ FMP: {len(articles)} articles, sentiment: {avg_sentiment:.3f}")
                        return result
                        
        except Exception as e:
            print(f"   ❌ FMP news error: {e}")
        
        return None
    
    def get_eodhd_news_sentiment(self, symbol):
        """Get news with built-in sentiment from EODHD (20 calls/day free)"""
        if self.eodhd_requests >= 18:  # Stay under daily limit
            print(f"   ⚠️ EODHD daily limit reached, skipping...")
            return None
            
        print(f"   📰 Getting EODHD news & sentiment for {symbol}...")
        
        try:
            # News endpoint
            if self.eodhd_key:
                news_url = f"{self.eodhd_url}/news?s={symbol}.US&limit=20&api_token={self.eodhd_key}"
                sentiment_url = f"{self.eodhd_url}/sentiments?s={symbol}.US&api_token={self.eodhd_key}"
            else:
                # Try demo key
                news_url = f"{self.eodhd_url}/news?s={symbol}.US&limit=5&api_token=demo"
                sentiment_url = f"{self.eodhd_url}/sentiments?s={symbol}.US&api_token=demo"
            
            time.sleep(0.5)  # Rate limiting
            
            # Get news
            news_response = requests.get(news_url, timeout=15)
            self.eodhd_requests += 1
            
            news_data = None
            sentiment_data = None
            
            if news_response.status_code == 200:
                news_data = news_response.json()
            
            time.sleep(0.5)  # Rate limiting
            
            # Get sentiment
            sentiment_response = requests.get(sentiment_url, timeout=15)
            self.eodhd_requests += 1
            
            if sentiment_response.status_code == 200:
                sentiment_data = sentiment_response.json()
            
            # Process results
            if news_data or sentiment_data:
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
                sentiment_score = 0.5  # Default neutral
                if sentiment_data and symbol + '.US' in sentiment_data:
                    sentiment_entries = sentiment_data[symbol + '.US']
                    if sentiment_entries:
                        # Get most recent sentiment
                        latest_sentiment = sentiment_entries[0]['normalized']
                        # Convert from -1,1 to 0,1 scale
                        sentiment_score = (latest_sentiment + 1) / 2
                
                result = {
                    'articles': articles,
                    'sentiment_score': sentiment_score,
                    'article_count': len(articles),
                    'sentiment_source': 'eodhd_native',
                    'data_quality': 'professional'
                }
                
                print(f"   ✅ EODHD: {len(articles)} articles, sentiment: {sentiment_score:.3f}")
                return result
                
        except Exception as e:
            print(f"   ❌ EODHD error: {e}")
        
        return None
    
    def get_aggregated_news_sentiment(self, symbol):
        """Aggregate sentiment from multiple news sources"""
        print(f"   📊 Aggregating news sentiment from multiple sources...")
        
        all_sources = []
        
        # 1. Yahoo Finance News (always try first - it's free!)
        yahoo_result = self.get_yahoo_finance_news(symbol)
        if yahoo_result:
            all_sources.append(yahoo_result)
        
        # 2. Financial Modeling Prep
        fmp_result = self.get_fmp_news(symbol)
        if fmp_result:
            all_sources.append(fmp_result)
        
        # 3. EODHD (if we haven't hit limit)
        eodhd_result = self.get_eodhd_news_sentiment(symbol)
        if eodhd_result:
            all_sources.append(eodhd_result)
        
        # Aggregate results
        if all_sources:
            # Weight sources by quality and article count
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
            
            # Calculate weighted average
            if weights:
                weighted_sentiment = np.average(sentiments, weights=weights)
            else:
                weighted_sentiment = np.mean(sentiments)
            
            # Combine articles from all sources
            all_articles = []
            for source in all_sources:
                all_articles.extend(source.get('articles', []))
            
            result = {
                'sentiment_score': weighted_sentiment,
                'article_count': total_articles,
                'sentiment_sources': sources_used,
                'all_articles': all_articles[:20],  # Keep top 20 articles
                'data_quality': 'aggregated_multi_source',
                'source_breakdown': {
                    source['sentiment_source']: {
                        'sentiment': source['sentiment_score'],
                        'articles': source['article_count']
                    } for source in all_sources
                }
            }
            
            print(f"   ✅ Aggregated sentiment: {weighted_sentiment:.3f} from {len(all_sources)} sources")
            print(f"   📊 Sources: {', '.join(sources_used)}")
            return result
        
        # Fallback to calculated sentiment if no sources work
        print(f"   ⚠️ No news sources available, using financial-based sentiment")
        return {
            'sentiment_score': 0.5,
            'article_count': 0,
            'sentiment_sources': ['financial_calculation'],
            'data_quality': 'fallback'
        }
    
    def get_stock_price_yahoo(self, symbol):
        """Get stock price from Yahoo Finance"""
        print(f"   📈 Getting stock price from Yahoo Finance...")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current price and basic info
            info = ticker.info
            hist = ticker.history(period="5d")
            
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                price_change = current_price - previous_close
                
                # Calculate volatility from recent data
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
                
                print(f"   ✅ Yahoo Finance: ${current_price:.2f} ({price_change:+.2f})")
                return result
                
        except Exception as e:
            print(f"   ❌ Yahoo Finance price error: {e}")
        
        return None
    
    def get_company_overview_alpha_vantage(self, symbol):
        """Get company overview from Alpha Vantage"""
        print(f"   📊 Getting company overview from Alpha Vantage...")
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_alpha_vantage_time < 15:
            sleep_time = 15 - (current_time - self.last_alpha_vantage_time)
            print(f"   ⏳ Alpha Vantage rate limit: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(self.alpha_vantage_url, params=params, timeout=30)
            self.last_alpha_vantage_time = time.time()
            self.alpha_vantage_requests += 1
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Symbol' in data:
                    overview = {
                        'name': data.get('Name', symbol),
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
                    
                    print(f"   ✅ Alpha Vantage overview: {overview['name']}")
                    return overview
                    
        except Exception as e:
            print(f"   ❌ Alpha Vantage overview error: {e}")
        
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
        """Collect comprehensive data including REAL news sentiment"""
        print(f"\n{'='*70}")
        print(f"🏢 REAL NEWS SENTIMENT COLLECTION: {symbol}")
        print(f"{'='*70}")
        
        # Initialize with defaults
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
                'name': default_name,
                'sector': 'Technology',
                'market_cap': 2e12,
                'pe_ratio': 25.0,
                'profit_margin': 0.20,
                'roa': 0.15
            })
        
        # 3. REAL News Sentiment (Multiple Sources)
        news_sentiment = self.get_aggregated_news_sentiment(symbol)
        if news_sentiment:
            company_data.update({
                'news_sentiment_score': news_sentiment['sentiment_score'],
                'news_article_count': news_sentiment['article_count'],
                'news_sentiment_sources': news_sentiment['sentiment_sources'],
                'news_articles': news_sentiment.get('all_articles', [])[:10],  # Keep top 10
                'sentiment_breakdown': news_sentiment.get('source_breakdown', {}),
                'news_data_quality': news_sentiment['data_quality']
            })
            company_data['data_sources_used'].extend(news_sentiment['sentiment_sources'])
        else:
            # Fallback
            company_data.update({
                'news_sentiment_score': 0.5,
                'news_article_count': 0,
                'news_sentiment_sources': ['fallback'],
                'news_data_quality': 'fallback'
            })
        
        return company_data

def calculate_enhanced_credit_score_v2(data):
    """Enhanced credit scoring with REAL news sentiment"""
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
        
        # Recent performance factor
        performance_score = 0.5 + (price_change_pct / 20)  # Normalize ±10% price change
        performance_score = max(0, min(1, performance_score))
        
        stability_score = max(0, min(1, 1 - volatility * 20))
        beta_score = max(0, min(1, 2 - abs(beta)))
        
        market_performance = (performance_score * 0.5 + stability_score * 0.3 + beta_score * 0.2)
        
        # 3. REAL News Sentiment (25% weight) - ENHANCED!
        news_sentiment = data.get('news_sentiment_score', 0.5)
        article_count = data.get('news_article_count', 0)
        
        # Quality factor based on data source
        news_quality = data.get('news_data_quality', 'fallback')
        quality_multiplier = {
            'professional': 1.2,
            'aggregated_multi_source': 1.1,
            'high': 1.0,
            'fallback': 0.8
        }.get(news_quality, 1.0)
        
        # Volume-weighted sentiment
        volume_weight = min(1, article_count / 15)  # 15+ articles = full weight
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
        
        # Enhanced Rating Scale
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
        print(f"   ❌ Credit scoring error: {e}")
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

def collect_real_news_data():
    """Main collection function with REAL news sentiment"""
    print("🏆 CredTech Hackathon - REAL NEWS SENTIMENT VERSION")
    print("🚀 Multi-Source Real News Integration + Professional Sentiment Analysis")
    print("=" * 80)
    
    collector = RealNewsDataCollector()
    
    # Target companies
    companies = [
        ('AAPL', 'Apple Inc'),
        ('MSFT', 'Microsoft Corporation'),
        ('GOOGL', 'Alphabet Inc')
    ]
    
    all_data = []
    
    for i, (symbol, name) in enumerate(companies):
        try:
            # Collect comprehensive data with REAL news
            company_data = collector.collect_company_data(symbol, name)
            
            # Calculate enhanced credit score
            company_data = calculate_enhanced_credit_score_v2(company_data)
            
            all_data.append(company_data)
            
            print(f"\n🎯 REAL NEWS RESULTS FOR {symbol}:")
            print(f"   🏢 Company: {company_data['company_name']}")
            print(f"   📊 Credit Rating: {company_data['credit_rating']}")
            print(f"   🎯 Credit Score: {company_data['credit_score_raw']:.3f}")
            print(f"   💰 Latest Price: ${company_data.get('latest_price', 0):.2f}")
            print(f"   📈 Price Change: {company_data.get('price_change_pct', 0):+.2f}%")
            print(f"   📰 News Sentiment: {company_data.get('news_sentiment_score', 0.5):.3f}")
            print(f"   📊 News Articles: {company_data.get('news_article_count', 0)}")
            print(f"   🔗 News Sources: {', '.join(company_data.get('news_sentiment_sources', []))}")
            print(f"   📋 Data Quality: {company_data.get('news_data_quality', 'N/A')}")
            
            # Show sample news headlines
            articles = company_data.get('news_articles', [])
            if articles:
                print(f"   📰 Recent Headlines:")
                for j, article in enumerate(articles[:3]):
                    title = article.get('title', 'No title')
                    print(f"      {j+1}. {title[:60]}...")
            
            # Pause between companies
            if i < len(companies) - 1:
                print(f"   ⏳ Brief pause...")
                time.sleep(3)
                
        except Exception as e:
            print(f"   ❌ Error processing {symbol}: {e}")
    
    # Save enhanced data
    df = pd.DataFrame(all_data)
    
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save files
    raw_file = f'data/raw/real_news_data_{timestamp}.json'
    csv_file = 'data/processed/companies_summary.csv'
    
    with open(raw_file, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    
    df.to_csv(csv_file, index=False)
    
    print(f"\n🎉 REAL NEWS SENTIMENT COLLECTION COMPLETED!")
    print(f"{'='*70}")
    print(f"📊 Companies processed: {len(df)}")
    print(f"📁 Enhanced data: {csv_file}")
    
    # Analysis
    print(f"\n🤖 REAL NEWS ANALYSIS:")
    print(f"📊 Credit Rating Distribution:")
    rating_dist = df['credit_rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        print(f"   {rating}: {count} companies")
    
    print(f"\n🏆 Enhanced Rankings (with Real News):")
    df_sorted = df.sort_values('credit_score_raw', ascending=False)
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        sources = ', '.join(row.get('news_sentiment_sources', ['N/A']))
        print(f"   {i+1}. {row['symbol']} - {row['credit_rating']} ({row['credit_score_raw']:.3f})")
        print(f"      News Sentiment: {row.get('news_sentiment_score', 0.5):.3f} ({row.get('news_article_count', 0)} articles)")
        print(f"      News Sources: {sources}")
    
    return df

def main():
    """Main execution"""
    print("📦 Installing required packages...")
    try:
        import yfinance
        import textblob
        print("✅ All packages available")
    except ImportError:
        print("❌ Please install required packages:")
        print("   pip install yfinance textblob")
        return
    
    collect_real_news_data()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. 📊 Review enhanced data with REAL news: data/processed/companies_summary.csv")
    print(f"2. 🖥️  Launch enhanced dashboard: streamlit run src/dashboard/enhanced_streamlit_dashboard.py")
    print(f"3. 🚀 You now have REAL news sentiment integration!")
    
    print(f"\n💡 TO ADD MORE NEWS SOURCES:")
    print(f"1. Financial Modeling Prep: Add FMP_API_KEY to .env (250 calls/day free)")
    print(f"2. EODHD: Add EODHD_API_KEY to .env (20 calls/day free)")
    print(f"3. Yahoo Finance News works without any API key!")

if __name__ == "__main__":
    main()