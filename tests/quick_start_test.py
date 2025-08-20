"""
CredTech Hackathon - Quick Start Test Script
============================================

INSTRUCTIONS:
1. Replace YOUR_API_KEY_HERE and YOUR_TOKEN_HERE with your actual API keys
2. Install required packages: pip install requests pandas
3. Run this script: python quick_start_test.py

This will test all your API connections and create sample data.
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time

# 🔑 REPLACE THESE WITH YOUR ACTUAL API KEYS
ALPHA_VANTAGE_API_KEY = "KXN6LO5LMNRZAI8S"  # From alphavantage.co
MARKETAUX_API_TOKEN = "MJuxicXoNOKhYS9JchDDiMvL0TO2jYhODWztVq3D"      # From marketaux.com

def test_alpha_vantage():
    """Test Alpha Vantage API and get Apple stock data"""
    print("🔍 Testing Alpha Vantage API...")
    
    # Test basic stock data
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': 'AAPL',
        'outputsize': 'compact',
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'Error Message' in data:
            print(f"❌ Alpha Vantage Error: {data['Error Message']}")
            return None
        elif 'Note' in data:
            print(f"⚠️  Alpha Vantage Note: {data['Note']}")
            print("You might be hitting rate limits. Try again in a minute.")
            return None
        elif 'Time Series (Daily)' in data:
            print("✅ Alpha Vantage: Successfully connected!")
            
            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.columns = [col.split('. ')[1] for col in df.columns]
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            print(f"📊 Retrieved {len(df)} days of AAPL stock data")
            print(f"🗓️  Date range: {df.index.min().date()} to {df.index.max().date()}")
            print(f"💰 Latest closing price: ${df['close'].iloc[-1]:.2f}")
            
            return df
        else:
            print(f"❌ Unexpected response format: {list(data.keys())}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_marketaux():
    """Test MarketAux API and get Apple news"""
    print("\n📰 Testing MarketAux API...")
    
    # Calculate date range (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        'symbols': 'AAPL',
        'filter_entities': 'true',
        'language': 'en',
        'published_after': start_date.strftime('%Y-%m-%dT%H:%M:%S'),
        'published_before': end_date.strftime('%Y-%m-%dT%H:%M:%S'),
        'limit': 10,
        'api_token': MARKETAUX_API_TOKEN
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data:
            print(f"❌ MarketAux Error: {data['error']}")
            return None
        elif 'data' in data and data['data']:
            print("✅ MarketAux: Successfully connected!")
            
            articles = data['data']
            print(f"📰 Found {len(articles)} news articles for AAPL")
            
            # Analyze sentiment
            sentiment_counts = {}
            for article in articles:
                sentiment = article.get('sentiment', 'neutral')
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            print("📈 Sentiment breakdown:")
            for sentiment, count in sentiment_counts.items():
                print(f"   {sentiment.title()}: {count} articles")
            
            # Show latest articles
            print("\n🔥 Latest articles:")
            for i, article in enumerate(articles[:3]):
                print(f"   {i+1}. {article['title'][:80]}...")
                print(f"      Sentiment: {article.get('sentiment', 'unknown')}")
                print(f"      Published: {article.get('published_at', 'unknown')}")
                print()
            
            return articles
        else:
            print("⚠️  No news articles found for AAPL in the last 7 days")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_sec_edgar():
    """Test SEC EDGAR API and get Apple's CIK"""
    print("\n🏛️  Testing SEC EDGAR API...")
    
    headers = {
        'User-Agent': 'CredTech Hackathon team@credtech.com'  # Required!
    }
    
    try:
        # First, get the ticker-to-CIK mapping
        tickers_url = 'https://www.sec.gov/files/company_tickers.json'
        response = requests.get(tickers_url, headers=headers, timeout=30)
        response.raise_for_status()
        tickers_data = response.json()
        
        # Find Apple's CIK
        aapl_cik = None
        for entry in tickers_data.values():
            if entry.get('ticker') == 'AAPL':
                aapl_cik = entry.get('cik_str')
                break
        
        if not aapl_cik:
            print("❌ Could not find Apple's CIK")
            return None
        
        print(f"✅ SEC EDGAR: Found Apple's CIK: {aapl_cik}")
        print("📊 SEC EDGAR API is working (skipping full financial data download for speed)")
        
        # Note: We skip the full financial data download here to save time
        # In real usage, you would uncomment the code below:
        """
        facts_url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{aapl_cik:0>10}.json'
        time.sleep(1)  # Be nice to the SEC servers
        response = requests.get(facts_url, headers=headers, timeout=60)
        financial_data = response.json()
        """
        
        return {'cik': aapl_cik, 'status': 'connected'}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def create_sample_dataset():
    """Combine all data sources into a sample dataset"""
    print("\n🔄 Creating sample dataset...")
    
    # Test all APIs
    stock_data = test_alpha_vantage()
    news_data = test_marketaux()
    financial_data = test_sec_edgar()
    
    # Create summary
    sample_data = {
        'company': 'Apple Inc.',
        'symbol': 'AAPL',
        'data_collection_date': datetime.now().isoformat(),
        'data_sources': {
            'stock_data_available': stock_data is not None and len(stock_data) > 0,
            'news_data_available': news_data is not None and len(news_data) > 0,
            'financial_data_available': financial_data is not None
        }
    }
    
    if stock_data is not None and len(stock_data) > 0:
        sample_data['stock_summary'] = {
            'latest_price': float(stock_data['close'].iloc[-1]),
            'price_change_1d': float(stock_data['close'].iloc[-1] - stock_data['close'].iloc[-2]),
            'volume_latest': float(stock_data['volume'].iloc[-1]),
            'data_points': len(stock_data)
        }
    
    if news_data is not None and len(news_data) > 0:
        sentiment_counts = {}
        for article in news_data:
            sentiment = article.get('sentiment', 'neutral')
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        sample_data['news_summary'] = {
            'article_count': len(news_data),
            'sentiment_breakdown': sentiment_counts,
            'latest_article_title': news_data[0].get('title', '') if news_data else ''
        }
    
    # Save sample data
    with open('sample_data.json', 'w') as f:
        json.dump(sample_data, f, indent=2, default=str)
    
    print("✅ Sample dataset created and saved to 'sample_data.json'")
    return sample_data

def main():
    """Main function to run all tests"""
    print("🚀 CredTech Hackathon - API Testing Script")
    print("=" * 50)
    
    # Check if API keys are set
    if ALPHA_VANTAGE_API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️  Please replace ALPHA_VANTAGE_API_KEY with your actual API key")
        print("   Get it from: https://www.alphavantage.co/support/#api-key")
        print("   1. Click 'Get your free API key today'")
        print("   2. Fill in your information")
        print("   3. Copy the key and paste it in this script")
        return False
    
    if MARKETAUX_API_TOKEN == "YOUR_TOKEN_HERE":
        print("⚠️  Please replace MARKETAUX_API_TOKEN with your actual API token")
        print("   Get it from: https://www.marketaux.com/")
        print("   1. Click 'Get Free API Key'")
        print("   2. Sign up with email")
        print("   3. Copy the token and paste it in this script")
        return False
    
    # Run tests
    sample_data = create_sample_dataset()
    
    # Print summary
    print("\n📊 DATA COLLECTION SUMMARY:")
    print("=" * 30)
    
    sources = sample_data['data_sources']
    for source, available in sources.items():
        status = "✅ Available" if available else "❌ Not Available"
        source_name = source.replace('_', ' ').title()
        print(f"{source_name}: {status}")
    
    if all(sources.values()):
        print("\n🎉 ALL DATA SOURCES ARE WORKING! You're ready to start building.")
        print("\n📋 NEXT STEPS:")
        print("1. Set up your project structure (see setup-guide.md)")
        print("2. Create a virtual environment")
        print("3. Install all required packages")
        print("4. Start building your credit scoring model!")
        print("\n📁 Your sample data is saved in 'sample_data.json'")
    else:
        print("\n⚠️  Some data sources need attention. Check the error messages above.")
        print("Don't worry - even if some sources don't work, you can still proceed!")
    
    return True

if __name__ == "__main__":
    main()