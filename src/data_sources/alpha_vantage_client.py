"""
Alpha Vantage API Client for CredTech Hackathon
Handles stock data, financial metrics, and technical indicators
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

class AlphaVantageClient:
    """Alpha Vantage API client for fetching financial data"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
        
        # Rate limiting: 5 requests/minute for free tier
        self.requests_per_minute = 5
        self.last_request_time = 0
        
    def _make_request(self, params: Dict) -> Dict:
        """Make API request with rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < 12:  # 12 seconds between requests
            sleep_time = 12 - time_since_last
            logger.info(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        
        params['apikey'] = self.api_key
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            self.last_request_time = time.time()
            
            data = response.json()
            
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if 'Note' in data:
                logger.warning(f"API Note: {data['Note']}")
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def get_daily_stock_data(self, symbol: str, outputsize: str = 'compact') -> pd.DataFrame:
        """Get daily stock price data"""
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': outputsize
        }
        
        logger.info(f"Fetching daily stock data for {symbol}")
        data = self._make_request(params)
        
        if 'Time Series (Daily)' not in data:
            logger.error(f"No data found for {symbol}")
            return pd.DataFrame()
        
        time_series = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Clean column names and convert to numeric
        df.columns = [col.split('. ')[1] for col in df.columns]
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        # Add technical indicators
        df['returns'] = df['adjusted close'].pct_change()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['sma_20'] = df['adjusted close'].rolling(window=20).mean()
        df['sma_50'] = df['adjusted close'].rolling(window=50).mean()
        df['rsi'] = self._calculate_rsi(df['adjusted close'])
        
        logger.info(f"Retrieved {len(df)} days of data for {symbol}")
        return df
    
    def get_company_overview(self, symbol: str) -> Dict:
        """Get company fundamental data"""
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol
        }
        
        logger.info(f"Fetching company overview for {symbol}")
        data = self._make_request(params)
        
        if not data or 'Symbol' not in data:
            logger.error(f"No overview data found for {symbol}")
            return {}
        
        # Convert numeric strings to float
        numeric_fields = [
            'MarketCapitalization', 'EBITDA', 'PERatio', 'PEGRatio', 'BookValue',
            'DividendPerShare', 'DividendYield', 'EPS', 'RevenuePerShareTTM',
            'ProfitMargin', 'OperatingMarginTTM', 'ReturnOnAssetsTTM', 'ReturnOnEquityTTM',
            'RevenueTTM', 'GrossProfitTTM', 'DilutedEPSTTM', 'QuarterlyEarningsGrowthYOY',
            'QuarterlyRevenueGrowthYOY', 'AnalystTargetPrice', 'TrailingPE', 'ForwardPE',
            'PriceToSalesRatioTTM', 'PriceToBookRatio', 'EVToRevenue', 'EVToEBITDA',
            'Beta', '52WeekHigh', '52WeekLow', '50DayMovingAverage', '200DayMovingAverage'
        ]
        
        for field in numeric_fields:
            if field in data and data[field] not in ['None', '-', 'N/A']:
                try:
                    data[field] = float(data[field])
                except (ValueError, TypeError):
                    data[field] = None
                    
        return data
    
    def get_technical_indicator(self, symbol: str, indicator: str = 'RSI',
                              interval: str = 'daily', time_period: int = 14) -> pd.DataFrame:
        """Get technical indicators"""
        params = {
            'function': indicator,
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': 'close'
        }
        
        logger.info(f"Fetching {indicator} for {symbol}")
        data = self._make_request(params)
        
        tech_data_key = None
        for key in data.keys():
            if 'Technical Analysis' in key or indicator.upper() in key.upper():
                tech_data_key = key
                break
                
        if not tech_data_key:
            logger.error(f"No {indicator} data found for {symbol}")
            return pd.DataFrame()
        
        tech_series = data[tech_data_key]
        df = pd.DataFrame.from_dict(tech_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_multiple_stocks_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Get data for multiple stocks with rate limiting"""
        stocks_data = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")
            
            try:
                stock_df = self.get_daily_stock_data(symbol, outputsize='compact')
                if not stock_df.empty:
                    stocks_data[symbol] = stock_df
                    logger.info(f"Successfully fetched data for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue
                
        return stocks_data

def test_alpha_vantage_client():
    """Test function for Alpha Vantage client"""
    API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    if not API_KEY:
        print("❌ ALPHA_VANTAGE_API_KEY not found in environment variables")
        return
    
    client = AlphaVantageClient(API_KEY)
    symbol = 'AAPL'
    
    print(f"Testing Alpha Vantage client with {symbol}")
    print("=" * 50)
    
    # Get stock data
    stock_data = client.get_daily_stock_data(symbol)
    if not stock_data.empty:
        print(f"✅ Stock data: {stock_data.shape}")
        print(f"Latest price: ${stock_data['adjusted close'].iloc[-1]:.2f}")
    
    # Get company overview
    overview = client.get_company_overview(symbol)
    if overview:
        print(f"✅ Company: {overview.get('Name', 'N/A')}")
        print(f"Market Cap: ${overview.get('MarketCapitalization', 'N/A')}")

if __name__ == "__main__":
    test_alpha_vantage_client()