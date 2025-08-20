"""
CredTech Hackathon - Integrated Data Collector
Collects data from all sources (Alpha Vantage, MarketAux, SEC EDGAR) for multiple companies
"""

import os
import json
import pandas as pd
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

from alpha_vantage_client import AlphaVantageClient
from marketaux_client import MarketAuxClient
from sec_edgar_client import SECEdgarClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedDataCollector:
    """Main data collection orchestrator for the hackathon"""
    
    def __init__(self):
        # Initialize all API clients
        self.av_client = AlphaVantageClient()
        self.ma_client = MarketAuxClient()
        self.sec_client = SECEdgarClient()
        
        # Target companies (diverse sectors for credit scoring)
        self.target_companies = [
            'AAPL',   # Technology - Apple Inc
            'MSFT',   # Technology - Microsoft
            'GOOGL',  # Technology - Alphabet
            'JPM',    # Financial - JPMorgan Chase
            'BAC',    # Financial - Bank of America
            'JNJ',    # Healthcare - Johnson & Johnson
            'PG',     # Consumer Goods - Procter & Gamble
            'XOM',    # Energy - ExxonMobil
            'WMT',    # Retail - Walmart
            'KO'      # Consumer Beverage - Coca-Cola
        ]
        
        self.collected_data = {}
        self.collection_metadata = {
            'start_time': None,
            'end_time': None,
            'total_companies': 0,
            'successful_companies': 0,
            'failed_companies': [],
            'data_sources_status': {}
        }
    
    def collect_company_data(self, symbol: str) -> Dict:
        """Collect comprehensive data for a single company"""
        logger.info(f"📊 Collecting data for {symbol}")
        
        company_data = {
            'symbol': symbol,
            'collection_timestamp': datetime.now().isoformat(),
            'data_sources': {
                'alpha_vantage': {'status': 'pending', 'data': {}},
                'marketaux': {'status': 'pending', 'data': {}},
                'sec_edgar': {'status': 'pending', 'data': {}}
            },
            'processed_metrics': {}
        }
        
        # 1. Collect Alpha Vantage data
        try:
            logger.info(f"   📈 Fetching Alpha Vantage data...")
            
            # Stock prices
            stock_data = self.av_client.get_daily_stock_data(symbol, outputsize='compact')
            
            # Company fundamentals
            time.sleep(12)  # Rate limiting
            overview = self.av_client.get_company_overview(symbol)
            
            company_data['data_sources']['alpha_vantage'] = {
                'status': 'success',
                'data': {
                    'stock_prices': stock_data.to_dict('index') if not stock_data.empty else {},
                    'company_overview': overview,
                    'latest_price': float(stock_data['adjusted close'].iloc[-1]) if not stock_data.empty else 0,
                    'price_change_1d': float(stock_data['adjusted close'].iloc[-1] - stock_data['adjusted close'].iloc[-2]) if len(stock_data) > 1 else 0,
                    'volatility_20d': float(stock_data['volatility_20'].iloc[-1]) if not stock_data.empty and 'volatility_20' in stock_data.columns else 0
                }
            }
            logger.info(f"   ✅ Alpha Vantage: Stock data retrieved")
            
        except Exception as e:
            logger.error(f"   ❌ Alpha Vantage failed: {e}")
            company_data['data_sources']['alpha_vantage']['status'] = 'failed'
            company_data['data_sources']['alpha_vantage']['error'] = str(e)
        
        time.sleep(2)
        
        # 2. Collect MarketAux news data
        try:
            logger.info(f"   📰 Fetching MarketAux news...")
            
            # Get news for past 30 days
            news_data = self.ma_client.get_news_for_symbols(symbol, days_back=30, limit=50)
            sentiment_summary = self.ma_client.get_sentiment_summary(symbol, days_back=30)
            
            company_data['data_sources']['marketaux'] = {
                'status': 'success',
                'data': {
                    'news_articles': news_data.to_dict('records') if not news_data.empty else [],
                    'sentiment_summary': sentiment_summary.get(symbol, {}),
                    'article_count': len(news_data),
                    'avg_sentiment_score': sentiment_summary.get(symbol, {}).get('avg_sentiment', 0.5)
                }
            }
            logger.info(f"   ✅ MarketAux: {len(news_data)} articles, sentiment: {sentiment_summary.get(symbol, {}).get('avg_sentiment', 0.5):.2f}")
            
        except Exception as e:
            logger.error(f"   ❌ MarketAux failed: {e}")
            company_data['data_sources']['marketaux']['status'] = 'failed'
            company_data['data_sources']['marketaux']['error'] = str(e)
        
        time.sleep(1)
        
        # 3. Collect SEC EDGAR data
        try:
            logger.info(f"   🏛️  Fetching SEC EDGAR data...")
            
            # Get CIK and financial data
            cik = self.sec_client.get_cik_by_ticker(symbol)
            
            if cik:
                time.sleep(1)  # Rate limiting
                company_facts = self.sec_client.get_company_facts(cik)
                financial_metrics = self.sec_client.extract_financial_metrics(company_facts)
                
                company_data['data_sources']['sec_edgar'] = {
                    'status': 'success',
                    'data': {
                        'cik': cik,
                        'financial_metrics': financial_metrics,
                        'entity_name': financial_metrics.get('entity_name', '')
                    }
                }
                logger.info(f"   ✅ SEC EDGAR: Financial statements retrieved")
            else:
                logger.warning(f"   ⚠️  SEC EDGAR: CIK not found for {symbol}")
                company_data['data_sources']['sec_edgar']['status'] = 'no_cik'
                
        except Exception as e:
            logger.error(f"   ❌ SEC EDGAR failed: {e}")
            company_data['data_sources']['sec_edgar']['status'] = 'failed'
            company_data['data_sources']['sec_edgar']['error'] = str(e)
        
        # 4. Process and combine metrics
        company_data['processed_metrics'] = self._process_combined_metrics(company_data)
        
        return company_data
    
    def _process_combined_metrics(self, company_data: Dict) -> Dict:
        """Process and combine metrics from all data sources"""
        metrics = {
            'symbol': company_data['symbol'],
            'collection_date': company_data['collection_timestamp']
        }
        
        # Alpha Vantage metrics
        av_data = company_data['data_sources']['alpha_vantage']['data']
        if av_data and 'company_overview' in av_data:
            overview = av_data['company_overview']
            metrics.update({
                'company_name': overview.get('Name', ''),
                'sector': overview.get('Sector', ''),
                'industry': overview.get('Industry', ''),
                'market_cap': float(overview.get('MarketCapitalization', 0) or 0),
                'pe_ratio': float(overview.get('PERatio', 0) or 0),
                'beta': float(overview.get('Beta', 1.0) or 1.0),
                'dividend_yield': float(overview.get('DividendYield', 0) or 0),
                'profit_margin': float(overview.get('ProfitMargin', 0) or 0),
                'roa': float(overview.get('ReturnOnAssetsTTM', 0) or 0),
                'roe': float(overview.get('ReturnOnEquityTTM', 0) or 0),
                'latest_price': av_data.get('latest_price', 0),
                'volatility': av_data.get('volatility_20d', 0)
            })
        
        # MarketAux sentiment metrics
        ma_data = company_data['data_sources']['marketaux']['data']
        if ma_data:
            metrics.update({
                'news_sentiment': ma_data.get('avg_sentiment_score', 0.5),
                'news_volume': ma_data.get('article_count', 0),
                'sentiment_trend': ma_data.get('sentiment_summary', {}).get('sentiment_trend', 'neutral')
            })
        
        # SEC EDGAR financial metrics
        sec_data = company_data['data_sources']['sec_edgar']['data']
        if sec_data and 'financial_metrics' in sec_data:
            fm = sec_data['financial_metrics'].get('financial_metrics', {})
            metrics.update({
                'revenue': float(fm.get('revenue', 0) or 0),
                'net_income': float(fm.get('net_income', 0) or 0),
                'total_assets': float(fm.get('total_assets', 0) or 0),
                'total_debt': float(fm.get('total_debt', 0) or 0),
                'shareholders_equity': float(fm.get('shareholders_equity', 0) or 0),
                'debt_to_equity': float(fm.get('debt_to_equity', 0) or 0),
                'sec_profit_margin': float(fm.get('profit_margin', 0) or 0),
                'sec_roa': float(fm.get('return_on_assets', 0) or 0)
            })
        
        # Calculate derived metrics
        self._calculate_derived_metrics(metrics)
        
        return metrics
    
    def _calculate_derived_metrics(self, metrics: Dict):
        """Calculate additional derived metrics for credit scoring"""
        
        # Financial strength score (0-1 scale)
        roa = metrics.get('roa', 0) if metrics.get('roa', 0) != 0 else metrics.get('sec_roa', 0)
        profit_margin = metrics.get('profit_margin', 0) if metrics.get('profit_margin', 0) != 0 else metrics.get('sec_profit_margin', 0)
        debt_ratio = metrics.get('debt_to_equity', 0)
        
        financial_strength = (
            max(0, min(1, (roa + 0.1) / 0.2)) * 0.4 +  # ROA normalized to 0-1
            max(0, min(1, (profit_margin + 0.1) / 0.2)) * 0.4 +  # Profit margin normalized
            max(0, min(1, 1 / (1 + debt_ratio))) * 0.2  # Inverse debt ratio
        )
        
        # Market stability (inverse of volatility and beta)
        beta = metrics.get('beta', 1.0)
        volatility = metrics.get('volatility', 0.02)  # Default 2% volatility
        
        market_stability = (
            max(0, min(1, 1 / (1 + beta - 0.5))) * 0.6 +  # Beta stability
            max(0, min(1, 1 / (1 + volatility * 50))) * 0.4  # Volatility stability
        )
        
        # News sentiment weighted by volume
        sentiment = metrics.get('news_sentiment', 0.5)
        news_volume = metrics.get('news_volume', 0)
        
        sentiment_weighted = sentiment * min(1, news_volume / 20)  # Normalize volume to 0-1
        
        # PE health (optimal PE around 15-20)
        pe_ratio = metrics.get('pe_ratio', 0)
        if pe_ratio > 0:
            pe_health = max(0, 1 - abs(pe_ratio - 17.5) / 17.5)
        else:
            pe_health = 0.3  # Unknown PE gets neutral score
        
        # Dividend reliability
        dividend_reliability = 1.0 if metrics.get('dividend_yield', 0) > 0 else 0.0
        
        # Composite credit score
        credit_score = (
            financial_strength * 0.35 +
            market_stability * 0.20 +
            sentiment_weighted * 0.15 +
            pe_health * 0.20 +
            dividend_reliability * 0.10
        )
        
        # Add derived metrics
        metrics.update({
            'financial_strength': financial_strength,
            'market_stability': market_stability,
            'sentiment_weighted': sentiment_weighted,
            'pe_health': pe_health,
            'dividend_reliability': dividend_reliability,
            'credit_score_raw': credit_score,
            'credit_rating': self._score_to_rating(credit_score)
        })
    
    def _score_to_rating(self, score: float) -> str:
        """Convert numerical score to credit rating"""
        if score >= 0.8:
            return 'A'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.4:
            return 'C'
        else:
            return 'D'
    
    def collect_all_data(self) -> Dict:
        """Collect data for all target companies"""
        self.collection_metadata['start_time'] = datetime.now().isoformat()
        self.collection_metadata['total_companies'] = len(self.target_companies)
        
        logger.info(f"🚀 Starting comprehensive data collection for {len(self.target_companies)} companies")
        logger.info(f"📊 Target companies: {', '.join(self.target_companies)}")
        
        for i, symbol in enumerate(self.target_companies):
            logger.info(f"\\n{'='*60}")
            logger.info(f"Processing {symbol} ({i+1}/{len(self.target_companies)})")
            logger.info(f"{'='*60}")
            
            try:
                company_data = self.collect_company_data(symbol)
                self.collected_data[symbol] = company_data
                self.collection_metadata['successful_companies'] += 1
                
                logger.info(f"✅ {symbol} completed successfully")
                
                # Show quick summary
                metrics = company_data['processed_metrics']
                logger.info(f"   💰 Latest Price: ${metrics.get('latest_price', 0):.2f}")
                logger.info(f"   📊 Credit Rating: {metrics.get('credit_rating', 'N/A')}")
                logger.info(f"   📰 News Sentiment: {metrics.get('news_sentiment', 0.5):.2f}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {symbol}: {e}")
                self.collection_metadata['failed_companies'].append({
                    'symbol': symbol,
                    'error': str(e)
                })
            
            # Rate limiting between companies
            if i < len(self.target_companies) - 1:
                logger.info(f"⏳ Waiting before next company...")
                time.sleep(3)
        
        self.collection_metadata['end_time'] = datetime.now().isoformat()
        
        # Calculate success rate
        success_rate = (self.collection_metadata['successful_companies'] / 
                       self.collection_metadata['total_companies']) * 100
        
        logger.info(f"\\n🎉 Data collection completed!")
        logger.info(f"📊 Success rate: {success_rate:.1f}% ({self.collection_metadata['successful_companies']}/{self.collection_metadata['total_companies']})")
        
        return self.collected_data
    
    def save_data(self, base_dir: str = 'data'):
        """Save collected data in multiple formats"""
        base_path = Path(base_dir)
        base_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (base_path / 'raw').mkdir(exist_ok=True)
        (base_path / 'processed').mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Save raw data as JSON
        raw_file = base_path / 'raw' / f'hackathon_data_{timestamp}.json'
        with open(raw_file, 'w') as f:
            json.dump(self.collected_data, f, indent=2, default=str)
        logger.info(f"💾 Raw data saved: {raw_file}")
        
        # 2. Save processed metrics as CSV
        processed_data = []
        for symbol, data in self.collected_data.items():
            if 'processed_metrics' in data:
                processed_data.append(data['processed_metrics'])
        
        if processed_data:
            df = pd.DataFrame(processed_data)
            csv_file = base_path / 'processed' / f'companies_summary_{timestamp}.csv'
            df.to_csv(csv_file, index=False)
            logger.info(f"💾 Processed data saved: {csv_file}")
            
            # Also save as latest
            latest_csv = base_path / 'processed' / 'companies_summary.csv'
            df.to_csv(latest_csv, index=False)
            logger.info(f"💾 Latest data saved: {latest_csv}")
        
        # 3. Save metadata
        metadata_file = base_path / 'raw' / f'collection_metadata_{timestamp}.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.collection_metadata, f, indent=2, default=str)
        logger.info(f"💾 Metadata saved: {metadata_file}")
        
        return {
            'raw_data_file': str(raw_file),
            'processed_data_file': str(csv_file),
            'metadata_file': str(metadata_file)
        }
    
    def generate_summary_report(self) -> Dict:
        """Generate a summary report of collected data"""
        if not self.collected_data:
            return {'error': 'No data collected'}
        
        summary = {
            'collection_summary': self.collection_metadata,
            'data_quality': {},
            'sector_analysis': {},
            'credit_rating_distribution': {},
            'top_performers': {},
            'data_completeness': {}
        }
        
        # Data quality analysis
        total_companies = len(self.collected_data)
        av_success = sum(1 for data in self.collected_data.values() 
                        if data['data_sources']['alpha_vantage']['status'] == 'success')
        ma_success = sum(1 for data in self.collected_data.values() 
                        if data['data_sources']['marketaux']['status'] == 'success')
        sec_success = sum(1 for data in self.collected_data.values() 
                         if data['data_sources']['sec_edgar']['status'] == 'success')
        
        summary['data_quality'] = {
            'alpha_vantage_success_rate': f"{(av_success/total_companies)*100:.1f}%",
            'marketaux_success_rate': f"{(ma_success/total_companies)*100:.1f}%",
            'sec_edgar_success_rate': f"{(sec_success/total_companies)*100:.1f}%",
            'overall_completeness': f"{((av_success + ma_success + sec_success)/(total_companies*3))*100:.1f}%"
        }
        
        # Sector and rating analysis
        metrics_df = pd.DataFrame([data['processed_metrics'] for data in self.collected_data.values()])
        
        if not metrics_df.empty:
            # Sector distribution
            sector_counts = metrics_df['sector'].value_counts()
            summary['sector_analysis'] = sector_counts.to_dict()
            
            # Credit rating distribution
            rating_counts = metrics_df['credit_rating'].value_counts()
            summary['credit_rating_distribution'] = rating_counts.to_dict()
            
            # Top performers by credit score
            top_performers = metrics_df.nlargest(5, 'credit_score_raw')[['symbol', 'company_name', 'credit_rating', 'credit_score_raw']]
            summary['top_performers'] = top_performers.to_dict('records')
        
        return summary

def main():
    """Main function to run comprehensive data collection"""
    collector = IntegratedDataCollector()
    
    # Collect all data
    print("🚀 Starting CredTech Hackathon Data Collection Pipeline")
    print("=" * 60)
    
    data = collector.collect_all_data()
    
    # Save data
    saved_files = collector.save_data()
    
    # Generate summary
    summary = collector.generate_summary_report()
    
    # Print final summary
    print("\\n📋 FINAL COLLECTION SUMMARY")
    print("=" * 40)
    print(f"Companies processed: {summary['collection_summary']['successful_companies']}/{summary['collection_summary']['total_companies']}")
    print(f"Data sources success rates:")
    print(f"  • Alpha Vantage: {summary['data_quality']['alpha_vantage_success_rate']}")
    print(f"  • MarketAux: {summary['data_quality']['marketaux_success_rate']}")
    print(f"  • SEC EDGAR: {summary['data_quality']['sec_edgar_success_rate']}")
    print(f"Overall completeness: {summary['data_quality']['overall_completeness']}")
    
    print(f"\\n📁 Files saved:")
    for file_type, file_path in saved_files.items():
        print(f"  • {file_type}: {file_path}")
    
    print(f"\\n🎯 Next steps:")
    print(f"  1. Review data/processed/companies_summary.csv")
    print(f"  2. Run model training: python src/models/model_trainer.py")
    print(f"  3. Launch dashboard: streamlit run src/dashboard/streamlit_app.py")

if __name__ == "__main__":
    main()