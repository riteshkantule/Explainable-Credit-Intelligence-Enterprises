"""
SEC EDGAR API Client for CredTech Hackathon
Handles SEC filings and financial statements data
"""

import requests
import pandas as pd
import time
import logging
import json
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class SECEdgarClient:
    """SEC EDGAR API client for fetching company filings and financial data"""
    
    def __init__(self):
        self.base_url = "https://data.sec.gov"
        self.session = requests.Session()
        
        # Required headers for SEC EDGAR API
        self.session.headers.update({
            'User-Agent': 'CredTech Hackathon team@credtech.com',  # Required!
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        })
        
        # Rate limiting: 10 requests/second max
        self.requests_per_second = 10
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Cache for ticker-CIK mapping
        self._ticker_cik_mapping = {}
        
    def _make_request(self, url: str) -> Dict:
        """Make API request with rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            self.last_request_time = time.time()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from {url}: {e}")
            raise
    
    def get_ticker_cik_mapping(self) -> Dict[str, str]:
        """Get mapping of ticker symbols to CIK numbers"""
        if self._ticker_cik_mapping:
            return self._ticker_cik_mapping
        
        url = f"{self.base_url}/files/company_tickers.json"
        logger.info("Fetching ticker-CIK mapping from SEC")
        
        data = self._make_request(url)
        
        # Convert to ticker -> CIK mapping
        for entry in data.values():
            ticker = entry.get('ticker')
            cik = str(entry.get('cik_str')).zfill(10)  # Pad with zeros
            if ticker:
                self._ticker_cik_mapping[ticker.upper()] = cik
        
        logger.info(f"Loaded {len(self._ticker_cik_mapping)} ticker-CIK mappings")
        return self._ticker_cik_mapping
    
    def get_cik_by_ticker(self, ticker: str) -> Optional[str]:
        """Get CIK number for a given ticker symbol"""
        mapping = self.get_ticker_cik_mapping()
        return mapping.get(ticker.upper())
    
    def get_company_facts(self, cik: Union[str, int]) -> Dict:
        """Get company facts (financial statements) by CIK"""
        if isinstance(cik, int):
            cik = str(cik).zfill(10)
        elif isinstance(cik, str) and not cik.startswith('CIK'):
            cik = cik.zfill(10)
            
        url = f"{self.base_url}/api/xbrl/companyfacts/CIK{cik}.json"
        logger.info(f"Fetching company facts for CIK {cik}")
        
        try:
            data = self._make_request(url)
            return data
        except Exception as e:
            logger.error(f"Failed to get company facts for CIK {cik}: {e}")
            return {}
    
    def get_company_submissions(self, cik: Union[str, int]) -> Dict:
        """Get all company submissions by CIK"""
        if isinstance(cik, int):
            cik = str(cik).zfill(10)
        elif isinstance(cik, str):
            cik = cik.zfill(10)
            
        url = f"{self.base_url}/submissions/CIK{cik}.json"
        logger.info(f"Fetching submissions for CIK {cik}")
        
        try:
            data = self._make_request(url)
            return data
        except Exception as e:
            logger.error(f"Failed to get submissions for CIK {cik}: {e}")
            return {}
    
    def extract_financial_metrics(self, company_facts: Dict) -> Dict:
        """Extract key financial metrics from company facts"""
        if not company_facts or 'facts' not in company_facts:
            return {}
        
        facts = company_facts['facts']
        us_gaap = facts.get('us-gaap', {})
        dei = facts.get('dei', {})
        
        # Key financial metrics to extract
        metrics = {}
        
        # Revenue metrics
        revenue_keys = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 
                       'SalesRevenueNet', 'RevenueFromContractWithCustomerIncludingAssessedTax']
        for key in revenue_keys:
            if key in us_gaap:
                metrics['revenue'] = self._extract_latest_value(us_gaap[key])
                break
        
        # Profit metrics
        if 'NetIncomeLoss' in us_gaap:
            metrics['net_income'] = self._extract_latest_value(us_gaap['NetIncomeLoss'])
        
        if 'GrossProfit' in us_gaap:
            metrics['gross_profit'] = self._extract_latest_value(us_gaap['GrossProfit'])
        
        # Balance sheet metrics
        if 'Assets' in us_gaap:
            metrics['total_assets'] = self._extract_latest_value(us_gaap['Assets'])
        
        if 'Liabilities' in us_gaap:
            metrics['total_liabilities'] = self._extract_latest_value(us_gaap['Liabilities'])
        
        if 'StockholdersEquity' in us_gaap:
            metrics['shareholders_equity'] = self._extract_latest_value(us_gaap['StockholdersEquity'])
        
        # Debt metrics
        debt_keys = ['DebtCurrent', 'DebtNoncurrent', 'LongTermDebt', 'ShortTermBorrowings']
        total_debt = 0
        for key in debt_keys:
            if key in us_gaap:
                debt_value = self._extract_latest_value(us_gaap[key])
                if debt_value:
                    total_debt += debt_value
        
        if total_debt > 0:
            metrics['total_debt'] = total_debt
        
        # Cash metrics
        if 'CashAndCashEquivalentsAtCarryingValue' in us_gaap:
            metrics['cash_and_equivalents'] = self._extract_latest_value(
                us_gaap['CashAndCashEquivalentsAtCarryingValue']
            )
        
        # Calculate derived ratios
        if 'revenue' in metrics and 'net_income' in metrics and metrics['revenue']:
            metrics['profit_margin'] = metrics['net_income'] / metrics['revenue']
        
        if 'total_debt' in metrics and 'shareholders_equity' in metrics and metrics['shareholders_equity']:
            metrics['debt_to_equity'] = metrics['total_debt'] / metrics['shareholders_equity']
        
        if 'net_income' in metrics and 'total_assets' in metrics and metrics['total_assets']:
            metrics['return_on_assets'] = metrics['net_income'] / metrics['total_assets']
        
        # Company information
        entity_name = company_facts.get('entityName', '')
        cik = company_facts.get('cik', '')
        
        return {
            'cik': cik,
            'entity_name': entity_name,
            'extraction_date': datetime.now().isoformat(),
            'financial_metrics': metrics
        }
    
    def _extract_latest_value(self, metric_data: Dict) -> Optional[float]:
        """Extract the latest value from SEC metric data structure"""
        if not metric_data or 'units' not in metric_data:
            return None
        
        units = metric_data['units']
        
        # Try USD first, then other units
        for unit_type in ['USD', 'shares', 'pure']:
            if unit_type in units:
                unit_data = units[unit_type]
                if not unit_data:
                    continue
                
                # Sort by end date to get most recent
                sorted_data = sorted(unit_data, key=lambda x: x.get('end', ''), reverse=True)
                
                # Get most recent annual data (10-K form)
                for item in sorted_data:
                    if item.get('form') in ['10-K', '10-Q'] and item.get('val'):
                        try:
                            return float(item['val'])
                        except (ValueError, TypeError):
                            continue
                
                # If no 10-K/10-Q, get any recent data
                if sorted_data and sorted_data[0].get('val'):
                    try:
                        return float(sorted_data[0]['val'])
                    except (ValueError, TypeError):
                        continue
        
        return None
    
    def get_latest_filings(self, cik: Union[str, int], form_types: List[str] = None, 
                          count: int = 10) -> pd.DataFrame:
        """Get latest filings for a company"""
        submissions = self.get_company_submissions(cik)
        
        if not submissions or 'filings' not in submissions:
            return pd.DataFrame()
        
        recent_filings = submissions['filings']['recent']
        
        # Create DataFrame
        filings_df = pd.DataFrame({
            'accessionNumber': recent_filings.get('accessionNumber', []),
            'form': recent_filings.get('form', []),
            'filingDate': recent_filings.get('filingDate', []),
            'reportDate': recent_filings.get('reportDate', []),
            'primaryDocument': recent_filings.get('primaryDocument', []),
            'primaryDocDescription': recent_filings.get('primaryDocDescription', [])
        })
        
        if filings_df.empty:
            return filings_df
        
        # Filter by form types if specified
        if form_types:
            filings_df = filings_df[filings_df['form'].isin(form_types)]
        
        # Convert dates
        filings_df['filingDate'] = pd.to_datetime(filings_df['filingDate'])
        filings_df['reportDate'] = pd.to_datetime(filings_df['reportDate'])
        
        # Sort by filing date and limit count
        filings_df = filings_df.sort_values('filingDate', ascending=False).head(count)
        
        return filings_df
    
    def get_multiple_companies_data(self, tickers: List[str]) -> Dict[str, Dict]:
        """Get financial data for multiple companies"""
        companies_data = {}
        
        for ticker in tickers:
            logger.info(f"Processing {ticker}")
            
            try:
                cik = self.get_cik_by_ticker(ticker)
                if not cik:
                    logger.warning(f"CIK not found for {ticker}")
                    continue
                
                # Get company facts
                facts = self.get_company_facts(cik)
                if facts:
                    # Extract metrics
                    metrics = self.extract_financial_metrics(facts)
                    companies_data[ticker] = {
                        'cik': cik,
                        'raw_facts': facts,
                        'processed_metrics': metrics
                    }
                    logger.info(f"✅ Successfully processed {ticker}")
                else:
                    logger.warning(f"No facts found for {ticker}")
                    
                # Rate limiting between companies
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                continue
        
        return companies_data

def test_sec_edgar_client():
    """Test function for SEC EDGAR client"""
    client = SECEdgarClient()
    
    print("Testing SEC EDGAR client")
    print("=" * 30)
    
    # Test ticker-CIK mapping
    aapl_cik = client.get_cik_by_ticker('AAPL')
    print(f"✅ Apple CIK: {aapl_cik}")
    
    # Test company facts
    if aapl_cik:
        facts = client.get_company_facts(aapl_cik)
        if facts:
            print("✅ Company facts retrieved")
            
            # Extract metrics
            metrics = client.extract_financial_metrics(facts)
            if metrics and 'financial_metrics' in metrics:
                fm = metrics['financial_metrics']
                print(f"Revenue: ${fm.get('revenue', 0):,.0f}")
                print(f"Net Income: ${fm.get('net_income', 0):,.0f}")
                print(f"Total Assets: ${fm.get('total_assets', 0):,.0f}")

if __name__ == "__main__":
    test_sec_edgar_client()