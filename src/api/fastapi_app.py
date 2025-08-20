"""
CredTech Hackathon - FastAPI Application
REST API for credit scoring predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.quick_credit_model import QuickCreditScorer
    from data_sources.data_collector import IntegratedDataCollector
except ImportError:
    print("Warning: Could not import models. Some endpoints may not work.")

# Initialize FastAPI app
app = FastAPI(
    title="CredTech Credit Intelligence API",
    description="Real-time explainable credit scoring API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
scorer = None

# Pydantic models
class CreditScoreRequest(BaseModel):
    symbol: str

class CreditScoreResponse(BaseModel):
    symbol: str
    company_name: str
    credit_rating: str
    credit_score: float
    confidence: float
    top_factors: List[Dict[str, float]]
    timestamp: str

class CompanySummary(BaseModel):
    symbol: str
    company_name: str
    sector: Optional[str] = None
    credit_rating: str
    credit_score: float
    market_cap: Optional[float] = None
    latest_price: Optional[float] = None
    news_sentiment: Optional[float] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global scorer
    try:
        scorer = QuickCreditScorer()
        df = scorer.load_data()
        if df is not None:
            df_features = scorer.engineer_features(df)
            df_final = scorer.create_credit_rating(df_features)
            X, y = scorer.prepare_features(df_final)
            scorer.train_model(X, y)
            scorer.setup_explainability(X[:5])
            print("✅ Model loaded and ready")
        else:
            print("⚠️ No training data found")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")

# Health check
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "CredTech Credit Intelligence API",
        "status": "operational",
        "version": "1.0.0",
        "model_ready": scorer is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        df = pd.read_csv('data/processed/companies_summary.csv')
        data_status = "available"
        company_count = len(df)
    except FileNotFoundError:
        data_status = "not_found"
        company_count = 0
    
    return {
        "api_status": "healthy",
        "model_status": "ready" if scorer is not None else "not_ready",
        "data_status": data_status,
        "company_count": company_count
    }

# Get all companies
@app.get("/api/v1/companies", response_model=List[CompanySummary])
async def get_all_companies():
    """Get all companies with their credit scores"""
    try:
        df = pd.read_csv('data/processed/companies_summary.csv')
        
        companies = []
        for _, row in df.iterrows():
            companies.append(CompanySummary(
                symbol=row['symbol'],
                company_name=row.get('company_name', ''),
                sector=row.get('sector'),
                credit_rating=row['credit_rating'],
                credit_score=float(row['credit_score_raw']),
                market_cap=float(row.get('market_cap', 0)) if pd.notna(row.get('market_cap')) else None,
                latest_price=float(row.get('latest_price', 0)) if pd.notna(row.get('latest_price')) else None,
                news_sentiment=float(row.get('news_sentiment', 0.5)) if pd.notna(row.get('news_sentiment')) else None
            ))
        
        return companies
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No company data found. Please run data collection first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading companies: {str(e)}")

# Get specific company
@app.get("/api/v1/companies/{symbol}", response_model=CompanySummary)
async def get_company(symbol: str):
    """Get specific company information"""
    try:
        df = pd.read_csv('data/processed/companies_summary.csv')
        company_data = df[df['symbol'].str.upper() == symbol.upper()]
        
        if company_data.empty:
            raise HTTPException(status_code=404, detail=f"Company {symbol} not found")
        
        row = company_data.iloc[0]
        return CompanySummary(
            symbol=row['symbol'],
            company_name=row.get('company_name', ''),
            sector=row.get('sector'),
            credit_rating=row['credit_rating'],
            credit_score=float(row['credit_score_raw']),
            market_cap=float(row.get('market_cap', 0)) if pd.notna(row.get('market_cap')) else None,
            latest_price=float(row.get('latest_price', 0)) if pd.notna(row.get('latest_price')) else None,
            news_sentiment=float(row.get('news_sentiment', 0.5)) if pd.notna(row.get('news_sentiment')) else None
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No company data found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading company: {str(e)}")

# Score endpoint
@app.post("/api/v1/score", response_model=CreditScoreResponse)
async def score_company(request: CreditScoreRequest):
    """Get credit score and explanation for a company"""
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not ready. Please wait for initialization.")
    
    try:
        # Load data and find company
        df = pd.read_csv('data/processed/companies_summary.csv')
        company_data = df[df['symbol'].str.upper() == request.symbol.upper()]
        
        if company_data.empty:
            raise HTTPException(status_code=404, detail=f"Company {request.symbol} not found")
        
        # Get prediction
        row = company_data.iloc[0]
        
        # Mock detailed prediction (in real implementation, use trained model)
        top_factors = [
            {"financial_strength": 0.1234},
            {"news_sentiment": 0.0987},
            {"pe_health": 0.0765},
            {"market_stability": -0.0234},
            {"dividend_reliability": 0.0567}
        ]
        
        return CreditScoreResponse(
            symbol=row['symbol'],
            company_name=row.get('company_name', ''),
            credit_rating=row['credit_rating'],
            credit_score=float(row['credit_score_raw']),
            confidence=0.92,  # Mock confidence
            top_factors=top_factors,
            timestamp=pd.Timestamp.now().isoformat()
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No company data found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

# Explanation endpoint
@app.get("/api/v1/explain/{symbol}")
async def explain_score(symbol: str):
    """Get detailed explanation for a company's credit score"""
    try:
        df = pd.read_csv('data/processed/companies_summary.csv')
        company_data = df[df['symbol'].str.upper() == symbol.upper()]
        
        if company_data.empty:
            raise HTTPException(status_code=404, detail=f"Company {symbol} not found")
        
        row = company_data.iloc[0]
        
        # Build explanation
        explanation = {
            "symbol": row['symbol'],
            "company_name": row.get('company_name', ''),
            "credit_rating": row['credit_rating'],
            "credit_score": float(row['credit_score_raw']),
            "explanation": {
                "summary": f"{row.get('company_name', symbol)} receives a {row['credit_rating']} rating based on comprehensive financial and market analysis.",
                "key_factors": {
                    "financial_strength": {
                        "value": float(row.get('financial_strength', 0.5)),
                        "impact": "positive" if row.get('financial_strength', 0.5) > 0.5 else "negative",
                        "description": "Overall financial health including profitability and asset management"
                    },
                    "news_sentiment": {
                        "value": float(row.get('news_sentiment', 0.5)),
                        "impact": "positive" if row.get('news_sentiment', 0.5) > 0.5 else "negative",
                        "description": "Market sentiment based on recent news coverage and analysis"
                    },
                    "market_stability": {
                        "value": float(row.get('market_stability', 0.5)),
                        "impact": "positive" if row.get('market_stability', 0.5) > 0.5 else "negative",
                        "description": "Stock price stability and beta coefficient analysis"
                    }
                },
                "risk_factors": self._identify_risk_factors(row),
                "positive_factors": self._identify_positive_factors(row)
            }
        }
        
        return explanation
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No company data found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

def _identify_risk_factors(row):
    """Identify risk factors for a company"""
    risk_factors = []
    
    if row.get('news_sentiment', 0.5) < 0.4:
        risk_factors.append("Negative news sentiment trend")
    
    if row.get('debt_to_equity', 0) > 1.5:
        risk_factors.append("High debt-to-equity ratio")
    
    if row.get('financial_strength', 0.5) < 0.3:
        risk_factors.append("Weak financial performance metrics")
    
    if row.get('pe_ratio', 0) > 30 or row.get('pe_ratio', 0) < 5:
        risk_factors.append("Unusual P/E ratio indicating potential overvaluation or underperformance")
    
    return risk_factors

def _identify_positive_factors(row):
    """Identify positive factors for a company"""
    positive_factors = []
    
    if row.get('news_sentiment', 0.5) > 0.6:
        positive_factors.append("Strong positive news sentiment")
    
    if row.get('dividend_reliability', 0) > 0:
        positive_factors.append("Consistent dividend payments")
    
    if row.get('financial_strength', 0.5) > 0.7:
        positive_factors.append("Strong financial performance")
    
    if row.get('market_stability', 0.5) > 0.6:
        positive_factors.append("Stable market performance")
    
    return positive_factors

# Batch scoring
@app.post("/api/v1/score/batch")
async def score_multiple_companies(symbols: List[str]):
    """Score multiple companies at once"""
    results = []
    
    for symbol in symbols:
        try:
            request = CreditScoreRequest(symbol=symbol)
            result = await score_company(request)
            results.append(result)
        except HTTPException as e:
            results.append({
                "symbol": symbol,
                "error": e.detail,
                "status_code": e.status_code
            })
    
    return {"results": results}

# Statistics endpoint
@app.get("/api/v1/stats")
async def get_statistics():
    """Get portfolio statistics"""
    try:
        df = pd.read_csv('data/processed/companies_summary.csv')
        
        stats = {
            "total_companies": len(df),
            "rating_distribution": df['credit_rating'].value_counts().to_dict(),
            "average_score": float(df['credit_score_raw'].mean()),
            "score_distribution": {
                "min": float(df['credit_score_raw'].min()),
                "max": float(df['credit_score_raw'].max()),
                "median": float(df['credit_score_raw'].median()),
                "std": float(df['credit_score_raw'].std())
            },
            "sector_breakdown": df['sector'].value_counts().to_dict() if 'sector' in df.columns else {},
            "high_risk_count": len(df[df['credit_rating'].isin(['C', 'D'])]),
            "investment_grade_count": len(df[df['credit_rating'].isin(['A', 'B'])])
        }
        
        return stats
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No data found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)