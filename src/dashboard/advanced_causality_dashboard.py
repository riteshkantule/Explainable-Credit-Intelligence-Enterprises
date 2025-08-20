"""
CredTech Hackathon - Advanced Causality Dashboard (Fixed & Improved)
Fixed AttributeError issues and improved UI layout with better spacing
Features: Waterfall charts, correlation analysis, news impact visualization, interactive causality explorer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import math
import ast

# Page configuration
st.set_page_config(
    page_title="CredTech - Advanced Credit Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern design and fixed spacing
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
    padding: 1rem;
    background: white;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_enhanced_data():
    """Load enhanced data with real news sentiment"""
    try:
        df = pd.read_csv('data/processed/companies_summary.csv')
        return df
    except FileNotFoundError:
        return None

def safe_eval_list(value):
    """Safely convert string representation of list to actual list"""
    if isinstance(value, str):
        try:
            # Try to parse as literal list
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If that fails, split by comma or return single item list
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
    """Create comprehensive multi-company comparison with better spacing"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Credit Scores vs Market Cap', 'Risk-Return Profile', 
                       'News Sentiment Impact', 'Financial Performance'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.25,  # Increased spacing
        vertical_spacing=0.25
    )
    
    # 1. Credit Score vs Market Cap
    fig.add_trace(
        go.Scatter(
            x=df['market_cap'] / 1e12,
            y=df['credit_score_raw'],
            mode='markers+text',
            text=df['symbol'],
            textposition="top center",
            marker=dict(
                size=np.sqrt(df['latest_price']) * 3,  # Better size scaling
                color=df['news_sentiment_score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="News Sentiment", x=0.48, len=0.4)
            ),
            name="Companies",
            hovertemplate='<b>%{text}</b><br>Credit Score: %{y:.3f}<br>Market Cap: $%{x:.2f}T<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Risk-Return Profile
    fig.add_trace(
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
                colorbar=dict(title="Credit Score", x=1.05, len=0.4)
            ),
            name="Risk-Return",
            hovertemplate='<b>%{text}</b><br>Volatility: %{x:.2f}%<br>Price Change: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. News Sentiment Impact
    fig.add_trace(
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
        ),
        row=2, col=1
    )
    
    # 4. Financial Metrics
    fig.add_trace(
        go.Bar(
            x=df['symbol'],
            y=df['roa'] * 100,
            name="ROA %",
            marker_color='rgba(102, 126, 234, 0.8)',
            offsetgroup=1,
            hovertemplate='<b>%{x}</b><br>ROA: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=df['symbol'],
            y=df['profit_margin'] * 100,
            name="Profit Margin %",
            marker_color='rgba(255, 127, 14, 0.8)',
            offsetgroup=2,
            hovertemplate='<b>%{x}</b><br>Profit Margin: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update layout with better spacing
    fig.update_xaxes(title_text="Market Cap (Trillions $)", row=1, col=1)
    fig.update_yaxes(title_text="Credit Score", row=1, col=1)
    fig.update_xaxes(title_text="Volatility (%)", row=1, col=2)
    fig.update_yaxes(title_text="Price Change (%)", row=1, col=2)
    fig.update_xaxes(title_text="Company", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment Score", row=2, col=1)
    fig.update_xaxes(title_text="Company", row=2, col=2)
    fig.update_yaxes(title_text="Percentage (%)", row=2, col=2)
    
    fig.update_layout(
        height=900,  # Increased height
        title_text="Comprehensive Multi-Company Analysis",
        font=dict(family="Inter, sans-serif", size=11),
        margin=dict(l=60, r=60, t=100, b=60),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

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
        st.info("📰 No recent news articles available")
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
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 CredTech Advanced Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-Time Credit Intelligence with Causality Analysis & News Sentiment</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_enhanced_data()
    
    if df is None:
        st.error("❌ No data found! Please run the real news sentiment collection first:")
        st.code("python real_news_sentiment.py")
        return
    
    # Sidebar with Company Selector (moved back to sidebar as requested)
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🎛️ Dashboard Controls</div>', unsafe_allow_html=True)
        
        # Company selector
        st.subheader("🏢 Select Company")
        company_options = df['symbol'].tolist()
        selected_company = st.selectbox("Choose company for detailed analysis:", company_options)
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.experimental_rerun()
        
        st.markdown("---")
        
        # Model info
        st.subheader("🤖 Enhanced Model v2.0")
        st.info("""
        **Real News Integration:**
        • Yahoo Finance News ✅
        • EODHD Professional Sentiment ✅
        • Financial Modeling Prep ✅
        
        **Advanced Features:**
        • Multi-source sentiment aggregation
        • Causality analysis
        • Real-time news headlines
        • Interactive factor exploration
        
        **Weighting:**
        • Financial Strength: 35%
        • Market Performance: 25% 
        • News Sentiment: 25%
        • Valuation Health: 15%
        """)
    
    # Main Navigation Tabs (kept in main area)
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
        
        # Multi-company comparison with better layout
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🔍 Comprehensive Multi-Company Analysis")
        comparison_chart = create_multi_company_comparison(df)
        st.plotly_chart(comparison_chart, use_container_width=True)
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
            sources = company_data.get('news_sentiment_sources', [])
            sources = safe_eval_list(sources)
            
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
            
            # Create comparison charts
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















# """
# CredTech Hackathon - Advanced Causality Dashboard
# Enhanced Streamlit dashboard with causality analysis and real news sentiment
# Features: Waterfall charts, correlation analysis, news impact visualization, interactive causality explorer
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.figure_factory as ff
# import sys
# import os
# from pathlib import Path
# from datetime import datetime
# import json
# import math

# # Page configuration
# st.set_page_config(
#     page_title="CredTech - Advanced Credit Intelligence Platform",
#     page_icon="🏦",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Enhanced CSS with modern design
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

# .main-header {
#     font-family: 'Inter', sans-serif;
#     font-size: 3rem;
#     font-weight: 700;
#     text-align: center;
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     margin-bottom: 1rem;
#     text-shadow: 0 4px 6px rgba(0,0,0,0.1);
# }

# .sub-header {
#     font-family: 'Inter', sans-serif;
#     font-size: 1.2rem;
#     text-align: center;
#     color: #6c757d;
#     margin-bottom: 2rem;
# }

# .metric-card {
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     padding: 2rem;
#     border-radius: 15px;
#     color: white;
#     margin: 1rem 0;
#     box-shadow: 0 10px 30px rgba(0,0,0,0.1);
#     transition: transform 0.3s ease;
# }

# .metric-card:hover {
#     transform: translateY(-5px);
# }

# .news-card {
#     background: #f8f9fa;
#     padding: 1.5rem;
#     border-radius: 10px;
#     border-left: 5px solid #007bff;
#     margin: 1rem 0;
#     box-shadow: 0 2px 10px rgba(0,0,0,0.05);
# }

# .rating-A, .rating-A-plus { 
#     color: #28a745; 
#     font-weight: bold; 
#     font-size: 2rem;
#     text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
#     background: linear-gradient(45deg, #28a745, #20c997);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
# }

# .rating-A-minus { 
#     color: #28a745; 
#     font-weight: bold; 
#     font-size: 2rem;
#     text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
# }

# .rating-B-plus, .rating-B { 
#     color: #ffc107; 
#     font-weight: bold; 
#     font-size: 2rem;
#     text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
# }

# .rating-B-minus, .rating-C { 
#     color: #fd7e14; 
#     font-weight: bold; 
#     font-size: 2rem;
#     text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
# }

# .rating-D { 
#     color: #dc3545; 
#     font-weight: bold; 
#     font-size: 2rem;
#     text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
# }

# .causality-box {
#     background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#     padding: 2rem;
#     border-radius: 15px;
#     color: white;
#     margin: 1rem 0;
#     box-shadow: 0 10px 30px rgba(0,0,0,0.15);
# }

# .factor-impact {
#     padding: 1rem;
#     margin: 0.5rem 0;
#     border-radius: 10px;
#     border-left: 5px solid;
#     font-family: 'Inter', sans-serif;
#     font-weight: 500;
# }

# .positive-impact {
#     background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
#     border-left-color: #28a745;
#     color: #155724;
# }

# .negative-impact {
#     background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
#     border-left-color: #dc3545;
#     color: #721c24;
# }

# .neutral-impact {
#     background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
#     border-left-color: #ffc107;
#     color: #856404;
# }

# .data-source-badge {
#     display: inline-block;
#     background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
#     color: white;
#     padding: 0.3rem 0.8rem;
#     border-radius: 20px;
#     font-size: 0.8rem;
#     font-weight: 500;
#     margin: 0.2rem;
#     box-shadow: 0 2px 10px rgba(0,0,0,0.1);
# }

# .correlation-legend {
#     font-size: 0.9rem;
#     color: #6c757d;
#     margin-top: 1rem;
# }

# .stSelectbox > div > div > div {
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     color: white;
#     border-radius: 10px;
# }

# .news-sentiment-positive {
#     background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
#     padding: 1rem;
#     border-radius: 10px;
#     border-left: 5px solid #28a745;
# }

# .news-sentiment-negative {
#     background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
#     padding: 1rem;
#     border-radius: 10px;
#     border-left: 5px solid #dc3545;
# }

# .news-sentiment-neutral {
#     background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
#     padding: 1rem;
#     border-radius: 10px;
#     border-left: 5px solid #ffc107;
# }

# .interactive-widget {
#     background: linear-gradient(135deg, #a8e6cf 0%, #88d8a3 100%);
#     padding: 1.5rem;
#     border-radius: 15px;
#     margin: 1rem 0;
#     box-shadow: 0 5px 20px rgba(0,0,0,0.1);
# }
# </style>
# """, unsafe_allow_html=True)

# @st.cache_data
# def load_enhanced_data():
#     """Load enhanced data with real news sentiment"""
#     try:
#         df = pd.read_csv('data/processed/companies_summary.csv')
#         return df
#     except FileNotFoundError:
#         return None

# def get_rating_color_class(rating):
#     """Get CSS class for rating color"""
#     rating_clean = rating.replace('+', '-plus').replace('-', '-minus')
#     return f"rating-{rating_clean}"

# def create_waterfall_chart(company_data):
#     """Create waterfall chart showing credit score buildup"""
#     symbol = company_data['symbol']
    
#     # Get component scores
#     financial = company_data.get('financial_strength', 0.5) * 0.35
#     market = company_data.get('market_performance', 0.5) * 0.25
#     sentiment = company_data.get('sentiment_weighted', 0.5) * 0.25
#     valuation = company_data.get('pe_health', 0.5) * 0.15
    
#     # Create waterfall data
#     categories = ['Start', 'Financial<br>Strength<br>(35%)', 'Market<br>Performance<br>(25%)', 
#                  'News<br>Sentiment<br>(25%)', 'Valuation<br>Health<br>(15%)', 'Final Score']
    
#     values = [0, financial, market, sentiment, valuation, 
#               financial + market + sentiment + valuation]
    
#     # Create waterfall chart
#     fig = go.Figure(go.Waterfall(
#         name="Credit Score Components",
#         orientation="v",
#         measure=["absolute", "relative", "relative", "relative", "relative", "total"],
#         x=categories,
#         textposition="outside",
#         text=[f"{v:.3f}" for v in values],
#         y=values,
#         connector={"line": {"color": "rgb(63, 63, 63)"}},
#         increasing={"marker": {"color": "rgba(40, 167, 69, 0.8)"}},
#         decreasing={"marker": {"color": "rgba(220, 53, 69, 0.8)"}},
#         totals={"marker": {"color": "rgba(102, 126, 234, 0.8)"}}
#     ))
    
#     fig.update_layout(
#         title=f"Credit Score Waterfall Analysis - {symbol}",
#         height=500,
#         font=dict(family="Inter, sans-serif", size=12),
#         showlegend=False,
#         plot_bgcolor='rgba(0,0,0,0)',
#         paper_bgcolor='rgba(0,0,0,0)'
#     )
    
#     fig.update_xaxes(showgrid=False)
#     fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', range=[0, 1])
    
#     return fig

# def create_correlation_heatmap(df):
#     """Create correlation heatmap between factors"""
    
#     # Select relevant columns for correlation
#     corr_columns = ['financial_strength', 'market_performance', 'sentiment_weighted', 
#                    'pe_health', 'credit_score_raw', 'latest_price', 'market_cap']
    
#     # Filter available columns
#     available_columns = [col for col in corr_columns if col in df.columns]
    
#     if len(available_columns) < 3:
#         return None
    
#     # Calculate correlation matrix
#     corr_matrix = df[available_columns].corr()
    
#     # Create heatmap
#     fig = ff.create_annotated_heatmap(
#         z=corr_matrix.values,
#         x=list(corr_matrix.columns),
#         y=list(corr_matrix.index),
#         annotation_text=corr_matrix.round(2).values,
#         showscale=True,
#         colorscale='RdYlGn',
#         font_colors=['white', 'black']
#     )
    
#     fig.update_layout(
#         title="Factor Correlation Matrix",
#         height=600,
#         font=dict(family="Inter, sans-serif", size=11)
#     )
    
#     return fig

# def create_radar_chart(company_data):
#     """Create radar chart for company factors"""
#     symbol = company_data['symbol']
    
#     # Get normalized factors (0-1 scale)
#     factors = {
#         'Financial<br>Strength': company_data.get('financial_strength', 0.5),
#         'Market<br>Performance': company_data.get('market_performance', 0.5),
#         'News<br>Sentiment': company_data.get('sentiment_weighted', 0.5),
#         'Valuation<br>Health': company_data.get('pe_health', 0.5),
#         'Profitability': company_data.get('roa', 0.05) * 10,  # Scale ROA
#         'Market<br>Stability': 1 - min(1, company_data.get('volatility', 0.02) * 20)  # Inverse volatility
#     }
    
#     categories = list(factors.keys())
#     values = list(factors.values())
    
#     # Add first value to end to close the radar chart
#     values += [values[0]]
#     categories += [categories[0]]
    
#     fig = go.Figure()
    
#     fig.add_trace(go.Scatterpolar(
#         r=values,
#         theta=categories,
#         fill='toself',
#         name=symbol,
#         line=dict(color='rgba(102, 126, 234, 0.8)', width=3),
#         fillcolor='rgba(102, 126, 234, 0.3)'
#     ))
    
#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 1],
#                 showticklabels=True,
#                 ticks="outside"
#             )),
#         showlegend=False,
#         title=f"Multi-Factor Analysis - {symbol}",
#         height=500,
#         font=dict(family="Inter, sans-serif", size=12)
#     )
    
#     return fig

# def create_news_sentiment_timeline(company_data):
#     """Create news sentiment impact visualization"""
#     symbol = company_data['symbol']
    
#     # Get news data
#     news_sentiment = company_data.get('news_sentiment_score', 0.5)
#     article_count = company_data.get('news_article_count', 0)
#     sources = company_data.get('news_sentiment_sources', [])
    
#     # Create impact visualization
#     fig = go.Figure()
    
#     # Sentiment impact bar
#     fig.add_trace(go.Bar(
#         name='News Sentiment Impact',
#         x=[f'Current Sentiment<br>({article_count} articles)'],
#         y=[news_sentiment],
#         marker=dict(
#             color=f'rgba({int(255*(1-news_sentiment))}, {int(255*news_sentiment)}, 100, 0.8)',
#             line=dict(color='rgba(0,0,0,0.8)', width=2)
#         ),
#         text=[f'{news_sentiment:.3f}'],
#         textposition='auto'
#     ))
    
#     # Add neutral line
#     fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
#                   annotation_text="Neutral Sentiment (0.5)")
    
#     fig.update_layout(
#         title=f"News Sentiment Analysis - {symbol}",
#         yaxis_title="Sentiment Score (0-1)",
#         height=400,
#         showlegend=False,
#         font=dict(family="Inter, sans-serif", size=12)
#     )
    
#     return fig

# def create_multi_company_comparison(df):
#     """Create comprehensive multi-company comparison"""
    
#     fig = make_subplots(
#         rows=2, cols=2,
#         subplot_titles=('Credit Scores vs Market Cap', 'Risk-Return Profile', 
#                        'News Sentiment Impact', 'Financial Metrics'),
#         specs=[[{"type": "scatter"}, {"type": "scatter"}],
#                [{"type": "bar"}, {"type": "bar"}]]
#     )
    
#     # 1. Credit Score vs Market Cap
#     fig.add_trace(
#         go.Scatter(
#             x=df['market_cap'] / 1e12,
#             y=df['credit_score_raw'],
#             mode='markers+text',
#             text=df['symbol'],
#             textposition="top center",
#             marker=dict(
#                 size=df['latest_price'] / 10,
#                 color=df['news_sentiment_score'],
#                 colorscale='RdYlGn',
#                 showscale=True,
#                 colorbar=dict(title="News Sentiment", x=0.45)
#             ),
#             name="Companies"
#         ),
#         row=1, col=1
#     )
    
#     # 2. Risk-Return Profile
#     fig.add_trace(
#         go.Scatter(
#             x=df['volatility'] * 100,
#             y=df['price_change_pct'],
#             mode='markers+text',
#             text=df['symbol'],
#             textposition="top center",
#             marker=dict(
#                 size=15,
#                 color=df['credit_score_raw'],
#                 colorscale='Viridis',
#                 showscale=True,
#                 colorbar=dict(title="Credit Score", x=1.02)
#             ),
#             name="Risk-Return"
#         ),
#         row=1, col=2
#     )
    
#     # 3. News Sentiment Impact
#     fig.add_trace(
#         go.Bar(
#             x=df['symbol'],
#             y=df['news_sentiment_score'],
#             marker=dict(
#                 color=df['news_sentiment_score'],
#                 colorscale='RdYlGn'
#             ),
#             text=df['news_article_count'],
#             textposition='auto',
#             name="Sentiment Score"
#         ),
#         row=2, col=1
#     )
    
#     # 4. Financial Metrics
#     fig.add_trace(
#         go.Bar(
#             x=df['symbol'],
#             y=df['roa'] * 100,
#             name="ROA %",
#             marker_color='rgba(102, 126, 234, 0.8)'
#         ),
#         row=2, col=2
#     )
    
#     fig.add_trace(
#         go.Bar(
#             x=df['symbol'],
#             y=df['profit_margin'] * 100,
#             name="Profit Margin %",
#             marker_color='rgba(255, 127, 14, 0.8)'
#         ),
#         row=2, col=2
#     )
    
#     # Update layout
#     fig.update_xaxes(title_text="Market Cap (Trillions $)", row=1, col=1)
#     fig.update_yaxes(title_text="Credit Score", row=1, col=1)
#     fig.update_xaxes(title_text="Volatility (%)", row=1, col=2)
#     fig.update_yaxes(title_text="Price Change (%)", row=1, col=2)
#     fig.update_xaxes(title_text="Company", row=2, col=1)
#     fig.update_yaxes(title_text="Sentiment Score", row=2, col=1)
#     fig.update_xaxes(title_text="Company", row=2, col=2)
#     fig.update_yaxes(title_text="Percentage (%)", row=2, col=2)
    
#     fig.update_layout(
#         height=800,
#         title_text="Comprehensive Multi-Company Analysis",
#         font=dict(family="Inter, sans-serif", size=11)
#     )
    
#     return fig

# def create_causality_explorer(company_data):
#     """Interactive causality explorer"""
#     symbol = company_data['symbol']
    
#     # Create causality chain
#     causality_chain = {
#         'Financial Performance': {
#             'factors': ['ROA', 'Profit Margin', 'Revenue Growth'],
#             'impact': company_data.get('financial_strength', 0.5),
#             'weight': 0.35,
#             'description': 'Core profitability and operational efficiency metrics'
#         },
#         'Market Dynamics': {
#             'factors': ['Stock Performance', 'Volatility', 'Beta'],
#             'impact': company_data.get('market_performance', 0.5),
#             'weight': 0.25,
#             'description': 'Market perception and trading characteristics'
#         },
#         'News & Sentiment': {
#             'factors': ['Media Coverage', 'Sentiment Analysis', 'News Volume'],
#             'impact': company_data.get('sentiment_weighted', 0.5),
#             'weight': 0.25,
#             'description': 'Real-time news sentiment and market perception'
#         },
#         'Valuation Metrics': {
#             'factors': ['P/E Ratio', 'Price-to-Book', 'Market Valuation'],
#             'impact': company_data.get('pe_health', 0.5),
#             'weight': 0.15,
#             'description': 'Valuation health and pricing efficiency'
#         }
#     }
    
#     return causality_chain

# def display_news_headlines(company_data):
#     """Display real news headlines with sentiment"""
#     articles = company_data.get('news_articles', [])
    
#     if not articles:
#         st.info("📰 No recent news articles available")
#         return
    
#     st.subheader("📰 Recent News Headlines")
    
#     for i, article in enumerate(articles[:5]):  # Show top 5
#         title = article.get('title', 'No title available')
#         sentiment = article.get('sentiment_polarity', 0)
#         url = article.get('url', '#')
        
#         # Determine sentiment class
#         if sentiment > 0.1:
#             sentiment_class = "news-sentiment-positive"
#             sentiment_icon = "😊"
#             sentiment_text = f"Positive ({sentiment:.2f})"
#         elif sentiment < -0.1:
#             sentiment_class = "news-sentiment-negative"
#             sentiment_icon = "😟"
#             sentiment_text = f"Negative ({sentiment:.2f})"
#         else:
#             sentiment_class = "news-sentiment-neutral"
#             sentiment_icon = "😐"
#             sentiment_text = f"Neutral ({sentiment:.2f})"
        
#         st.markdown(f"""
#         <div class="{sentiment_class}">
#             <h5>{sentiment_icon} {title}</h5>
#             <p><strong>Sentiment:</strong> {sentiment_text}</p>
#         </div>
#         """, unsafe_allow_html=True)

# def main():
#     """Main dashboard application"""
    
#     # Header
#     st.markdown('<h1 class="main-header">🏦 CredTech Advanced Analytics</h1>', unsafe_allow_html=True)
#     st.markdown('<p class="sub-header">Real-Time Credit Intelligence with Causality Analysis & News Sentiment</p>', unsafe_allow_html=True)
    
#     # Load data
#     df = load_enhanced_data()
    
#     if df is None:
#         st.error("❌ No data found! Please run the real news sentiment collection first:")
#         st.code("python real_news_sentiment.py")
#         return
    
#     # Sidebar
#     with st.sidebar:
#         st.header("🎛️ Analysis Controls")
        
#         # Company selector
#         company_options = df['symbol'].tolist()
#         selected_company = st.selectbox("🏢 Select Company", company_options)
        
#         # Analysis type
#         analysis_type = st.selectbox("📊 Analysis Type", [
#             "📈 Comprehensive Overview",
#             "🔍 Causality Analysis", 
#             "📰 News Impact Analysis",
#             "🔗 Factor Correlations",
#             "⚖️ Comparative Analysis"
#         ])
        
#         # Refresh button
#         if st.button("🔄 Refresh Data"):
#             st.cache_data.clear()
#             st.experimental_rerun()
        
#         st.markdown("---")
        
#         # Model info
#         st.subheader("🤖 Enhanced Model v2.0")
#         st.info("""
#         **Real News Integration:**
#         • Yahoo Finance News ✅
#         • EODHD Professional Sentiment ✅
#         • Financial Modeling Prep ✅
        
#         **Advanced Features:**
#         • Multi-source sentiment aggregation
#         • Causality analysis
#         • Real-time news headlines
#         • Interactive factor exploration
        
#         **Weighting:**
#         • Financial Strength: 35%
#         • Market Performance: 25% 
#         • News Sentiment: 25%
#         • Valuation Health: 15%
#         """)
    
#     # Main content based on analysis type
#     if analysis_type == "📈 Comprehensive Overview":
        
#         # Key metrics row
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             total_companies = len(df)
#             st.metric("🏢 Companies", total_companies)
        
#         with col2:
#             avg_score = df['credit_score_raw'].mean()
#             st.metric("⭐ Avg Credit Score", f"{avg_score:.3f}")
        
#         with col3:
#             avg_sentiment = df['news_sentiment_score'].mean()
#             st.metric("📰 Avg News Sentiment", f"{avg_sentiment:.3f}")
        
#         with col4:
#             total_articles = df['news_article_count'].sum()
#             st.metric("📊 Total News Articles", int(total_articles))
        
#         # Multi-company comparison
#         st.header("🔍 Comprehensive Multi-Company Analysis")
#         comparison_chart = create_multi_company_comparison(df)
#         st.plotly_chart(comparison_chart, use_container_width=True)
        
#         # Enhanced rankings table
#         st.header("🏆 Enhanced Company Rankings")
        
#         display_df = df[['symbol', 'company_name', 'credit_rating', 'credit_score_raw',
#                         'latest_price', 'price_change_pct', 'news_sentiment_score', 
#                         'news_article_count']].copy()
        
#         display_df = display_df.sort_values('credit_score_raw', ascending=False)
#         display_df['latest_price'] = display_df['latest_price'].apply(lambda x: f"${x:.2f}")
#         display_df['price_change_pct'] = display_df['price_change_pct'].apply(lambda x: f"{x:+.2f}%")
#         display_df['news_sentiment_score'] = display_df['news_sentiment_score'].apply(lambda x: f"{x:.3f}")
        
#         display_df.columns = ['Symbol', 'Company', 'Rating', 'Score', 'Price', 'Change %', 'Sentiment', 'Articles']
        
#         st.dataframe(display_df, use_container_width=True, hide_index=True)
    
#     elif analysis_type == "🔍 Causality Analysis":
        
#         # Get selected company data
#         company_data = df[df['symbol'] == selected_company].iloc[0]
        
#         st.header(f"🔍 Credit Rating Causality Analysis - {selected_company}")
        
#         # Company profile card
#         col1, col2 = st.columns([3, 1])
        
#         with col1:
#             st.subheader(f"🏢 {company_data['company_name']} ({selected_company})")
            
#             col_a, col_b, col_c, col_d = st.columns(4)
#             with col_a:
#                 st.metric("💰 Stock Price", f"${company_data['latest_price']:.2f}")
#             with col_b:
#                 st.metric("📈 Price Change", f"{company_data.get('price_change_pct', 0):+.2f}%")
#             with col_c:
#                 st.metric("📰 News Sentiment", f"{company_data['news_sentiment_score']:.3f}")
#             with col_d:
#                 st.metric("📊 News Articles", int(company_data['news_article_count']))
        
#         with col2:
#             rating = company_data['credit_rating']
#             score = company_data['credit_score_raw']
#             st.markdown(f"""
#             <div class="metric-card">
#                 <h3>🎯 Credit Assessment</h3>
#                 <div class="{get_rating_color_class(rating)}">{rating}</div>
#                 <p>Score: {score:.3f}/1.000</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         # Waterfall chart
#         st.header("🌊 Credit Score Waterfall Analysis")
#         waterfall_chart = create_waterfall_chart(company_data)
#         st.plotly_chart(waterfall_chart, use_container_width=True)
        
#         # Causality explorer
#         st.header("🔗 Interactive Causality Explorer")
#         causality_data = create_causality_explorer(company_data)
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Radar chart
#             radar_chart = create_radar_chart(company_data)
#             st.plotly_chart(radar_chart, use_container_width=True)
        
#         with col2:
#             # Factor breakdown
#             st.subheader("📊 Factor Impact Analysis")
            
#             for category, data in causality_data.items():
#                 impact = data['impact']
#                 weight = data['weight']
#                 contribution = impact * weight
                
#                 if impact > 0.7:
#                     impact_class = "positive-impact"
#                     icon = "🟢"
#                 elif impact < 0.3:
#                     impact_class = "negative-impact"
#                     icon = "🔴"
#                 else:
#                     impact_class = "neutral-impact"
#                     icon = "🟡"
                
#                 st.markdown(f"""
#                 <div class="factor-impact {impact_class}">
#                     <h5>{icon} {category}</h5>
#                     <p><strong>Impact Score:</strong> {impact:.3f} (Weight: {weight*100:.0f}%)</p>
#                     <p><strong>Contribution:</strong> {contribution:.3f} to final score</p>
#                     <p><strong>Key Factors:</strong> {', '.join(data['factors'])}</p>
#                     <p>{data['description']}</p>
#                 </div>
#                 """, unsafe_allow_html=True)
    
#     elif analysis_type == "📰 News Impact Analysis":
        
#         company_data = df[df['symbol'] == selected_company].iloc[0]
        
#         st.header(f"📰 News Sentiment Impact Analysis - {selected_company}")
        
#         # News sentiment visualization
#         col1, col2 = st.columns(2)
        
#         with col1:
#             news_chart = create_news_sentiment_timeline(company_data)
#             st.plotly_chart(news_chart, use_container_width=True)
        
#         with col2:
#             # News sources breakdown
#             sources = company_data.get('news_sentiment_sources', [])
#             if isinstance(sources, str):
#                 sources = eval(sources) if sources.startswith('[') else [sources]
            
#             st.subheader("📊 News Sources")
#             for source in sources:
#                 source_name = source.replace('_', ' ').title()
#                 st.markdown(f'<span class="data-source-badge">{source_name}</span>', 
#                           unsafe_allow_html=True)
            
#             # Sentiment breakdown
#             st.subheader("📈 Sentiment Analysis")
#             sentiment_score = company_data['news_sentiment_score']
            
#             if sentiment_score > 0.6:
#                 sentiment_color = "positive-impact"
#                 sentiment_desc = "Positive sentiment indicates strong market confidence"
#             elif sentiment_score < 0.4:
#                 sentiment_color = "negative-impact"
#                 sentiment_desc = "Negative sentiment suggests market concerns"
#             else:
#                 sentiment_color = "neutral-impact"
#                 sentiment_desc = "Neutral sentiment reflects balanced market view"
            
#             st.markdown(f"""
#             <div class="factor-impact {sentiment_color}">
#                 <h5>Overall News Sentiment</h5>
#                 <p><strong>Score:</strong> {sentiment_score:.3f}/1.000</p>
#                 <p><strong>Articles:</strong> {company_data['news_article_count']}</p>
#                 <p>{sentiment_desc}</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         # Display news headlines
#         display_news_headlines(company_data)
    
#     elif analysis_type == "🔗 Factor Correlations":
        
#         st.header("🔗 Factor Correlation Analysis")
        
#         # Correlation heatmap
#         corr_chart = create_correlation_heatmap(df)
#         if corr_chart:
#             st.plotly_chart(corr_chart, use_container_width=True)
            
#             st.markdown("""
#             <div class="correlation-legend">
#                 <h5>📋 How to Read the Correlation Matrix:</h5>
#                 <ul>
#                     <li><strong>+1.0:</strong> Perfect positive correlation (green)</li>
#                     <li><strong>0.0:</strong> No correlation (yellow)</li>
#                     <li><strong>-1.0:</strong> Perfect negative correlation (red)</li>
#                 </ul>
#                 <p>Strong correlations (|r| > 0.7) indicate factors that move together, 
#                 which can help identify causal relationships in credit scoring.</p>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.error("Not enough data for correlation analysis")
        
#         # Factor importance table
#         st.subheader("📊 Factor Statistics Across All Companies")
        
#         factor_columns = ['financial_strength', 'market_performance', 'sentiment_weighted', 'pe_health']
#         available_factors = [col for col in factor_columns if col in df.columns]
        
#         if available_factors:
#             stats_df = df[available_factors].describe()
#             st.dataframe(stats_df.round(4), use_container_width=True)
    
#     elif analysis_type == "⚖️ Comparative Analysis":
        
#         st.header("⚖️ Multi-Company Comparative Analysis")
        
#         # Select multiple companies for comparison
#         selected_companies = st.multiselect(
#             "🏢 Select Companies to Compare", 
#             df['symbol'].tolist(), 
#             default=df['symbol'].tolist()[:2]
#         )
        
#         if len(selected_companies) >= 2:
#             comparison_df = df[df['symbol'].isin(selected_companies)]
            
#             # Create comparison charts
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 # Score comparison
#                 fig = px.bar(
#                     comparison_df, 
#                     x='symbol', 
#                     y='credit_score_raw',
#                     color='credit_rating',
#                     title="Credit Score Comparison",
#                     text='credit_score_raw'
#                 )
#                 fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
#                 fig.update_layout(height=400)
#                 st.plotly_chart(fig, use_container_width=True)
            
#             with col2:
#                 # News sentiment comparison
#                 fig = px.bar(
#                     comparison_df,
#                     x='symbol',
#                     y='news_sentiment_score',
#                     color='news_article_count',
#                     title="News Sentiment Comparison",
#                     text='news_sentiment_score'
#                 )
#                 fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
#                 fig.update_layout(height=400)
#                 st.plotly_chart(fig, use_container_width=True)
            
#             # Detailed comparison table
#             st.subheader("📋 Detailed Factor Comparison")
            
#             comparison_metrics = comparison_df[['symbol', 'company_name', 'credit_rating', 
#                                               'credit_score_raw', 'financial_strength', 
#                                               'market_performance', 'sentiment_weighted',
#                                               'pe_health', 'news_sentiment_score']].round(3)
            
#             st.dataframe(comparison_metrics, use_container_width=True, hide_index=True)
            
#         else:
#             st.info("Please select at least 2 companies for comparison")
    
#     # Footer
#     st.markdown("---")
#     st.markdown("""
#     <div style='text-align: center; color: #6c757d; font-family: Inter, sans-serif;'>
#         <p>🏆 <strong>CredTech Advanced Analytics Platform</strong> - Hackathon 2025</p>
#         <p>Real-Time News Sentiment • Multi-Source Intelligence • Causality Analysis • Interactive Insights</p>
#         <p>Built with ❤️ using Streamlit • Alpha Vantage • Yahoo Finance • EODHD • Financial Modeling Prep</p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()