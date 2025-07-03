"""
AI Trading Agent - Interactive Web Dashboard
"""

import streamlit as st
import asyncio
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 AI Trading Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .agent-status {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-active {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-inactive {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .recommendation-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    
    .news-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agents_running' not in st.session_state:
    st.session_state.agents_running = False
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = {}
if 'market_data' not in st.session_state:
    st.session_state.market_data = {}
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Trading Agent Dashboard</h1>
    <p>Intelligent Multi-Agent Trading System Powered by OpenAI GPT-4</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    
    # Agent controls
    st.markdown("### 🤖 Agent Control")
    
    if st.button("🚀 Start Agents", type="primary", use_container_width=True):
        st.session_state.agents_running = True
        st.success("Agents started successfully!")
        st.rerun()
    
    if st.button("⏹️ Stop Agents", use_container_width=True):
        st.session_state.agents_running = False
        st.warning("Agents stopped")
        st.rerun()
    
    # Settings
    st.markdown("### ⚙️ Settings")
    
    risk_tolerance = st.selectbox(
        "Risk Tolerance",
        ["Conservative", "Moderate", "Aggressive"],
        index=1
    )
    
    max_position_size = st.slider(
        "Max Position Size ($)",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000
    )
    
    auto_execute = st.checkbox("Auto-execute trades", value=False)
    
    # Broker selection
    st.markdown("### 🏦 Broker Settings")
    
    selected_broker = st.selectbox(
        "Select Broker",
        ["Alpaca (Paper)", "Alpaca (Live)", "Interactive Brokers", "Simulation"],
        index=0
    )
    
    broker_connected = st.checkbox("Broker Connected", value=True, disabled=True)
    
    # Market data sources
    st.markdown("### 📊 Data Sources")
    
    data_sources = st.multiselect(
        "Active Data Sources",
        ["Alpha Vantage", "Yahoo Finance", "News API", "Economic Calendar"],
        default=["Alpha Vantage", "Yahoo Finance"]
    )

# Main content area
col1, col2, col3, col4 = st.columns(4)

# Portfolio metrics
with col1:
    st.metric(
        label="📈 Portfolio Value",
        value="$75,240",
        delta="$1,240 (1.7%)"
    )

with col2:
    st.metric(
        label="💰 Cash Available",
        value="$25,000",
        delta="-$2,000"
    )

with col3:
    st.metric(
        label="📊 Active Positions",
        value="12",
        delta="2"
    )

with col4:
    st.metric(
        label="🎯 Today's P&L",
        value="$342",
        delta="$342 (0.46%)"
    )

# Agent status section
st.markdown("## 🤖 Agent Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📈 Market Analyst")
    status_class = "status-active" if st.session_state.agents_running else "status-inactive"
    status_text = "🟢 Active" if st.session_state.agents_running else "🔴 Inactive"
    
    st.markdown(f"""
    <div class="agent-status {status_class}">
        {status_text}
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.agents_running:
        st.write("📊 Analyzing market trends...")
        st.write("🔍 Last analysis: 2 minutes ago")
        st.write("🎯 Confidence: 85%")

with col2:
    st.markdown("### ⚠️ Risk Manager")
    status_class = "status-active" if st.session_state.agents_running else "status-inactive"
    status_text = "🟢 Active" if st.session_state.agents_running else "🔴 Inactive"
    
    st.markdown(f"""
    <div class="agent-status {status_class}">
        {status_text}
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.agents_running:
        st.write("🛡️ Monitoring portfolio risk...")
        st.write("📊 Current VaR: $1,245")
        st.write("⚖️ Risk Level: Moderate")

with col3:
    st.markdown("### 📰 News Analyst")
    status_class = "status-active" if st.session_state.agents_running else "status-inactive"
    status_text = "🟢 Active" if st.session_state.agents_running else "🔴 Inactive"
    
    st.markdown(f"""
    <div class="agent-status {status_class}">
        {status_text}
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.agents_running:
        st.write("📰 Processing market news...")
        st.write("💭 Sentiment: Positive")
        st.write("📈 Impact Score: 7.2/10")

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Portfolio", "🎯 Recommendations", "📈 Market Data", "📰 News Analysis", "⚙️ Trading"
])

with tab1:
    st.markdown("## 📊 Portfolio Overview")
    
    # Portfolio allocation chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Asset Allocation")
        
        # Sample portfolio data
        portfolio_data = {
            'Asset': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'Cash'],
            'Value': [15500, 12800, 9200, 8500, 6700, 25000],
            'Percentage': [20.1, 16.6, 11.9, 11.0, 8.7, 32.4]
        }
        
        df_portfolio = pd.DataFrame(portfolio_data)
        
        fig_pie = px.pie(
            df_portfolio, 
            values='Value', 
            names='Asset',
            title="Portfolio Allocation",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### Performance Chart")
        
        # Sample performance data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        portfolio_values = 70000 + np.cumsum(np.random.randn(len(dates)) * 100)
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#667eea', width=3)
        ))
        
        fig_line.update_layout(
            title="Portfolio Performance (30 Days)",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Positions table
    st.markdown("### Current Positions")
    
    positions_data = {
        'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
        'Shares': [100, 5, 30, 40, 10],
        'Avg Cost': [155.00, 2560.00, 306.67, 212.50, 670.00],
        'Current Price': [158.50, 2580.00, 310.00, 215.00, 672.00],
        'Market Value': [15850, 12900, 9300, 8600, 6720],
        'Unrealized P&L': [350, 100, 100, 100, 20],
        'P&L %': [2.3, 0.8, 1.1, 1.2, 0.3]
    }
    
    df_positions = pd.DataFrame(positions_data)
    
    # Color code P&L
    def color_pnl(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        return f'color: {color}'
    
    styled_df = df_positions.style.applymap(color_pnl, subset=['Unrealized P&L', 'P&L %'])
    st.dataframe(styled_df, use_container_width=True)

with tab2:
    st.markdown("## 🎯 AI Recommendations")
    
    if st.session_state.agents_running:
        # Sample recommendations
        recommendations = [
            {
                "agent": "Market Analyst",
                "symbol": "AAPL",
                "action": "BUY",
                "confidence": 0.85,
                "reasoning": "Strong technical indicators and positive earnings outlook",
                "target_price": 165.00,
                "stop_loss": 150.00,
                "priority": "High"
            },
            {
                "agent": "Risk Manager",
                "symbol": "TSLA",
                "action": "REDUCE",
                "confidence": 0.78,
                "reasoning": "Position size exceeds risk tolerance limits",
                "target_price": None,
                "stop_loss": 200.00,
                "priority": "Medium"
            },
            {
                "agent": "News Analyst",
                "symbol": "GOOGL",
                "action": "HOLD",
                "confidence": 0.72,
                "reasoning": "Mixed sentiment from recent regulatory news",
                "target_price": 2600.00,
                "stop_loss": 2450.00,
                "priority": "Low"
            }
        ]
        
        for i, rec in enumerate(recommendations):
            priority_color = {
                "High": "#dc3545",
                "Medium": "#ffc107", 
                "Low": "#28a745"
            }.get(rec["priority"], "#6c757d")
            
            st.markdown(f"""
            <div class="recommendation-card" style="border-left-color: {priority_color};">
                <h4>🎯 {rec['symbol']} - {rec['action']}</h4>
                <p><strong>Agent:</strong> {rec['agent']}</p>
                <p><strong>Confidence:</strong> {rec['confidence']:.1%}</p>
                <p><strong>Priority:</strong> <span style="color: {priority_color};">{rec['priority']}</span></p>
                <p><strong>Reasoning:</strong> {rec['reasoning']}</p>
                {f"<p><strong>Target Price:</strong> ${rec['target_price']:.2f}</p>" if rec['target_price'] else ""}
                {f"<p><strong>Stop Loss:</strong> ${rec['stop_loss']:.2f}</p>" if rec['stop_loss'] else ""}
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button(f"✅ Accept", key=f"accept_{i}"):
                    st.success(f"Recommendation for {rec['symbol']} accepted!")
            with col2:
                if st.button(f"❌ Reject", key=f"reject_{i}"):
                    st.warning(f"Recommendation for {rec['symbol']} rejected")
    else:
        st.info("🤖 Start the agents to see AI-powered recommendations")

with tab3:
    st.markdown("## 📈 Market Data")
    
    # Market overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Major Indices")
        
        indices_data = {
            'Index': ['S&P 500', 'NASDAQ', 'DOW', 'Russell 2000'],
            'Value': [4785.32, 14912.45, 37234.56, 1987.23],
            'Change': [23.45, 156.78, 234.12, -12.34],
            'Change %': [0.49, 1.06, 0.63, -0.62]
        }
        
        df_indices = pd.DataFrame(indices_data)
        
        def color_change(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
            return f'color: {color}'
        
        styled_indices = df_indices.style.applymap(color_change, subset=['Change', 'Change %'])
        st.dataframe(styled_indices, use_container_width=True)
    
    with col2:
        st.markdown("### Market Sentiment")
        
        # Sentiment gauge
        sentiment_score = 7.2
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = sentiment_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Market Sentiment"},
            delta = {'reference': 5.0},
            gauge = {
                'axis': {'range': [None, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 3], 'color': "lightgray"},
                    {'range': [3, 7], 'color': "gray"},
                    {'range': [7, 10], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 8
                }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Stock prices chart
    st.markdown("### Stock Prices")
    
    # Generate sample stock data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    fig_stocks = go.Figure()
    
    for symbol in symbols:
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        base_price = {'AAPL': 155, 'GOOGL': 2560, 'MSFT': 307, 'TSLA': 213, 'AMZN': 670}[symbol]
        prices = base_price + np.cumsum(np.random.randn(len(dates)) * 2)
        
        fig_stocks.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name=symbol,
            line=dict(width=2)
        ))
    
    fig_stocks.update_layout(
        title="Stock Price Trends (30 Days)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_stocks, use_container_width=True)

with tab4:
    st.markdown("## 📰 News Analysis")
    
    if st.session_state.agents_running:
        # Sample news items
        news_items = [
            {
                "headline": "Apple Reports Strong Q4 Earnings, Beats Expectations",
                "summary": "Apple Inc. reported quarterly earnings that exceeded analyst expectations, driven by strong iPhone sales and services revenue.",
                "sentiment": "Positive",
                "impact_score": 8.5,
                "symbols": ["AAPL"],
                "timestamp": "2024-01-15 09:30:00"
            },
            {
                "headline": "Fed Signals Potential Rate Cuts in 2024",
                "summary": "Federal Reserve officials hint at possible interest rate reductions later this year, boosting market sentiment.",
                "sentiment": "Positive",
                "impact_score": 9.2,
                "symbols": ["SPY", "QQQ"],
                "timestamp": "2024-01-15 08:45:00"
            },
            {
                "headline": "Tesla Faces Production Challenges in Shanghai",
                "summary": "Tesla's Shanghai factory reports temporary production slowdowns due to supply chain issues.",
                "sentiment": "Negative",
                "impact_score": 6.8,
                "symbols": ["TSLA"],
                "timestamp": "2024-01-15 07:20:00"
            }
        ]
        
        for news in news_items:
            sentiment_color = {
                "Positive": "#28a745",
                "Negative": "#dc3545",
                "Neutral": "#6c757d"
            }.get(news["sentiment"], "#6c757d")
            
            st.markdown(f"""
            <div class="news-item">
                <h4>{news['headline']}</h4>
                <p>{news['summary']}</p>
                <div style="margin-top: 10px;">
                    <span style="background-color: {sentiment_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px;">
                        {news['sentiment']}
                    </span>
                    <span style="margin-left: 10px; font-weight: bold;">
                        Impact Score: {news['impact_score']}/10
                    </span>
                    <span style="margin-left: 10px; color: #666;">
                        Symbols: {', '.join(news['symbols'])}
                    </span>
                    <span style="margin-left: 10px; color: #666; font-size: 12px;">
                        {news['timestamp']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("🤖 Start the agents to see news analysis")

with tab5:
    st.markdown("## ⚙️ Trading Interface")
    
    # Manual trading section
    st.markdown("### Manual Trading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Place Order")
        
        symbol = st.text_input("Symbol", value="AAPL")
        action = st.selectbox("Action", ["BUY", "SELL"])
        quantity = st.number_input("Quantity", min_value=1, value=100)
        order_type = st.selectbox("Order Type", ["Market", "Limit", "Stop"])
        
        if order_type == "Limit":
            limit_price = st.number_input("Limit Price", value=155.00)
        
        if st.button("🚀 Place Order", type="primary"):
            st.success(f"Order placed: {action} {quantity} shares of {symbol}")
    
    with col2:
        st.markdown("#### Order History")
        
        order_history = [
            {"Time": "10:30 AM", "Symbol": "AAPL", "Action": "BUY", "Qty": 100, "Price": "$155.50", "Status": "Filled"},
            {"Time": "09:45 AM", "Symbol": "GOOGL", "Action": "BUY", "Qty": 5, "Price": "$2,580.00", "Status": "Filled"},
            {"Time": "09:15 AM", "Symbol": "MSFT", "Action": "SELL", "Qty": 50, "Price": "$310.00", "Status": "Filled"},
        ]
        
        df_orders = pd.DataFrame(order_history)
        st.dataframe(df_orders, use_container_width=True)
    
    # Auto-trading settings
    st.markdown("### Auto-Trading Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_trading_enabled = st.checkbox("Enable Auto-Trading", value=auto_execute)
        
    with col2:
        min_confidence = st.slider("Min Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
        
    with col3:
        max_daily_trades = st.number_input("Max Daily Trades", min_value=1, max_value=50, value=10)
    
    if auto_trading_enabled:
        st.success("🤖 Auto-trading is enabled. The system will execute trades based on AI recommendations.")
    else:
        st.info("📋 Auto-trading is disabled. All recommendations require manual approval.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🤖 AI Trading Agent v1.0 | Powered by OpenAI GPT-4 | Built with ❤️ using Streamlit</p>
    <p>⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for real-time updates
if st.session_state.agents_running:
    import time
    time.sleep(1)
    st.rerun() 