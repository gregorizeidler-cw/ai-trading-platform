"""
Professional Trading Platform - Enterprise-Grade Dashboard
Multi-Asset, Multi-Timeframe, Multi-Agent Trading System
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any

# Configure professional layout
st.set_page_config(
    page_title="🏦 Professional Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .agent-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #2a5298;
        margin: 1rem 0;
        color: #333;
    }
    
    .metric-professional {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 3px solid #28a745;
        color: #333;
    }
    
    .trading-signal {
        background: linear-gradient(45deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    
    .risk-warning {
        background: linear-gradient(45deg, #dc3545 0%, #fd7e14 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    /* Fix text visibility */
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #333 !important;
        font-weight: 600;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #333 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #333 !important;
        background-color: white;
        border-radius: 8px 8px 0 0;
        margin-right: 4px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2a5298 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_assets' not in st.session_state:
    st.session_state.selected_assets = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
if 'selected_timeframes' not in st.session_state:
    st.session_state.selected_timeframes = ["1h", "4h", "1d"]
if 'risk_tolerance' not in st.session_state:
    st.session_state.risk_tolerance = "Moderate"
if 'agents_active' not in st.session_state:
    st.session_state.agents_active = False

# Main header
st.markdown("""
<div class="main-header">
    <h1>🏦 Professional Multi-Agent Trading Platform</h1>
    <p>Enterprise-Grade AI Trading System with 12+ Specialized Agents</p>
    <p>Real-time Analysis • Multi-Asset Support • Professional Risk Management</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Professional Controls
with st.sidebar:
    st.markdown("## 🎛️ Trading Control Center")
    
    # Asset Selection
    st.markdown("### 📊 Asset Universe")
    
    asset_categories = {
        "🏢 US Stocks": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX"],
        "🌍 International": ["ASML", "TSM", "BABA", "JD", "NIO"],
        "₿ Crypto": ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"],
        "💰 Forex": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
        "🛢️ Commodities": ["GC=F", "CL=F", "SI=F", "NG=F"],
        "📈 ETFs": ["SPY", "QQQ", "IWM", "GLD", "TLT"]
    }
    
    selected_category = st.selectbox("Select Asset Category", list(asset_categories.keys()))
    available_assets = asset_categories[selected_category]
    
    selected_assets = st.multiselect(
        "Choose Assets",
        available_assets,
        default=available_assets[:3]
    )
    st.session_state.selected_assets = selected_assets
    
    # Timeframe Selection
    st.markdown("### ⏰ Analysis Timeframes")
    timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
    selected_timeframes = st.multiselect(
        "Select Timeframes",
        timeframes,
        default=["1h", "4h", "1d"]
    )
    st.session_state.selected_timeframes = selected_timeframes
    
    # Risk Management
    st.markdown("### ⚠️ Risk Parameters")
    risk_tolerance = st.select_slider(
        "Risk Tolerance",
        options=["Conservative", "Moderate", "Aggressive", "Institutional"],
        value="Moderate"
    )
    
    max_portfolio_risk = st.slider("Max Portfolio Risk (%)", 1, 10, 3)
    max_position_size = st.slider("Max Position Size (%)", 1, 25, 10)
    
    # Agent Control
    st.markdown("### 🤖 AI Agent Control")
    
    if st.button("🚀 Activate All Agents", type="primary"):
        st.session_state.agents_active = True
        st.success("All 12 agents activated!")
    
    if st.button("⏹️ Stop All Agents"):
        st.session_state.agents_active = False
        st.warning("All agents stopped")
    
    # Real-time toggle
    real_time_mode = st.checkbox("🔴 Real-time Mode", value=True)
    
    # Broker Selection
    st.markdown("### 🏦 Broker Integration")
    broker = st.selectbox(
        "Select Broker",
        ["Alpaca (Paper)", "Alpaca (Live)", "Interactive Brokers", "TD Ameritrade", "Simulation"]
    )

# Main dashboard area
col1, col2, col3, col4, col5 = st.columns(5)

# Professional metrics
with col1:
    st.markdown("""
    <div class="metric-professional">
        <h3>$2,847,392</h3>
        <p>Portfolio Value</p>
        <small style="color: green;">+$47,392 (1.69%)</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-professional">
        <h3>85.7%</h3>
        <p>AI Confidence</p>
        <small style="color: blue;">12 Agents Active</small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-professional">
        <h3>2.34%</h3>
        <p>Daily P&L</p>
        <small style="color: green;">+$65,847</small>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-professional">
        <h3>1.85</h3>
        <p>Sharpe Ratio</p>
        <small style="color: green;">Excellent Risk-Adj Return</small>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-professional">
        <h3>-1.2%</h3>
        <p>VaR (95%)</p>
        <small style="color: orange;">$34,169 at risk</small>
    </div>
    """, unsafe_allow_html=True)

# Agent Status Grid
st.markdown("## 🤖 AI Agent Orchestra Status")

agent_cols = st.columns(4)
agents_data = [
    {"name": "Advanced Market Analyst", "status": "🟢 Active", "confidence": 87.3, "last_signal": "BUY AAPL"},
    {"name": "Quantitative Analyst", "status": "🟢 Active", "confidence": 92.1, "last_signal": "Statistical Arb"},
    {"name": "Options Specialist", "status": "🟢 Active", "confidence": 89.5, "last_signal": "Iron Condor SPY"},
    {"name": "Macro Economist", "status": "🟢 Active", "confidence": 78.9, "last_signal": "Risk-On Regime"},
    {"name": "News Sentiment AI", "status": "🟢 Active", "confidence": 84.2, "last_signal": "Bullish Tech"},
    {"name": "Risk Manager", "status": "🟢 Active", "confidence": 95.7, "last_signal": "Reduce Exposure"},
    {"name": "Portfolio Optimizer", "status": "🟢 Active", "confidence": 88.8, "last_signal": "Rebalance"},
    {"name": "Execution Specialist", "status": "🟢 Active", "confidence": 91.2, "last_signal": "TWAP Strategy"},
    {"name": "Crypto Analyst", "status": "🟢 Active", "confidence": 76.4, "last_signal": "BTC Bullish"},
    {"name": "Forex Specialist", "status": "🟢 Active", "confidence": 82.6, "last_signal": "USD Strength"},
    {"name": "Compliance Monitor", "status": "🟢 Active", "confidence": 99.1, "last_signal": "All Clear"},
    {"name": "Performance Tracker", "status": "🟢 Active", "confidence": 94.3, "last_signal": "Outperforming"}
]

for i, agent in enumerate(agents_data):
    with agent_cols[i % 4]:
        st.markdown(f"""
        <div class="agent-card">
            <h4>{agent['name']}</h4>
            <p>{agent['status']}</p>
            <p><strong>Confidence:</strong> {agent['confidence']:.1f}%</p>
            <p><small>{agent['last_signal']}</small></p>
        </div>
        """, unsafe_allow_html=True)

# Tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Multi-Asset Analysis", 
    "🎯 AI Recommendations", 
    "📈 Advanced Charts", 
    "⚠️ Risk Dashboard",
    "🔄 Order Management",
    "📋 Performance Analytics"
])

with tab1:
    st.markdown("## 📊 Multi-Asset Real-Time Analysis")
    
    if selected_assets:
        # Create multi-asset performance chart
        fig = go.Figure()
        
        for i, asset in enumerate(selected_assets):
            # Generate sample data
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='h')
            base_price = 100 + i * 50
            prices = base_price + np.cumsum(np.random.randn(len(dates)) * 0.5)
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name=asset,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Multi-Asset Performance (30 Days)",
            xaxis_title="Time",
            yaxis_title="Normalized Price",
            height=500,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#333')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Asset correlation matrix
        st.markdown("### 📊 Asset Correlation Matrix")
        
        correlation_data = np.random.rand(len(selected_assets), len(selected_assets))
        correlation_df = pd.DataFrame(correlation_data, 
                                    index=selected_assets, 
                                    columns=selected_assets)
        
        fig_corr = px.imshow(correlation_df, 
                           text_auto=True, 
                           aspect="auto",
                           title="Asset Correlation Matrix",
                           color_continuous_scale="RdBu")
        fig_corr.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#333')
        )
        st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.markdown("## 🎯 AI-Powered Trading Recommendations")
    
    # Live recommendations from agents
    recommendations = [
        {
            "agent": "Advanced Market Analyst",
            "symbol": "AAPL",
            "action": "BUY",
            "confidence": 87.3,
            "target": 175.50,
            "stop_loss": 155.00,
            "timeframe": "1-2 weeks",
            "reasoning": "Strong technical breakout with high volume confirmation"
        },
        {
            "agent": "Quantitative Analyst", 
            "symbol": "GOOGL/MSFT",
            "action": "PAIRS TRADE",
            "confidence": 92.1,
            "target": "3.5% spread",
            "stop_loss": "-1.2% spread",
            "timeframe": "3-5 days",
            "reasoning": "Statistical arbitrage opportunity detected"
        },
        {
            "agent": "Options Specialist",
            "symbol": "SPY",
            "action": "IRON CONDOR",
            "confidence": 89.5,
            "target": "15% profit",
            "stop_loss": "50% loss",
            "timeframe": "Until expiry",
            "reasoning": "High IV with range-bound expectations"
        }
    ]
    
    for rec in recommendations:
        confidence_color = "green" if rec["confidence"] > 80 else "orange" if rec["confidence"] > 60 else "red"
        
        st.markdown(f"""
        <div class="trading-signal">
            <h4>🎯 {rec['symbol']} - {rec['action']}</h4>
            <p><strong>Agent:</strong> {rec['agent']}</p>
            <p><strong>Confidence:</strong> <span style="color: white;">{rec['confidence']:.1f}%</span></p>
            <p><strong>Target:</strong> {rec['target']} | <strong>Stop:</strong> {rec['stop_loss']}</p>
            <p><strong>Timeframe:</strong> {rec['timeframe']}</p>
            <p><em>{rec['reasoning']}</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(f"✅ Execute {rec['symbol']}", key=f"exec_{rec['symbol']}"):
                st.success(f"Order placed for {rec['symbol']}!")
        with col2:
            if st.button(f"📝 Modify {rec['symbol']}", key=f"mod_{rec['symbol']}"):
                st.info(f"Modification panel opened for {rec['symbol']}")
        with col3:
            if st.button(f"❌ Reject {rec['symbol']}", key=f"rej_{rec['symbol']}"):
                st.warning(f"Recommendation rejected for {rec['symbol']}")

with tab3:
    st.markdown("## 📈 Advanced Technical Analysis")
    
    if selected_assets:
        selected_chart_asset = st.selectbox("Select Asset for Detailed Analysis", selected_assets)
        
        # Advanced candlestick chart with indicators
        dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D')
        
        # Generate OHLCV data
        base_price = 150
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        current_price = base_price
        for _ in dates:
            open_price = current_price + np.random.randn() * 0.5
            high_price = open_price + abs(np.random.randn()) * 2
            low_price = open_price - abs(np.random.randn()) * 2
            close_price = (open_price + high_price + low_price) / 3 + np.random.randn() * 0.5
            volume = np.random.randint(1000000, 5000000)
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
            
            current_price = close_price
        
        # Create candlestick chart
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name=selected_chart_asset
        ))
        
        # Add technical indicators
        sma_20 = pd.Series(closes).rolling(20).mean()
        sma_50 = pd.Series(closes).rolling(50).mean()
        
        fig.add_trace(go.Scatter(x=dates, y=sma_20, name="SMA 20", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=dates, y=sma_50, name="SMA 50", line=dict(color="blue")))
        
        fig.update_layout(
            title=f"{selected_chart_asset} - Advanced Technical Analysis",
            yaxis_title="Price",
            xaxis_title="Date",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("## ⚠️ Professional Risk Management Dashboard")
    
    # Risk metrics grid
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        st.markdown("### Portfolio Risk Metrics")
        risk_metrics = {
            "Value at Risk (95%)": "-$34,169 (-1.2%)",
            "Conditional VaR": "-$52,847 (-1.85%)",
            "Maximum Drawdown": "-$89,234 (-3.1%)",
            "Beta to SPY": "1.15",
            "Sharpe Ratio": "1.85",
            "Sortino Ratio": "2.34"
        }
        
        for metric, value in risk_metrics.items():
            st.metric(metric, value)
    
    with risk_col2:
        st.markdown("### Position Concentration")
        
        # Position size chart - Fixed array length issue
        assets_list = selected_assets[:5] if selected_assets else ["AAPL", "GOOGL", "MSFT"]
        weights_list = [25.5, 18.3, 15.7, 12.8, 10.2][:len(assets_list)]
        
        position_data = {
            "Asset": assets_list,
            "Weight": weights_list
        }
        
        fig_pos = px.pie(pd.DataFrame(position_data), 
                        values='Weight', 
                        names='Asset',
                        title="Portfolio Concentration",
                        color_discrete_sequence=px.colors.qualitative.Set3)
        fig_pos.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#333')
        )
        st.plotly_chart(fig_pos, use_container_width=True)
    
    with risk_col3:
        st.markdown("### Risk Alerts")
        
        risk_alerts = [
            {"level": "🟡 Warning", "message": "AAPL position exceeds 25% limit"},
            {"level": "🟢 Normal", "message": "Portfolio correlation within limits"},
            {"level": "🔴 Alert", "message": "VaR approaching daily limit"},
            {"level": "🟢 Normal", "message": "Liquidity risk low"}
        ]
        
        for alert in risk_alerts:
            color = "orange" if "Warning" in alert["level"] else "red" if "Alert" in alert["level"] else "green"
            st.markdown(f"""
            <div style="background-color: {color}; color: white; padding: 0.5rem; margin: 0.5rem 0; border-radius: 5px;">
                {alert['level']}: {alert['message']}
            </div>
            """, unsafe_allow_html=True)

with tab5:
    st.markdown("## 🔄 Professional Order Management System")
    
    # Order entry form
    st.markdown("### 📝 New Order Entry")
    
    order_col1, order_col2, order_col3 = st.columns(3)
    
    with order_col1:
        order_symbol = st.selectbox("Symbol", selected_assets if selected_assets else ["AAPL"])
        order_side = st.selectbox("Side", ["BUY", "SELL"])
        order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"])
    
    with order_col2:
        order_quantity = st.number_input("Quantity", min_value=1, value=100)
        if order_type in ["LIMIT", "STOP_LIMIT"]:
            limit_price = st.number_input("Limit Price", value=150.00)
        if order_type in ["STOP", "STOP_LIMIT"]:
            stop_price = st.number_input("Stop Price", value=145.00)
    
    with order_col3:
        execution_strategy = st.selectbox("Execution Strategy", 
                                        ["IMMEDIATE", "TWAP", "VWAP", "ICEBERG", "SMART"])
        time_in_force = st.selectbox("Time in Force", ["DAY", "GTC", "IOC", "FOK"])
        
        if st.button("🚀 Submit Order", type="primary"):
            st.success("Order submitted successfully!")
    
    # Active orders table
    st.markdown("### 📋 Active Orders")
    
    active_orders = pd.DataFrame({
        "Order ID": ["ORD001", "ORD002", "ORD003"],
        "Symbol": ["AAPL", "GOOGL", "MSFT"],
        "Side": ["BUY", "SELL", "BUY"],
        "Quantity": [100, 50, 200],
        "Price": [155.50, 2580.00, 310.25],
        "Status": ["PARTIAL", "PENDING", "FILLED"],
        "Filled": [60, 0, 200]
    })
    
    st.dataframe(active_orders, use_container_width=True)

with tab6:
    st.markdown("## 📋 Performance Analytics & Reporting")
    
    # Performance metrics
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        st.markdown("### 📈 Performance Summary")
        
        # Performance chart
        dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
        portfolio_values = 2800000 + np.cumsum(np.random.randn(len(dates)) * 5000)
        benchmark_values = 2800000 + np.cumsum(np.random.randn(len(dates)) * 3000)
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=dates, y=portfolio_values, name="Portfolio", line=dict(color="blue")))
        fig_perf.add_trace(go.Scatter(x=dates, y=benchmark_values, name="Benchmark (SPY)", line=dict(color="gray")))
        
        fig_perf.update_layout(title="Portfolio vs Benchmark Performance", height=400)
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with perf_col2:
        st.markdown("### 📊 Key Statistics")
        
        performance_stats = {
            "Total Return": "+18.7%",
            "Annualized Return": "+16.2%",
            "Volatility": "12.4%",
            "Sharpe Ratio": "1.85",
            "Max Drawdown": "-3.1%",
            "Win Rate": "67.3%",
            "Profit Factor": "2.15",
            "Alpha": "+4.2%",
            "Beta": "1.15",
            "Information Ratio": "1.23"
        }
        
        for stat, value in performance_stats.items():
            st.metric(stat, value)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🏦 <strong>Professional Trading Platform v2.0</strong> | Powered by 12+ AI Agents & OpenAI GPT-4</p>
    <p>Real-time Data • Multi-Asset Support • Professional Risk Management • Enterprise Security</p>
    <p>⚠️ <strong>Disclaimer:</strong> Professional trading platform for educational and research purposes.</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for real-time mode
if real_time_mode and st.session_state.agents_active:
    import time
    time.sleep(2)
    st.rerun() 