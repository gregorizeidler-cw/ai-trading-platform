"""
Complete Multi-Agent Trading System Demo
Demonstrates all new features: additional agents, broker integration, and dashboard
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import our AI agents
from ai_agents.agents.market_analyst import MarketAnalystAgent
from ai_agents.agents.risk_manager import RiskManagerAgent
from ai_agents.agents.news_analyst import NewsAnalystAgent
from ai_agents.agents.portfolio_manager import PortfolioManagerAgent
from ai_agents.decision_engine.coordinator import AgentCoordinator

# Import broker integration
from order_execution.brokers.alpaca_broker import AlpacaBroker
from order_execution.brokers.base_broker import Order, OrderSide, OrderType
from order_execution.execution.order_manager import OrderManager, ExecutionStrategy

# Import data pipeline
from data_pipeline.collectors.yahoo_finance_collector import YahooFinanceCollector
from data_pipeline.processors.data_processor import DataProcessor

# Import configuration
from config.settings import Settings


class CompleteTradingSystem:
    """Complete multi-agent trading system with all features"""
    
    def __init__(self):
        self.settings = Settings()
        self.agents = {}
        self.coordinator = None
        self.order_manager = OrderManager()
        self.data_collector = YahooFinanceCollector()
        self.data_processor = DataProcessor()
        self.brokers = {}
        self.portfolio_data = {}
        self.market_data = {}
        self.running = False
        
    async def initialize(self):
        """Initialize all system components"""
        print("🚀 Initializing Complete Multi-Agent Trading System...")
        
        # Initialize agents
        await self._initialize_agents()
        
        # Initialize brokers
        await self._initialize_brokers()
        
        # Initialize data pipeline
        await self._initialize_data_pipeline()
        
        # Initialize coordinator
        await self._initialize_coordinator()
        
        print("✅ System initialization complete!")
        
    async def _initialize_agents(self):
        """Initialize all AI agents"""
        print("🤖 Initializing AI Agents...")
        
        # Create specialized agents
        self.agents = {
            'market_analyst': MarketAnalystAgent(),
            'risk_manager': RiskManagerAgent(),
            'news_analyst': NewsAnalystAgent(),
            'portfolio_manager': PortfolioManagerAgent()
        }
        
        print(f"✅ Initialized {len(self.agents)} AI agents")
        
    async def _initialize_brokers(self):
        """Initialize broker connections"""
        print("🏦 Initializing Broker Connections...")
        
        # Initialize Alpaca broker (paper trading)
        alpaca_config = {
            'api_key': self.settings.alpaca_api_key or 'demo_key',
            'secret_key': self.settings.alpaca_secret_key or 'demo_secret',
            'base_url': 'https://paper-api.alpaca.markets'
        }
        
        alpaca_broker = AlpacaBroker(alpaca_config)
        self.brokers['alpaca'] = alpaca_broker
        
        # Add broker to order manager
        self.order_manager.add_broker(alpaca_broker)
        
        # Connect to brokers
        connection_results = await self.order_manager.connect_all_brokers()
        
        for broker_name, connected in connection_results.items():
            status = "✅ Connected" if connected else "❌ Failed"
            print(f"  {broker_name}: {status}")
            
    async def _initialize_data_pipeline(self):
        """Initialize data collection and processing"""
        print("📊 Initializing Data Pipeline...")
        
        # Test data collection
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        try:
            # Collect market data
            market_data = await self.data_collector.collect_data(symbols)
            
            if market_data:
                # Process the data
                processed_data = self.data_processor.process_market_data(market_data)
                self.market_data = processed_data
                print(f"✅ Collected data for {len(symbols)} symbols")
            else:
                print("⚠️ No market data collected - using simulated data")
                self.market_data = self._generate_sample_market_data(symbols)
                
        except Exception as e:
            print(f"⚠️ Data collection error: {e} - using simulated data")
            self.market_data = self._generate_sample_market_data(symbols)
            
    async def _initialize_coordinator(self):
        """Initialize the agent coordinator"""
        print("🎯 Initializing Agent Coordinator...")
        
        # Create coordinator with all agents
        agent_list = list(self.agents.values())
        self.coordinator = AgentCoordinator(agent_list)
        
        print("✅ Agent coordinator initialized")
        
    def _generate_sample_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Generate sample market data for demonstration"""
        import random
        
        market_data = {}
        
        for symbol in symbols:
            base_prices = {
                'AAPL': 155.0, 'GOOGL': 2560.0, 'MSFT': 307.0, 
                'TSLA': 213.0, 'AMZN': 670.0
            }
            
            base_price = base_prices.get(symbol, 100.0)
            
            market_data[symbol] = {
                'price': base_price + random.uniform(-5, 5),
                'volume': random.randint(1000000, 10000000),
                'rsi': random.uniform(30, 70),
                'macd': random.uniform(-2, 2),
                'bollinger_upper': base_price + random.uniform(5, 15),
                'bollinger_lower': base_price - random.uniform(5, 15),
                'volatility': random.uniform(0.15, 0.35),
                'beta': random.uniform(0.8, 1.5),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        return market_data
        
    async def run_complete_analysis(self):
        """Run complete multi-agent analysis"""
        print("\n" + "="*80)
        print("🎯 RUNNING COMPLETE MULTI-AGENT ANALYSIS")
        print("="*80)
        
        # Prepare analysis data
        analysis_data = {
            'market_data': self.market_data,
            'portfolio_data': await self._get_portfolio_data(),
            'news_data': self._generate_sample_news(),
            'symbols': list(self.market_data.keys()),
            'risk_profile': 'moderate',
            'market_conditions': {
                'volatility': 0.25,
                'sentiment': 'positive',
                'trend': 'bullish'
            }
        }
        
        # Run coordinated analysis
        coordinated_result = await self.coordinator.coordinate_decision(analysis_data)
        
        # Display results
        await self._display_analysis_results(coordinated_result)
        
        return coordinated_result
        
    async def _get_portfolio_data(self) -> Dict[str, Any]:
        """Get current portfolio data from brokers"""
        try:
            portfolio = await self.order_manager.get_consolidated_portfolio()
            
            if not portfolio or not portfolio.get('positions'):
                # Generate sample portfolio data
                portfolio = {
                    'total_value': 75000.0,
                    'cash': 25000.0,
                    'positions': {
                        'AAPL': {
                            'shares': 100,
                            'avg_cost': 150.0,
                            'value': 15500.0,
                            'volatility': 0.25,
                            'beta': 1.2
                        },
                        'GOOGL': {
                            'shares': 5,
                            'avg_cost': 2560.0,
                            'value': 12800.0,
                            'volatility': 0.30,
                            'beta': 1.1
                        },
                        'MSFT': {
                            'shares': 30,
                            'avg_cost': 306.0,
                            'value': 9200.0,
                            'volatility': 0.22,
                            'beta': 0.9
                        }
                    }
                }
                
            return portfolio
            
        except Exception as e:
            print(f"⚠️ Error getting portfolio data: {e}")
            return {}
            
    def _generate_sample_news(self) -> List[Dict[str, Any]]:
        """Generate sample news data"""
        return [
            {
                'headline': 'Apple Reports Strong Q4 Earnings, Beats Expectations',
                'summary': 'Apple Inc. reported quarterly earnings that exceeded analyst expectations, driven by strong iPhone sales and services revenue.',
                'sentiment': 'positive',
                'symbols': ['AAPL'],
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'Financial News'
            },
            {
                'headline': 'Fed Signals Potential Rate Cuts in 2024',
                'summary': 'Federal Reserve officials hint at possible interest rate reductions later this year, boosting market sentiment.',
                'sentiment': 'positive',
                'symbols': ['SPY', 'QQQ'],
                'timestamp': (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                'source': 'Reuters'
            },
            {
                'headline': 'Tesla Faces Production Challenges in Shanghai',
                'summary': 'Tesla Shanghai factory reports temporary production slowdowns due to supply chain issues.',
                'sentiment': 'negative',
                'symbols': ['TSLA'],
                'timestamp': (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                'source': 'Bloomberg'
            }
        ]
        
    async def _display_analysis_results(self, results: Dict[str, Any]):
        """Display comprehensive analysis results"""
        print("\n📊 ANALYSIS RESULTS")
        print("-" * 50)
        
        # Overall decision
        final_decision = results.get('final_decision', {})
        print(f"🎯 Final Decision: {final_decision.get('action', 'HOLD')}")
        print(f"🔍 Confidence: {final_decision.get('confidence', 0):.1%}")
        print(f"💭 Reasoning: {final_decision.get('reasoning', 'No reasoning provided')}")
        
        # Agent consensus
        consensus = results.get('consensus_analysis', {})
        print(f"\n🤝 Agent Consensus: {consensus.get('consensus_level', 0):.1%}")
        print(f"📈 Agreement Score: {consensus.get('agreement_score', 0):.2f}")
        
        # Individual agent results
        print("\n🤖 INDIVIDUAL AGENT ANALYSES")
        print("-" * 50)
        
        agent_results = results.get('agent_results', {})
        
        for agent_name, agent_result in agent_results.items():
            print(f"\n📋 {agent_name.upper()}")
            
            if 'error' in agent_result:
                print(f"  ❌ Error: {agent_result['error']}")
                continue
                
            # Display key metrics
            confidence = agent_result.get('confidence_score', 0)
            print(f"  🎯 Confidence: {confidence:.1%}")
            
            # Display recommendations
            recommendations = agent_result.get('recommendations', [])
            if recommendations:
                print(f"  📝 Recommendations ({len(recommendations)}):")
                for i, rec in enumerate(recommendations[:3]):  # Show top 3
                    symbol = rec.get('symbol', 'N/A')
                    action = rec.get('action', 'N/A')
                    rec_confidence = rec.get('confidence', 0)
                    print(f"    {i+1}. {symbol}: {action} (Confidence: {rec_confidence:.1%})")
                    
        # Risk analysis
        risk_analysis = results.get('risk_analysis', {})
        if risk_analysis:
            print(f"\n⚠️ RISK ANALYSIS")
            print("-" * 50)
            risk_level = risk_analysis.get('overall_risk_level', 'Unknown')
            print(f"📊 Overall Risk Level: {risk_level}")
            
            if 'var_95' in risk_analysis:
                print(f"📉 Value at Risk (95%): ${risk_analysis['var_95']:,.2f}")
                
        # Market sentiment
        sentiment_analysis = results.get('sentiment_analysis', {})
        if sentiment_analysis:
            print(f"\n📰 MARKET SENTIMENT")
            print("-" * 50)
            overall_sentiment = sentiment_analysis.get('overall_sentiment', 'neutral')
            sentiment_score = sentiment_analysis.get('sentiment_score', 0.5)
            print(f"💭 Overall Sentiment: {overall_sentiment.title()}")
            print(f"📊 Sentiment Score: {sentiment_score:.2f}")
            
    async def execute_recommendations(self, results: Dict[str, Any]):
        """Execute trading recommendations"""
        print("\n" + "="*80)
        print("🚀 EXECUTING TRADING RECOMMENDATIONS")
        print("="*80)
        
        final_decision = results.get('final_decision', {})
        action = final_decision.get('action', 'HOLD')
        confidence = final_decision.get('confidence', 0)
        
        if action == 'HOLD' or confidence < 0.7:
            print("📋 No high-confidence trading actions recommended")
            return
            
        # Get specific recommendations
        agent_results = results.get('agent_results', {})
        
        for agent_name, agent_result in agent_results.items():
            recommendations = agent_result.get('recommendations', [])
            
            for rec in recommendations:
                if rec.get('confidence', 0) >= 0.7:
                    await self._execute_single_recommendation(rec)
                    
    async def _execute_single_recommendation(self, recommendation: Dict[str, Any]):
        """Execute a single trading recommendation"""
        symbol = recommendation.get('symbol', '')
        action = recommendation.get('action', '')
        confidence = recommendation.get('confidence', 0)
        
        if not symbol or action not in ['BUY', 'SELL']:
            print(f"⚠️ Invalid recommendation: {recommendation}")
            return
            
        print(f"\n🎯 Executing: {action} {symbol} (Confidence: {confidence:.1%})")
        
        try:
            # Create order
            order_side = OrderSide.BUY if action == 'BUY' else OrderSide.SELL
            quantity = 10  # Small test quantity
            
            order = Order(
                symbol=symbol,
                side=order_side,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
            
            # Submit order using smart execution
            order_id = await self.order_manager.submit_order(
                order=order,
                strategy=ExecutionStrategy.SMART
            )
            
            print(f"✅ Order submitted successfully - ID: {order_id}")
            
            # Monitor order status
            await asyncio.sleep(2)  # Wait a bit
            status = await self.order_manager.get_order_status(order_id)
            print(f"📊 Order status: {status}")
            
        except Exception as e:
            print(f"❌ Error executing recommendation: {e}")
            
    async def run_dashboard_demo(self):
        """Demonstrate dashboard functionality"""
        print("\n" + "="*80)
        print("📊 DASHBOARD DEMO")
        print("="*80)
        
        print("🌐 Starting web dashboard...")
        print("📱 Dashboard URL: http://localhost:8501")
        print("💡 Features available:")
        print("  • Real-time portfolio monitoring")
        print("  • AI agent status and recommendations")
        print("  • Interactive trading interface")
        print("  • News sentiment analysis")
        print("  • Risk management dashboard")
        print("  • Multi-broker integration")
        
        print("\n🚀 To start the dashboard, run:")
        print("  streamlit run web_dashboard/app.py")
        
    async def run_full_demo(self):
        """Run complete system demonstration"""
        print("\n" + "🚀" * 40)
        print("🎯 COMPLETE MULTI-AGENT TRADING SYSTEM DEMO")
        print("🚀" * 40)
        
        # Initialize system
        await self.initialize()
        
        # Run analysis
        results = await self.run_complete_analysis()
        
        # Execute recommendations (simulation)
        await self.execute_recommendations(results)
        
        # Show dashboard info
        await self.run_dashboard_demo()
        
        # Show system statistics
        await self._show_system_statistics()
        
        print("\n" + "✅" * 40)
        print("🎉 DEMO COMPLETE - System Ready for Production!")
        print("✅" * 40)
        
    async def _show_system_statistics(self):
        """Show system performance statistics"""
        print("\n📈 SYSTEM STATISTICS")
        print("-" * 50)
        
        # Order manager stats
        order_stats = self.order_manager.get_execution_statistics()
        print(f"📊 Total Orders: {order_stats.get('total_orders', 0)}")
        print(f"💰 Daily Volume: ${order_stats.get('daily_volume', 0):,.2f}")
        print(f"🏦 Connected Brokers: {order_stats.get('connected_brokers', 0)}/{order_stats.get('total_brokers', 0)}")
        
        # Portfolio stats
        portfolio = await self._get_portfolio_data()
        if portfolio:
            print(f"💼 Portfolio Value: ${portfolio.get('total_value', 0):,.2f}")
            print(f"💵 Cash Available: ${portfolio.get('cash', 0):,.2f}")
            print(f"📈 Active Positions: {len(portfolio.get('positions', {}))}")
            
        # Agent performance
        if self.coordinator:
            coord_stats = self.coordinator.get_coordination_metrics()
            print(f"🤖 Agent Analyses: {coord_stats.get('total_analyses', 0)}")
            print(f"🎯 Average Confidence: {coord_stats.get('average_confidence', 0):.1%}")
            print(f"🤝 Consensus Rate: {coord_stats.get('consensus_rate', 0):.1%}")


async def main():
    """Main demo function"""
    try:
        # Create and run the complete trading system
        trading_system = CompleteTradingSystem()
        await trading_system.run_full_demo()
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🤖 AI Trading Agent - Complete System Demo")
    print("=" * 60)
    print("🚀 Starting comprehensive demonstration...")
    print("📋 This demo showcases:")
    print("  • Multi-agent AI analysis")
    print("  • Real broker integration")
    print("  • Interactive web dashboard")
    print("  • Risk management")
    print("  • News sentiment analysis")
    print("  • Portfolio optimization")
    print("=" * 60)
    
    asyncio.run(main()) 