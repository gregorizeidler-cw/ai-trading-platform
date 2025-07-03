"""
Example: Multi-Agent Trading System using OpenAI SDK

This example demonstrates how to use multiple AI agents for collaborative trading decisions.
"""

import asyncio
import json
from datetime import datetime
from ai_agents.decision_engine.coordinator import AgentCoordinator
from data_pipeline.collectors.yahoo_finance_collector import YahooFinanceCollector
from data_pipeline.processors.data_processor import DataProcessor


async def main():
    """Main example function"""
    print("🤖 Iniciando Sistema Multi-Agente de Trading com OpenAI SDK")
    print("=" * 60)
    
    # 1. Initialize the agent coordinator
    coordinator = AgentCoordinator()
    
    # 2. Display registered agents
    agents = coordinator.list_agents()
    print(f"\n📋 Agentes Registrados ({len(agents)}):")
    for agent in agents:
        print(f"  • {agent['name']}: {agent['description']}")
        print(f"    Status: {'🟢 Ativo' if agent['active'] else '🔴 Inativo'}")
    
    # 3. Collect sample market data
    print("\n📊 Coletando dados de mercado...")
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    # Simulate market data (in real scenario, use data collectors)
    market_data = {
        "symbols": symbols,
        "timestamp": datetime.utcnow().isoformat(),
        "prices": {
            "AAPL": 150.25,
            "GOOGL": 2800.50,
            "MSFT": 380.75,
            "TSLA": 220.30
        },
        "technical_indicators": {
            "AAPL": {
                "rsi": 65.2,
                "macd": 1.25,
                "macd_signal": 1.10,
                "sma_20": 148.50,
                "sma_50": 145.30
            },
            "GOOGL": {
                "rsi": 58.7,
                "macd": 15.30,
                "macd_signal": 12.80,
                "sma_20": 2750.20,
                "sma_50": 2720.10
            }
        },
        "conditions": {
            "market_sentiment": "neutral",
            "volatility": 0.18,
            "volume_trend": "increasing"
        }
    }
    
    # 4. Simulate portfolio data
    portfolio_data = {
        "total_value": 100000.0,
        "cash": 20000.0,
        "positions": {
            "AAPL": {
                "shares": 100,
                "value": 15025.0,
                "avg_cost": 145.00,
                "volatility": 0.25,
                "beta": 1.2,
                "avg_volume": 50000000
            },
            "GOOGL": {
                "shares": 10,
                "value": 28005.0,
                "avg_cost": 2750.00,
                "volatility": 0.22,
                "beta": 1.1,
                "avg_volume": 1500000
            }
        },
        "volatility": 0.20
    }
    
    # 5. Additional context
    additional_context = {
        "context": "Market showing mixed signals with tech earnings season approaching",
        "proposed_trades": [
            {
                "symbol": "MSFT",
                "action": "BUY",
                "position_size": 50,
                "entry_price": 380.75,
                "stop_loss": 370.00,
                "take_profit": 400.00
            }
        ]
    }
    
    print(f"  ✅ Dados coletados para {len(symbols)} símbolos")
    print(f"  💰 Valor do portfólio: ${portfolio_data['total_value']:,.2f}")
    
    # 6. Make coordinated trading decision
    print("\n🤝 Fazendo decisão colaborativa entre agentes...")
    print("  (Usando OpenAI GPT-4 para coordenação)")
    
    try:
        decision = await coordinator.make_trading_decision(
            market_data=market_data,
            portfolio_data=portfolio_data,
            additional_context=additional_context
        )
        
        if decision["status"] == "success":
            print("  ✅ Decisão gerada com sucesso!")
            
            # Display results
            print_decision_results(decision)
            
        else:
            print(f"  ❌ Erro na decisão: {decision.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"  ❌ Erro durante execução: {e}")
    
    # 7. Display coordination status
    print("\n📈 Status da Coordenação:")
    status = coordinator.get_coordination_status()
    print(f"  • Agentes ativos: {status['active_agents']}/{status['registered_agents']}")
    print(f"  • Limiar de consenso: {status['consensus_threshold']*100:.1f}%")
    
    metrics = status['coordination_metrics']
    print(f"  • Total de decisões: {metrics['total_decisions']}")
    print(f"  • Taxa de consenso: {metrics['agent_agreement_rate']*100:.1f}%")
    print(f"  • Confiança média: {metrics['average_confidence']*100:.1f}%")


def print_decision_results(decision):
    """Print formatted decision results"""
    print("\n" + "="*60)
    print("🎯 RESULTADO DA DECISÃO MULTI-AGENTE")
    print("="*60)
    
    # Processing info
    print(f"⏱️  Tempo de processamento: {decision['processing_time']:.2f}s")
    print(f"🆔 ID da decisão: {decision['decision_id']}")
    
    # Agent responses summary
    agent_responses = decision.get('agent_responses', {})
    print(f"\n👥 Respostas dos Agentes ({len(agent_responses)}):")
    
    for agent_name, response in agent_responses.items():
        if response.get('status') == 'success':
            confidence = response.get('confidence_score', 0.0)
            recommendations_count = len(response.get('recommendations', []))
            print(f"  • {agent_name}: ✅ {confidence*100:.1f}% confiança, {recommendations_count} recomendações")
        else:
            print(f"  • {agent_name}: ❌ Erro - {response.get('error', 'Unknown')}")
    
    # Consensus analysis
    consensus = decision.get('consensus_analysis', {})
    print(f"\n🤝 Análise de Consenso:")
    print(f"  • Consenso alcançado: {'✅ Sim' if consensus.get('consensus_reached') else '❌ Não'}")
    print(f"  • Score de consenso: {consensus.get('consensus_score', 0.0)*100:.1f}%")
    print(f"  • Nível de acordo: {consensus.get('agreement_level', 'unknown').title()}")
    
    # Conflicting recommendations
    conflicts = consensus.get('conflicting_recommendations', [])
    if conflicts:
        print(f"  ⚠️  Conflitos detectados: {len(conflicts)}")
        for conflict in conflicts[:2]:  # Show first 2 conflicts
            print(f"    - {conflict['symbol']}: {', '.join(conflict['conflicting_actions'])}")
    
    # Final decision
    final_decision = decision.get('final_decision', {})
    print(f"\n🎯 Decisão Final:")
    
    # Recommendations
    recommendations = final_decision.get('final_recommendations', [])
    if recommendations:
        print(f"  📋 Recomendações ({len(recommendations)}):")
        for rec in recommendations:
            action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(rec.get('action', 'HOLD'), "⚪")
            symbol = rec.get('symbol', 'N/A')
            action = rec.get('action', 'HOLD')
            confidence = rec.get('confidence', 0.0)
            risk_level = rec.get('risk_level', 'unknown')
            
            print(f"    {action_emoji} {symbol}: {action} (Confiança: {confidence*100:.1f}%, Risco: {risk_level})")
            
            # Show reasoning if available
            reasoning = rec.get('reasoning', '')
            if reasoning and len(reasoning) < 100:
                print(f"      💭 {reasoning}")
    
    # Portfolio adjustments
    adjustments = final_decision.get('portfolio_adjustments', [])
    if adjustments:
        print(f"  🔄 Ajustes de Portfólio ({len(adjustments)}):")
        for adj in adjustments:
            priority_emoji = {"high": "🔥", "medium": "⚠️", "low": "ℹ️"}.get(adj.get('priority', 'low'), "ℹ️")
            print(f"    {priority_emoji} {adj.get('action', 'N/A')}: {adj.get('description', 'N/A')}")
    
    # Coordination summary
    coord_summary = final_decision.get('coordination_summary', {})
    print(f"\n📊 Resumo da Coordenação:")
    print(f"  • Qualidade da decisão: {coord_summary.get('decision_quality', 'unknown').title()}")
    print(f"  • Nível de consenso: {coord_summary.get('consensus_level', 'unknown').title()}")
    print(f"  • Avaliação de risco: {coord_summary.get('risk_assessment', 'unknown').title()}")
    print(f"  • Score de confiança: {coord_summary.get('confidence_score', 0.0)*100:.1f}%")
    
    # Key factors
    key_factors = coord_summary.get('key_factors', [])
    if key_factors:
        print(f"  🔑 Fatores principais: {', '.join(key_factors[:3])}")
    
    # Execution plan
    execution_plan = final_decision.get('execution_plan', {})
    immediate_actions = execution_plan.get('immediate_actions', [])
    if immediate_actions:
        print(f"\n⚡ Ações Imediatas:")
        for action in immediate_actions[:3]:
            print(f"  • {action}")
    
    # Model info
    if final_decision.get('coordination_method') == 'llm_synthesis':
        model_used = final_decision.get('model_used', 'unknown')
        tokens_used = final_decision.get('tokens_used', 0)
        print(f"\n🤖 Coordenação via LLM:")
        print(f"  • Modelo: {model_used}")
        print(f"  • Tokens utilizados: {tokens_used:,}")


async def demo_individual_agents():
    """Demonstrate individual agent capabilities"""
    print("\n" + "="*60)
    print("🔬 DEMONSTRAÇÃO DE AGENTES INDIVIDUAIS")
    print("="*60)
    
    from ai_agents.agents.market_analyst import MarketAnalystAgent
    from ai_agents.agents.risk_manager import RiskManagerAgent
    
    # Test Market Analyst
    print("\n📊 Testando Market Analyst Agent...")
    market_analyst = MarketAnalystAgent()
    
    sample_data = {
        "market_data": {
            "AAPL": {"price": 150.25, "volume": 50000000},
            "recent_prices": [148.5, 149.2, 150.25]
        },
        "technical_indicators": {
            "rsi": 65.2,
            "macd": 1.25,
            "sma_20": 148.50
        }
    }
    
    try:
        analyst_result = await market_analyst.process_request({"data": sample_data})
        if analyst_result.get("status") == "success":
            print("  ✅ Análise técnica concluída")
            confidence = analyst_result.get("confidence_score", 0.0)
            print(f"  📈 Confiança: {confidence*100:.1f}%")
            
            recommendations = analyst_result.get("recommendations", [])
            print(f"  💡 Recomendações: {len(recommendations)}")
        else:
            print(f"  ❌ Erro: {analyst_result.get('error', 'Unknown')}")
    except Exception as e:
        print(f"  ❌ Erro durante teste: {e}")
    
    # Test Risk Manager
    print("\n⚠️ Testando Risk Manager Agent...")
    risk_manager = RiskManagerAgent()
    
    risk_data = {
        "portfolio_data": {
            "total_value": 100000.0,
            "positions": {
                "AAPL": {"value": 15000, "volatility": 0.25},
                "GOOGL": {"value": 28000, "volatility": 0.22}
            },
            "volatility": 0.20
        },
        "market_conditions": {
            "volatility": 0.18,
            "sentiment": "neutral"
        }
    }
    
    try:
        risk_result = await risk_manager.process_request({"data": risk_data})
        if risk_result.get("status") == "success":
            print("  ✅ Análise de risco concluída")
            confidence = risk_result.get("confidence_score", 0.0)
            print(f"  ⚖️ Confiança: {confidence*100:.1f}%")
            
            recommendations = risk_result.get("recommendations", [])
            print(f"  🛡️ Recomendações de risco: {len(recommendations)}")
        else:
            print(f"  ❌ Erro: {risk_result.get('error', 'Unknown')}")
    except Exception as e:
        print(f"  ❌ Erro durante teste: {e}")


if __name__ == "__main__":
    print("🚀 Exemplo: Sistema Multi-Agente de Trading com OpenAI")
    print("Certifique-se de ter configurado sua OPENAI_API_KEY no ambiente\n")
    
    # Run main example
    asyncio.run(main())
    
    # Run individual agent demo
    asyncio.run(demo_individual_agents())
    
    print("\n" + "="*60)
    print("✨ Exemplo concluído!")
    print("Para usar em produção:")
    print("1. Configure todas as API keys necessárias")
    print("2. Implemente coleta de dados em tempo real")
    print("3. Integre com sistema de execução de ordens")
    print("4. Configure monitoramento e alertas")
    print("="*60) 