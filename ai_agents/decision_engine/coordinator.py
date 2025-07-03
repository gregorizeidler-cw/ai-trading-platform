"""
Agent Coordinator - Manages multiple AI agents for collaborative trading decisions.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

from ai_agents.agents.base_agent import BaseAgent
from ai_agents.agents.market_analyst import MarketAnalystAgent
from ai_agents.agents.risk_manager import RiskManagerAgent
from ai_agents.llm.openai_client import OpenAIClient


class AgentCoordinator:
    """Coordinates multiple AI agents for collaborative trading decisions"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.llm_client = OpenAIClient()
        self.consensus_threshold = 0.7  # 70% agreement threshold
        self.active = True
        
        # Initialize default agents
        self._initialize_default_agents()
        
        # Coordination metrics
        self.coordination_metrics = {
            "total_decisions": 0,
            "consensus_decisions": 0,
            "average_confidence": 0.0,
            "agent_agreement_rate": 0.0
        }
    
    def _initialize_default_agents(self):
        """Initialize default set of agents"""
        # Market Analyst Agent
        market_analyst = MarketAnalystAgent()
        self.register_agent("market_analyst", market_analyst)
        
        # Risk Manager Agent
        risk_manager = RiskManagerAgent()
        self.register_agent("risk_manager", risk_manager)
        
        logger.info("Default agents initialized")
    
    def register_agent(self, name: str, agent: BaseAgent):
        """Register a new agent"""
        self.agents[name] = agent
        logger.info(f"Agent '{name}' registered: {agent.description}")
    
    def remove_agent(self, name: str):
        """Remove an agent"""
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Agent '{name}' removed")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self.agents.get(name)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        return [
            {
                "name": name,
                "description": agent.description,
                "active": agent.active,
                "performance": agent.performance_metrics
            }
            for name, agent in self.agents.items()
        ]
    
    async def make_trading_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        additional_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Coordinate agents to make collaborative trading decision"""
        try:
            start_time = datetime.utcnow()
            
            # Prepare data for agents
            agent_data = {
                "market_data": market_data,
                "portfolio_data": portfolio_data,
                "technical_indicators": market_data.get("technical_indicators", {}),
                "market_conditions": market_data.get("conditions", {}),
                "proposed_trades": additional_context.get("proposed_trades", []) if additional_context else [],
                "context": additional_context.get("context", "") if additional_context else ""
            }
            
            # Collect responses from all active agents
            agent_responses = await self._collect_agent_responses(agent_data)
            
            # Analyze agent consensus
            consensus_analysis = await self._analyze_consensus(agent_responses)
            
            # Generate final decision using OpenAI coordination
            final_decision = await self._generate_coordinated_decision(
                agent_responses,
                consensus_analysis,
                agent_data
            )
            
            # Update coordination metrics
            self._update_coordination_metrics(agent_responses, consensus_analysis)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "decision_id": self._generate_decision_id(),
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time": processing_time,
                "agent_responses": agent_responses,
                "consensus_analysis": consensus_analysis,
                "final_decision": final_decision,
                "coordination_metrics": self.coordination_metrics,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in coordinated decision making: {e}")
            return {
                "decision_id": self._generate_decision_id(),
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "status": "error"
            }
    
    async def _collect_agent_responses(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect responses from all active agents"""
        agent_responses = {}
        
        # Create tasks for parallel execution
        tasks = []
        agent_names = []
        
        for name, agent in self.agents.items():
            if agent.active:
                task = agent.process_request({"data": data})
                tasks.append(task)
                agent_names.append(name)
        
        # Execute all agent tasks in parallel
        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, response in enumerate(responses):
                agent_name = agent_names[i]
                if isinstance(response, Exception):
                    logger.error(f"Agent {agent_name} failed: {response}")
                    agent_responses[agent_name] = {
                        "status": "error",
                        "error": str(response)
                    }
                else:
                    agent_responses[agent_name] = response
        
        return agent_responses
    
    async def _analyze_consensus(self, agent_responses: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consensus among agent responses"""
        successful_responses = {
            name: response for name, response in agent_responses.items()
            if response.get("status") == "success"
        }
        
        if not successful_responses:
            return {
                "consensus_reached": False,
                "consensus_score": 0.0,
                "agreement_level": "no_consensus",
                "conflicting_recommendations": []
            }
        
        # Collect all recommendations
        all_recommendations = []
        for name, response in successful_responses.items():
            recommendations = response.get("recommendations", [])
            for rec in recommendations:
                rec["agent"] = name
                all_recommendations.append(rec)
        
        # Group recommendations by symbol
        symbol_recommendations = {}
        for rec in all_recommendations:
            symbol = rec.get("symbol", "portfolio")
            if symbol not in symbol_recommendations:
                symbol_recommendations[symbol] = []
            symbol_recommendations[symbol].append(rec)
        
        # Calculate consensus for each symbol
        consensus_results = {}
        overall_agreement = 0.0
        
        for symbol, recs in symbol_recommendations.items():
            symbol_consensus = self._calculate_symbol_consensus(recs)
            consensus_results[symbol] = symbol_consensus
            overall_agreement += symbol_consensus["agreement_score"]
        
        # Calculate overall consensus
        overall_agreement = overall_agreement / len(consensus_results) if consensus_results else 0.0
        consensus_reached = overall_agreement >= self.consensus_threshold
        
        return {
            "consensus_reached": consensus_reached,
            "consensus_score": overall_agreement,
            "agreement_level": self._categorize_agreement_level(overall_agreement),
            "symbol_consensus": consensus_results,
            "conflicting_recommendations": self._identify_conflicts(symbol_recommendations)
        }
    
    def _calculate_symbol_consensus(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus for a specific symbol"""
        if not recommendations:
            return {"agreement_score": 0.0, "consensus_action": "hold"}
        
        # Count actions
        action_counts = {}
        confidence_sum = 0.0
        
        for rec in recommendations:
            action = rec.get("action", "hold").lower()
            confidence = rec.get("confidence", 0.0)
            
            if action not in action_counts:
                action_counts[action] = {"count": 0, "total_confidence": 0.0}
            
            action_counts[action]["count"] += 1
            action_counts[action]["total_confidence"] += confidence
            confidence_sum += confidence
        
        # Find consensus action
        max_count = max(action_counts.values(), key=lambda x: x["count"])["count"]
        total_recommendations = len(recommendations)
        
        consensus_actions = [
            action for action, data in action_counts.items()
            if data["count"] == max_count
        ]
        
        # Calculate agreement score
        agreement_score = max_count / total_recommendations
        
        # Average confidence
        avg_confidence = confidence_sum / total_recommendations if total_recommendations > 0 else 0.0
        
        return {
            "agreement_score": agreement_score,
            "consensus_action": consensus_actions[0] if len(consensus_actions) == 1 else "conflicted",
            "average_confidence": avg_confidence,
            "action_distribution": action_counts,
            "total_recommendations": total_recommendations
        }
    
    def _identify_conflicts(self, symbol_recommendations: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify conflicting recommendations"""
        conflicts = []
        
        for symbol, recs in symbol_recommendations.items():
            if len(recs) > 1:
                actions = [rec.get("action", "hold").lower() for rec in recs]
                unique_actions = set(actions)
                
                if len(unique_actions) > 1:
                    # There's a conflict
                    conflict_details = {
                        "symbol": symbol,
                        "conflicting_actions": list(unique_actions),
                        "agent_positions": [
                            {
                                "agent": rec.get("agent", "unknown"),
                                "action": rec.get("action", "hold"),
                                "confidence": rec.get("confidence", 0.0),
                                "reasoning": rec.get("reasoning", "")
                            }
                            for rec in recs
                        ]
                    }
                    conflicts.append(conflict_details)
        
        return conflicts
    
    async def _generate_coordinated_decision(
        self,
        agent_responses: Dict[str, Any],
        consensus_analysis: Dict[str, Any],
        original_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final coordinated decision using OpenAI"""
        try:
            # Prepare coordination prompt
            coordination_prompt = self._build_coordination_prompt(
                agent_responses,
                consensus_analysis,
                original_data
            )
            
            # Use OpenAI to coordinate final decision
            response = await self.llm_client.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a master trading coordinator that synthesizes input from multiple AI agents to make final trading decisions. Consider consensus, conflicts, and risk factors."
                    },
                    {
                        "role": "user",
                        "content": coordination_prompt
                    }
                ],
                max_tokens=4096,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            coordinated_decision = json.loads(response.choices[0].message.content)
            
            # Add coordination metadata
            coordinated_decision["coordination_method"] = "llm_synthesis"
            coordinated_decision["tokens_used"] = response.usage.total_tokens
            coordinated_decision["model_used"] = "gpt-4-turbo-preview"
            
            return coordinated_decision
            
        except Exception as e:
            logger.error(f"Error in LLM coordination: {e}")
            
            # Fallback to rule-based coordination
            return self._fallback_coordination(agent_responses, consensus_analysis)
    
    def _build_coordination_prompt(
        self,
        agent_responses: Dict[str, Any],
        consensus_analysis: Dict[str, Any],
        original_data: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM coordination"""
        return f"""
        You are coordinating multiple AI trading agents to make final trading decisions.
        
        Agent Responses:
        {json.dumps(agent_responses, indent=2)}
        
        Consensus Analysis:
        {json.dumps(consensus_analysis, indent=2)}
        
        Market Data Context:
        {json.dumps(original_data.get("market_data", {}), indent=2)}
        
        Portfolio Context:
        {json.dumps(original_data.get("portfolio_data", {}), indent=2)}
        
        Please provide a coordinated decision in the following JSON format:
        {{
            "final_recommendations": [
                {{
                    "symbol": "AAPL",
                    "action": "BUY/SELL/HOLD",
                    "confidence": 0.85,
                    "position_size": 0.05,
                    "entry_price": 150.00,
                    "stop_loss": 145.00,
                    "take_profit": 160.00,
                    "reasoning": "Coordinated reasoning from all agents",
                    "risk_level": "low/medium/high",
                    "time_horizon": "short/medium/long",
                    "agent_consensus": "high/medium/low"
                }}
            ],
            "portfolio_adjustments": [
                {{
                    "action": "rebalance/hedge/reduce_risk",
                    "description": "Portfolio-level adjustment",
                    "priority": "high/medium/low"
                }}
            ],
            "coordination_summary": {{
                "decision_quality": "high/medium/low",
                "consensus_level": "high/medium/low/conflicted",
                "risk_assessment": "low/medium/high",
                "confidence_score": 0.80,
                "key_factors": ["factor1", "factor2", "factor3"]
            }},
            "execution_plan": {{
                "immediate_actions": ["action1", "action2"],
                "monitoring_requirements": ["requirement1", "requirement2"],
                "exit_conditions": ["condition1", "condition2"]
            }}
        }}
        """
    
    def _fallback_coordination(
        self,
        agent_responses: Dict[str, Any],
        consensus_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback rule-based coordination when LLM fails"""
        logger.info("Using fallback rule-based coordination")
        
        # Simple rule-based coordination
        final_recommendations = []
        
        if consensus_analysis.get("consensus_reached", False):
            # Use consensus recommendations
            symbol_consensus = consensus_analysis.get("symbol_consensus", {})
            
            for symbol, consensus_data in symbol_consensus.items():
                if symbol != "portfolio":
                    recommendation = {
                        "symbol": symbol,
                        "action": consensus_data.get("consensus_action", "hold").upper(),
                        "confidence": consensus_data.get("average_confidence", 0.0),
                        "reasoning": f"Consensus decision with {consensus_data.get('agreement_score', 0.0):.2f} agreement",
                        "risk_level": "medium",
                        "agent_consensus": "high" if consensus_data.get("agreement_score", 0.0) > 0.8 else "medium"
                    }
                    final_recommendations.append(recommendation)
        else:
            # Conservative approach when no consensus
            final_recommendations.append({
                "symbol": "portfolio",
                "action": "HOLD",
                "confidence": 0.5,
                "reasoning": "No consensus reached among agents - maintaining conservative position",
                "risk_level": "low",
                "agent_consensus": "low"
            })
        
        return {
            "final_recommendations": final_recommendations,
            "portfolio_adjustments": [],
            "coordination_summary": {
                "decision_quality": "medium",
                "consensus_level": "high" if consensus_analysis.get("consensus_reached", False) else "low",
                "risk_assessment": "medium",
                "confidence_score": consensus_analysis.get("consensus_score", 0.0),
                "key_factors": ["rule_based_coordination", "consensus_analysis"]
            },
            "execution_plan": {
                "immediate_actions": ["monitor_positions"],
                "monitoring_requirements": ["track_consensus_changes"],
                "exit_conditions": ["significant_consensus_change"]
            }
        }
    
    def _categorize_agreement_level(self, agreement_score: float) -> str:
        """Categorize agreement level"""
        if agreement_score >= 0.8:
            return "high"
        elif agreement_score >= 0.6:
            return "medium"
        elif agreement_score >= 0.4:
            return "low"
        else:
            return "conflicted"
    
    def _update_coordination_metrics(
        self,
        agent_responses: Dict[str, Any],
        consensus_analysis: Dict[str, Any]
    ):
        """Update coordination metrics"""
        self.coordination_metrics["total_decisions"] += 1
        
        if consensus_analysis.get("consensus_reached", False):
            self.coordination_metrics["consensus_decisions"] += 1
        
        # Update average confidence
        total_confidence = 0.0
        confidence_count = 0
        
        for response in agent_responses.values():
            if response.get("status") == "success":
                confidence = response.get("confidence_score", 0.0)
                total_confidence += confidence
                confidence_count += 1
        
        if confidence_count > 0:
            current_avg = self.coordination_metrics["average_confidence"]
            total_decisions = self.coordination_metrics["total_decisions"]
            
            self.coordination_metrics["average_confidence"] = (
                (current_avg * (total_decisions - 1) + total_confidence / confidence_count) / total_decisions
            )
        
        # Update agreement rate
        total_decisions = self.coordination_metrics["total_decisions"]
        consensus_decisions = self.coordination_metrics["consensus_decisions"]
        
        self.coordination_metrics["agent_agreement_rate"] = (
            consensus_decisions / total_decisions if total_decisions > 0 else 0.0
        )
    
    def _generate_decision_id(self) -> str:
        """Generate unique decision ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"decision_{timestamp}_{len(self.agents)}_agents"
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get coordination status and metrics"""
        return {
            "active": self.active,
            "registered_agents": len(self.agents),
            "active_agents": sum(1 for agent in self.agents.values() if agent.active),
            "consensus_threshold": self.consensus_threshold,
            "coordination_metrics": self.coordination_metrics,
            "agents_status": [
                {
                    "name": name,
                    "active": agent.active,
                    "performance": agent.performance_metrics
                }
                for name, agent in self.agents.items()
            ]
        }
    
    def set_consensus_threshold(self, threshold: float):
        """Set consensus threshold"""
        if 0.0 <= threshold <= 1.0:
            self.consensus_threshold = threshold
            logger.info(f"Consensus threshold set to {threshold}")
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
    
    def reset_metrics(self):
        """Reset coordination metrics"""
        self.coordination_metrics = {
            "total_decisions": 0,
            "consensus_decisions": 0,
            "average_confidence": 0.0,
            "agent_agreement_rate": 0.0
        }
        logger.info("Coordination metrics reset") 