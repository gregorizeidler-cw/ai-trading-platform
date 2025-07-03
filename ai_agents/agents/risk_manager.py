"""
Risk Manager Agent - Specialized in portfolio risk assessment and management.
"""

import json
from typing import Dict, List, Any
from datetime import datetime
from loguru import logger
from .base_agent import BaseAgent


class RiskManagerAgent(BaseAgent):
    """Agent specialized in risk management using OpenAI GPT"""
    
    def __init__(self):
        super().__init__(
            name="Risk Manager",
            description="Specialized in portfolio risk assessment and management"
        )
        self.risk_metrics = [
            "portfolio_var",
            "position_sizing",
            "correlation_analysis",
            "drawdown_analysis",
            "volatility_assessment"
        ]
        self.max_position_size = 0.10  # 10% max per position
        self.max_portfolio_risk = 0.02  # 2% max portfolio risk per trade
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk analysis"""
        try:
            portfolio_data = data.get("portfolio_data", {})
            market_conditions = data.get("market_conditions", {})
            proposed_trades = data.get("proposed_trades", [])
            
            # Use OpenAI to assess portfolio risk
            risk_assessment = await self.llm_client.assess_portfolio_risk(
                portfolio_data=portfolio_data,
                market_conditions=market_conditions
            )
            
            if "error" in risk_assessment:
                raise Exception(risk_assessment["error"])
            
            # Parse the JSON response
            risk_json = json.loads(risk_assessment["risk_assessment"])
            
            # Add our specialized risk calculations
            enhanced_risk_analysis = {
                **risk_json,
                "position_risk_analysis": self._analyze_position_risks(portfolio_data),
                "correlation_matrix": self._calculate_correlations(portfolio_data),
                "var_analysis": self._calculate_var(portfolio_data, market_conditions),
                "stress_test_extended": self._perform_stress_tests(portfolio_data),
                "liquidity_assessment": self._assess_liquidity(portfolio_data),
                "concentration_risk": self._assess_concentration_risk(portfolio_data),
                "proposed_trades_risk": self._assess_proposed_trades_risk(proposed_trades, portfolio_data),
                "confidence_score": self._calculate_risk_confidence(risk_json, portfolio_data),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "tokens_used": risk_assessment.get("tokens_used", 0)
            }
            
            return enhanced_risk_analysis
            
        except Exception as e:
            logger.error(f"Risk analysis error: {e}")
            return {
                "error": str(e),
                "confidence_score": 0.0,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if "error" in analysis:
            return recommendations
        
        try:
            # Portfolio-level recommendations
            portfolio_recommendations = self._get_portfolio_recommendations(analysis)
            recommendations.extend(portfolio_recommendations)
            
            # Position-level recommendations
            position_recommendations = self._get_position_recommendations(analysis)
            recommendations.extend(position_recommendations)
            
            # Trade-level recommendations
            trade_recommendations = self._get_trade_recommendations(analysis)
            recommendations.extend(trade_recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating risk recommendations: {e}")
            return []
    
    def _analyze_position_risks(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual position risks"""
        positions = portfolio_data.get("positions", {})
        position_risks = {}
        
        for symbol, position in positions.items():
            position_value = position.get("value", 0)
            portfolio_value = portfolio_data.get("total_value", 1)
            
            position_risks[symbol] = {
                "position_size_pct": position_value / portfolio_value if portfolio_value > 0 else 0,
                "risk_contribution": self._calculate_position_risk_contribution(position),
                "liquidity_score": self._assess_position_liquidity(position),
                "volatility": position.get("volatility", 0.0),
                "beta": position.get("beta", 1.0),
                "risk_rating": self._rate_position_risk(position)
            }
        
        return position_risks
    
    def _calculate_correlations(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlation matrix for portfolio positions"""
        positions = portfolio_data.get("positions", {})
        symbols = list(positions.keys())
        
        # Simplified correlation calculation
        # In a real implementation, this would use historical price data
        correlations = {}
        for symbol1 in symbols:
            correlations[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    correlations[symbol1][symbol2] = 1.0
                else:
                    # Placeholder correlation - would be calculated from historical data
                    correlations[symbol1][symbol2] = 0.3  # Assume moderate correlation
        
        return correlations
    
    def _calculate_var(self, portfolio_data: Dict[str, Any], market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Value at Risk (VaR) metrics"""
        portfolio_value = portfolio_data.get("total_value", 0)
        portfolio_volatility = portfolio_data.get("volatility", 0.15)
        
        # Calculate VaR at different confidence levels
        var_95 = portfolio_value * portfolio_volatility * 1.645  # 95% confidence
        var_99 = portfolio_value * portfolio_volatility * 2.326  # 99% confidence
        
        return {
            "var_95_1day": var_95,
            "var_99_1day": var_99,
            "var_95_1week": var_95 * (7 ** 0.5),
            "var_99_1week": var_99 * (7 ** 0.5),
            "expected_shortfall": var_95 * 1.2,  # Simplified ES calculation
            "portfolio_volatility": portfolio_volatility
        }
    
    def _perform_stress_tests(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform extended stress tests"""
        portfolio_value = portfolio_data.get("total_value", 0)
        
        stress_scenarios = {
            "market_crash_20pct": portfolio_value * -0.20,
            "market_crash_30pct": portfolio_value * -0.30,
            "sector_rotation": portfolio_value * -0.15,
            "interest_rate_shock": portfolio_value * -0.10,
            "currency_crisis": portfolio_value * -0.12,
            "liquidity_crisis": portfolio_value * -0.25,
            "black_swan_event": portfolio_value * -0.40
        }
        
        return stress_scenarios
    
    def _assess_liquidity(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess portfolio liquidity"""
        positions = portfolio_data.get("positions", {})
        
        total_value = portfolio_data.get("total_value", 0)
        liquid_value = 0
        
        for symbol, position in positions.items():
            # Simplified liquidity assessment
            position_value = position.get("value", 0)
            avg_volume = position.get("avg_volume", 0)
            
            if avg_volume > 1000000:  # High volume
                liquid_value += position_value
            elif avg_volume > 100000:  # Medium volume
                liquid_value += position_value * 0.8
            else:  # Low volume
                liquid_value += position_value * 0.5
        
        liquidity_ratio = liquid_value / total_value if total_value > 0 else 0
        
        return {
            "liquidity_ratio": liquidity_ratio,
            "liquid_value": liquid_value,
            "illiquid_value": total_value - liquid_value,
            "liquidity_score": self._calculate_liquidity_score(liquidity_ratio)
        }
    
    def _assess_concentration_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess concentration risk"""
        positions = portfolio_data.get("positions", {})
        total_value = portfolio_data.get("total_value", 0)
        
        if total_value == 0:
            return {"concentration_score": 0, "top_positions": []}
        
        # Calculate position weights
        position_weights = []
        for symbol, position in positions.items():
            weight = position.get("value", 0) / total_value
            position_weights.append({"symbol": symbol, "weight": weight})
        
        # Sort by weight
        position_weights.sort(key=lambda x: x["weight"], reverse=True)
        
        # Calculate concentration metrics
        top_5_concentration = sum(pos["weight"] for pos in position_weights[:5])
        top_10_concentration = sum(pos["weight"] for pos in position_weights[:10])
        
        # Herfindahl-Hirschman Index (HHI)
        hhi = sum(pos["weight"] ** 2 for pos in position_weights)
        
        return {
            "hhi_index": hhi,
            "top_5_concentration": top_5_concentration,
            "top_10_concentration": top_10_concentration,
            "concentration_score": self._calculate_concentration_score(hhi),
            "top_positions": position_weights[:10]
        }
    
    def _assess_proposed_trades_risk(self, proposed_trades: List[Dict[str, Any]], portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of proposed trades"""
        if not proposed_trades:
            return {"total_risk_impact": 0, "trade_risks": []}
        
        portfolio_value = portfolio_data.get("total_value", 0)
        trade_risks = []
        total_risk_impact = 0
        
        for trade in proposed_trades:
            trade_value = trade.get("position_size", 0) * trade.get("entry_price", 0)
            risk_pct = trade_value / portfolio_value if portfolio_value > 0 else 0
            
            trade_risk = {
                "symbol": trade.get("symbol", ""),
                "action": trade.get("action", ""),
                "position_size_pct": risk_pct,
                "risk_rating": self._rate_trade_risk(trade, risk_pct),
                "stop_loss_distance": self._calculate_stop_loss_distance(trade),
                "risk_reward_ratio": self._calculate_risk_reward_ratio(trade)
            }
            
            trade_risks.append(trade_risk)
            total_risk_impact += risk_pct
        
        return {
            "total_risk_impact": total_risk_impact,
            "trade_risks": trade_risks,
            "exceeds_risk_limits": total_risk_impact > self.max_portfolio_risk
        }
    
    def _get_portfolio_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get portfolio-level recommendations"""
        recommendations = []
        
        # Check overall risk level
        risk_level = analysis.get("overall_risk_level", "medium")
        if risk_level == "high":
            recommendations.append({
                "type": "portfolio_adjustment",
                "priority": "high",
                "action": "reduce_overall_risk",
                "description": "Portfolio risk level is high - consider reducing position sizes",
                "specific_actions": ["Reduce position sizes", "Increase cash allocation", "Add hedging positions"]
            })
        
        # Check concentration risk
        concentration = analysis.get("concentration_risk", {})
        if concentration.get("top_5_concentration", 0) > 0.6:
            recommendations.append({
                "type": "diversification",
                "priority": "medium",
                "action": "improve_diversification",
                "description": "High concentration risk detected - diversify holdings",
                "specific_actions": ["Reduce largest positions", "Add positions in different sectors"]
            })
        
        return recommendations
    
    def _get_position_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get position-level recommendations"""
        recommendations = []
        
        position_risks = analysis.get("position_risk_analysis", {})
        
        for symbol, risk_data in position_risks.items():
            if risk_data.get("position_size_pct", 0) > self.max_position_size:
                recommendations.append({
                    "type": "position_sizing",
                    "priority": "high",
                    "action": "reduce_position",
                    "symbol": symbol,
                    "description": f"Position size exceeds maximum limit for {symbol}",
                    "current_size": risk_data.get("position_size_pct", 0),
                    "recommended_size": self.max_position_size
                })
        
        return recommendations
    
    def _get_trade_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get trade-level recommendations"""
        recommendations = []
        
        proposed_trades_risk = analysis.get("proposed_trades_risk", {})
        
        if proposed_trades_risk.get("exceeds_risk_limits", False):
            recommendations.append({
                "type": "trade_adjustment",
                "priority": "high",
                "action": "reduce_trade_sizes",
                "description": "Proposed trades exceed portfolio risk limits",
                "total_risk_impact": proposed_trades_risk.get("total_risk_impact", 0),
                "max_allowed_risk": self.max_portfolio_risk
            })
        
        return recommendations
    
    # Helper methods
    def _calculate_position_risk_contribution(self, position: Dict[str, Any]) -> float:
        """Calculate position's contribution to portfolio risk"""
        return position.get("volatility", 0.0) * position.get("beta", 1.0)
    
    def _assess_position_liquidity(self, position: Dict[str, Any]) -> float:
        """Assess individual position liquidity"""
        avg_volume = position.get("avg_volume", 0)
        if avg_volume > 1000000:
            return 0.9
        elif avg_volume > 100000:
            return 0.6
        else:
            return 0.3
    
    def _rate_position_risk(self, position: Dict[str, Any]) -> str:
        """Rate position risk level"""
        volatility = position.get("volatility", 0.0)
        if volatility > 0.4:
            return "high"
        elif volatility > 0.2:
            return "medium"
        else:
            return "low"
    
    def _calculate_liquidity_score(self, liquidity_ratio: float) -> str:
        """Calculate liquidity score"""
        if liquidity_ratio > 0.8:
            return "high"
        elif liquidity_ratio > 0.5:
            return "medium"
        else:
            return "low"
    
    def _calculate_concentration_score(self, hhi: float) -> str:
        """Calculate concentration score based on HHI"""
        if hhi > 0.25:
            return "high"
        elif hhi > 0.15:
            return "medium"
        else:
            return "low"
    
    def _rate_trade_risk(self, trade: Dict[str, Any], risk_pct: float) -> str:
        """Rate individual trade risk"""
        if risk_pct > 0.05:  # 5%
            return "high"
        elif risk_pct > 0.02:  # 2%
            return "medium"
        else:
            return "low"
    
    def _calculate_stop_loss_distance(self, trade: Dict[str, Any]) -> float:
        """Calculate stop loss distance as percentage"""
        entry_price = trade.get("entry_price", 0)
        stop_loss = trade.get("stop_loss", 0)
        
        if entry_price > 0 and stop_loss > 0:
            return abs(entry_price - stop_loss) / entry_price
        return 0.0
    
    def _calculate_risk_reward_ratio(self, trade: Dict[str, Any]) -> float:
        """Calculate risk-reward ratio"""
        entry_price = trade.get("entry_price", 0)
        stop_loss = trade.get("stop_loss", 0)
        take_profit = trade.get("take_profit", 0)
        
        if entry_price > 0 and stop_loss > 0 and take_profit > 0:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            return reward / risk if risk > 0 else 0
        return 0.0
    
    def _calculate_risk_confidence(self, risk_json: Dict[str, Any], portfolio_data: Dict[str, Any]) -> float:
        """Calculate confidence score for risk analysis"""
        base_confidence = 0.7
        
        # Adjust based on portfolio size
        portfolio_value = portfolio_data.get("total_value", 0)
        if portfolio_value > 100000:
            base_confidence += 0.1
        
        # Adjust based on data quality
        positions_count = len(portfolio_data.get("positions", {}))
        if positions_count > 5:
            base_confidence += 0.1
        
        # Adjust based on risk score clarity
        risk_score = risk_json.get("risk_score", 0.5)
        if risk_score != 0.5:  # Not neutral
            base_confidence += 0.1
        
        return min(base_confidence, 1.0) 