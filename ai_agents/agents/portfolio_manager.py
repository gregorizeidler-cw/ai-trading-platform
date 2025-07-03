"""
Portfolio Manager Agent - Specialized in portfolio optimization and asset allocation.
"""

import json
from typing import Dict, List, Any
from datetime import datetime
from loguru import logger
from .base_agent import BaseAgent


class PortfolioManagerAgent(BaseAgent):
    """Agent specialized in portfolio management using OpenAI GPT"""
    
    def __init__(self):
        super().__init__(
            name="Portfolio Manager",
            description="Specialized in portfolio optimization and asset allocation"
        )
        self.allocation_strategies = [
            "conservative", "moderate", "aggressive", "growth", "income"
        ]
        self.rebalancing_triggers = {
            "drift_threshold": 0.05,  # 5% drift
            "time_based": "quarterly",
            "volatility_threshold": 0.25
        }
        self.target_allocations = {
            "conservative": {"stocks": 0.4, "bonds": 0.5, "cash": 0.1},
            "moderate": {"stocks": 0.6, "bonds": 0.3, "cash": 0.1},
            "aggressive": {"stocks": 0.8, "bonds": 0.15, "cash": 0.05}
        }
    
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive portfolio analysis"""
        try:
            portfolio_data = data.get("portfolio_data", {})
            market_conditions = data.get("market_conditions", {})
            risk_profile = data.get("risk_profile", "moderate")
            
            if not portfolio_data:
                return {
                    "portfolio_health": "unknown",
                    "optimization_score": 0.0,
                    "confidence_score": 0.0,
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            
            # Use OpenAI for portfolio analysis
            portfolio_analysis = await self.llm_client.assess_portfolio_risk(
                portfolio_data=portfolio_data,
                market_conditions=market_conditions
            )
            
            if "error" in portfolio_analysis:
                raise Exception(portfolio_analysis["error"])
            
            # Parse the JSON response
            analysis_json = json.loads(portfolio_analysis["risk_assessment"])
            
            # Add our specialized portfolio analysis
            enhanced_analysis = {
                **analysis_json,
                "allocation_analysis": self._analyze_current_allocation(portfolio_data),
                "performance_metrics": self._calculate_performance_metrics(portfolio_data),
                "rebalancing_needs": self._assess_rebalancing_needs(portfolio_data, risk_profile),
                "diversification_analysis": self._analyze_diversification(portfolio_data),
                "optimization_opportunities": self._identify_optimization_opportunities(
                    portfolio_data, market_conditions, risk_profile
                ),
                "sector_allocation": self._analyze_sector_allocation(portfolio_data),
                "liquidity_analysis": self._analyze_portfolio_liquidity(portfolio_data),
                "tax_efficiency": self._assess_tax_efficiency(portfolio_data),
                "confidence_score": self._calculate_portfolio_confidence(analysis_json, portfolio_data),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "tokens_used": portfolio_analysis.get("tokens_used", 0)
            }
            
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"Portfolio analysis error: {e}")
            return {
                "error": str(e),
                "confidence_score": 0.0,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate portfolio management recommendations"""
        recommendations = []
        
        if "error" in analysis:
            return recommendations
        
        try:
            # Rebalancing recommendations
            rebalancing_recs = self._get_rebalancing_recommendations(analysis)
            recommendations.extend(rebalancing_recs)
            
            # Optimization recommendations
            optimization_recs = self._get_optimization_recommendations(analysis)
            recommendations.extend(optimization_recs)
            
            # Risk management recommendations
            risk_recs = self._get_risk_management_recommendations(analysis)
            recommendations.extend(risk_recs)
            
            # Tax efficiency recommendations
            tax_recs = self._get_tax_efficiency_recommendations(analysis)
            recommendations.extend(tax_recs)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating portfolio recommendations: {e}")
            return []
    
    def _analyze_current_allocation(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current portfolio allocation"""
        positions = portfolio_data.get("positions", {})
        total_value = portfolio_data.get("total_value", 0)
        cash = portfolio_data.get("cash", 0)
        
        if total_value == 0:
            return {"error": "No portfolio value data"}
        
        # Calculate asset class allocation
        allocation = {
            "stocks": 0.0,
            "bonds": 0.0,
            "cash": cash / total_value,
            "other": 0.0
        }
        
        # Simplified asset classification
        stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]
        bond_symbols = ["TLT", "IEF", "SHY", "AGG", "BND"]
        
        for symbol, position in positions.items():
            position_value = position.get("value", 0)
            weight = position_value / total_value
            
            if symbol in stock_symbols:
                allocation["stocks"] += weight
            elif symbol in bond_symbols:
                allocation["bonds"] += weight
            else:
                allocation["other"] += weight
        
        # Calculate allocation drift from target
        target_allocation = self.target_allocations.get("moderate", {})
        drift = {}
        for asset_class, current_weight in allocation.items():
            target_weight = target_allocation.get(asset_class, 0.0)
            drift[asset_class] = current_weight - target_weight
        
        return {
            "current_allocation": allocation,
            "target_allocation": target_allocation,
            "allocation_drift": drift,
            "max_drift": max(abs(d) for d in drift.values()),
            "needs_rebalancing": max(abs(d) for d in drift.values()) > self.rebalancing_triggers["drift_threshold"]
        }
    
    def _calculate_performance_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        positions = portfolio_data.get("positions", {})
        total_value = portfolio_data.get("total_value", 0)
        
        total_cost = 0
        total_current_value = 0
        unrealized_pnl = 0
        
        for symbol, position in positions.items():
            shares = position.get("shares", 0)
            avg_cost = position.get("avg_cost", 0)
            current_value = position.get("value", 0)
            
            position_cost = shares * avg_cost
            total_cost += position_cost
            total_current_value += current_value
            unrealized_pnl += current_value - position_cost
        
        # Calculate returns
        total_return = (unrealized_pnl / total_cost) if total_cost > 0 else 0.0
        
        # Calculate portfolio beta (simplified)
        weighted_beta = 0.0
        for symbol, position in positions.items():
            weight = position.get("value", 0) / total_value if total_value > 0 else 0
            beta = position.get("beta", 1.0)
            weighted_beta += weight * beta
        
        # Calculate portfolio volatility (simplified)
        portfolio_volatility = portfolio_data.get("volatility", 0.15)
        
        # Risk-adjusted returns (Sharpe ratio approximation)
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        sharpe_ratio = (total_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            "total_return": total_return,
            "unrealized_pnl": unrealized_pnl,
            "portfolio_beta": weighted_beta,
            "portfolio_volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            "total_cost_basis": total_cost,
            "current_value": total_current_value
        }
    
    def _assess_rebalancing_needs(self, portfolio_data: Dict[str, Any], risk_profile: str) -> Dict[str, Any]:
        """Assess portfolio rebalancing needs"""
        allocation_analysis = self._analyze_current_allocation(portfolio_data)
        
        needs_rebalancing = allocation_analysis.get("needs_rebalancing", False)
        max_drift = allocation_analysis.get("max_drift", 0.0)
        
        # Determine rebalancing urgency
        if max_drift > 0.15:
            urgency = "high"
        elif max_drift > 0.10:
            urgency = "medium"
        elif max_drift > 0.05:
            urgency = "low"
        else:
            urgency = "none"
        
        # Calculate rebalancing trades needed
        rebalancing_trades = []
        if needs_rebalancing:
            drift = allocation_analysis.get("allocation_drift", {})
            total_value = portfolio_data.get("total_value", 0)
            
            for asset_class, drift_amount in drift.items():
                if abs(drift_amount) > self.rebalancing_triggers["drift_threshold"]:
                    trade_value = drift_amount * total_value
                    action = "sell" if drift_amount > 0 else "buy"
                    
                    rebalancing_trades.append({
                        "asset_class": asset_class,
                        "action": action,
                        "amount": abs(trade_value),
                        "drift_percentage": drift_amount
                    })
        
        return {
            "needs_rebalancing": needs_rebalancing,
            "urgency": urgency,
            "max_drift": max_drift,
            "rebalancing_trades": rebalancing_trades,
            "estimated_cost": len(rebalancing_trades) * 10  # Simplified cost estimate
        }
    
    def _analyze_diversification(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio diversification"""
        positions = portfolio_data.get("positions", {})
        total_value = portfolio_data.get("total_value", 0)
        
        if not positions or total_value == 0:
            return {"diversification_score": 0.0}
        
        # Calculate position weights
        weights = []
        for position in positions.values():
            weight = position.get("value", 0) / total_value
            weights.append(weight)
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        hhi = sum(w ** 2 for w in weights)
        
        # Diversification score (inverse of HHI)
        diversification_score = 1 - hhi
        
        # Number of holdings
        num_holdings = len(positions)
        
        # Sector diversification (simplified)
        sectors = self._get_sector_distribution(positions)
        sector_hhi = sum((count / num_holdings) ** 2 for count in sectors.values())
        sector_diversification = 1 - sector_hhi
        
        return {
            "diversification_score": diversification_score,
            "hhi_index": hhi,
            "num_holdings": num_holdings,
            "sector_diversification": sector_diversification,
            "largest_position_weight": max(weights) if weights else 0.0,
            "top_5_concentration": sum(sorted(weights, reverse=True)[:5])
        }
    
    def _identify_optimization_opportunities(
        self, 
        portfolio_data: Dict[str, Any], 
        market_conditions: Dict[str, Any], 
        risk_profile: str
    ) -> List[Dict[str, Any]]:
        """Identify portfolio optimization opportunities"""
        opportunities = []
        
        # Check for underperforming positions
        positions = portfolio_data.get("positions", {})
        for symbol, position in positions.items():
            # Simplified underperformance check
            if position.get("volatility", 0) > 0.4:  # High volatility
                opportunities.append({
                    "type": "high_volatility",
                    "symbol": symbol,
                    "description": f"{symbol} has high volatility ({position.get('volatility', 0):.2f})",
                    "recommendation": "Consider reducing position size or adding hedging"
                })
        
        # Check for concentration risk
        diversification = self._analyze_diversification(portfolio_data)
        if diversification.get("largest_position_weight", 0) > 0.2:
            opportunities.append({
                "type": "concentration_risk",
                "description": "Large position concentration detected",
                "recommendation": "Consider reducing largest positions to improve diversification"
            })
        
        # Check for low diversification
        if diversification.get("diversification_score", 1.0) < 0.7:
            opportunities.append({
                "type": "low_diversification",
                "description": "Portfolio lacks sufficient diversification",
                "recommendation": "Add more holdings across different sectors and asset classes"
            })
        
        # Market condition based opportunities
        market_volatility = market_conditions.get("volatility", 0.15)
        if market_volatility > 0.25:
            opportunities.append({
                "type": "high_market_volatility",
                "description": "High market volatility detected",
                "recommendation": "Consider increasing cash allocation or adding defensive positions"
            })
        
        return opportunities
    
    def _analyze_sector_allocation(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sector allocation"""
        positions = portfolio_data.get("positions", {})
        total_value = portfolio_data.get("total_value", 0)
        
        # Simplified sector mapping
        sector_mapping = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
            "TSLA": "Automotive", "F": "Automotive",
            "JPM": "Financial", "BAC": "Financial",
            "JNJ": "Healthcare", "PFE": "Healthcare"
        }
        
        sector_allocation = {}
        
        for symbol, position in positions.items():
            sector = sector_mapping.get(symbol, "Other")
            position_value = position.get("value", 0)
            
            if sector not in sector_allocation:
                sector_allocation[sector] = 0.0
            
            sector_allocation[sector] += position_value / total_value if total_value > 0 else 0
        
        # Check for sector concentration
        max_sector_weight = max(sector_allocation.values()) if sector_allocation else 0.0
        sector_concentration_risk = max_sector_weight > 0.4
        
        return {
            "sector_allocation": sector_allocation,
            "max_sector_weight": max_sector_weight,
            "sector_concentration_risk": sector_concentration_risk,
            "num_sectors": len(sector_allocation)
        }
    
    def _analyze_portfolio_liquidity(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio liquidity"""
        positions = portfolio_data.get("positions", {})
        total_value = portfolio_data.get("total_value", 0)
        cash = portfolio_data.get("cash", 0)
        
        liquid_value = cash
        illiquid_value = 0
        
        for position in positions.values():
            position_value = position.get("value", 0)
            avg_volume = position.get("avg_volume", 0)
            
            # Classify liquidity based on volume
            if avg_volume > 1000000:  # High volume
                liquid_value += position_value
            elif avg_volume > 100000:  # Medium volume
                liquid_value += position_value * 0.8
                illiquid_value += position_value * 0.2
            else:  # Low volume
                liquid_value += position_value * 0.5
                illiquid_value += position_value * 0.5
        
        liquidity_ratio = liquid_value / total_value if total_value > 0 else 0
        
        return {
            "liquidity_ratio": liquidity_ratio,
            "liquid_value": liquid_value,
            "illiquid_value": illiquid_value,
            "cash_percentage": cash / total_value if total_value > 0 else 0,
            "liquidity_score": "high" if liquidity_ratio > 0.8 else "medium" if liquidity_ratio > 0.5 else "low"
        }
    
    def _assess_tax_efficiency(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess tax efficiency of portfolio"""
        positions = portfolio_data.get("positions", {})
        
        # Simplified tax efficiency analysis
        total_unrealized_gains = 0
        total_unrealized_losses = 0
        
        for position in positions.values():
            shares = position.get("shares", 0)
            avg_cost = position.get("avg_cost", 0)
            current_value = position.get("value", 0)
            
            cost_basis = shares * avg_cost
            unrealized_pnl = current_value - cost_basis
            
            if unrealized_pnl > 0:
                total_unrealized_gains += unrealized_pnl
            else:
                total_unrealized_losses += abs(unrealized_pnl)
        
        # Tax loss harvesting opportunities
        tax_loss_harvesting_potential = total_unrealized_losses
        
        return {
            "total_unrealized_gains": total_unrealized_gains,
            "total_unrealized_losses": total_unrealized_losses,
            "tax_loss_harvesting_potential": tax_loss_harvesting_potential,
            "tax_efficiency_score": 0.7  # Simplified score
        }
    
    def _get_sector_distribution(self, positions: Dict[str, Any]) -> Dict[str, int]:
        """Get sector distribution of positions"""
        sector_mapping = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
            "TSLA": "Automotive", "F": "Automotive",
            "JPM": "Financial", "BAC": "Financial",
            "JNJ": "Healthcare", "PFE": "Healthcare"
        }
        
        sectors = {}
        for symbol in positions.keys():
            sector = sector_mapping.get(symbol, "Other")
            sectors[sector] = sectors.get(sector, 0) + 1
        
        return sectors
    
    def _get_rebalancing_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get rebalancing recommendations"""
        recommendations = []
        
        rebalancing_needs = analysis.get("rebalancing_needs", {})
        
        if rebalancing_needs.get("needs_rebalancing", False):
            urgency = rebalancing_needs.get("urgency", "low")
            
            recommendation = {
                "type": "rebalancing",
                "priority": urgency,
                "action": "rebalance_portfolio",
                "description": f"Portfolio drift detected - {urgency} priority rebalancing needed",
                "specific_actions": [],
                "estimated_cost": rebalancing_needs.get("estimated_cost", 0)
            }
            
            # Add specific rebalancing trades
            for trade in rebalancing_needs.get("rebalancing_trades", []):
                recommendation["specific_actions"].append(
                    f"{trade['action'].title()} {trade['asset_class']} - ${trade['amount']:,.2f}"
                )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_optimization_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get optimization recommendations"""
        recommendations = []
        
        opportunities = analysis.get("optimization_opportunities", [])
        
        for opportunity in opportunities:
            priority = "high" if opportunity["type"] in ["concentration_risk", "high_market_volatility"] else "medium"
            
            recommendation = {
                "type": "optimization",
                "priority": priority,
                "action": opportunity["type"],
                "description": opportunity["description"],
                "specific_actions": [opportunity["recommendation"]]
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_risk_management_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get risk management recommendations"""
        recommendations = []
        
        # Check overall risk level
        risk_level = analysis.get("overall_risk_level", "medium")
        
        if risk_level == "high":
            recommendations.append({
                "type": "risk_management",
                "priority": "high",
                "action": "reduce_portfolio_risk",
                "description": "Portfolio risk level is high",
                "specific_actions": [
                    "Reduce position sizes in high-volatility stocks",
                    "Increase cash allocation",
                    "Consider adding defensive positions"
                ]
            })
        
        # Check diversification
        diversification = analysis.get("diversification_analysis", {})
        
        if diversification.get("diversification_score", 1.0) < 0.6:
            recommendations.append({
                "type": "risk_management",
                "priority": "medium",
                "action": "improve_diversification",
                "description": "Portfolio lacks sufficient diversification",
                "specific_actions": [
                    "Add holdings in underrepresented sectors",
                    "Consider index funds for broader exposure",
                    "Reduce concentration in largest positions"
                ]
            })
        
        return recommendations
    
    def _get_tax_efficiency_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get tax efficiency recommendations"""
        recommendations = []
        
        tax_analysis = analysis.get("tax_efficiency", {})
        tax_loss_potential = tax_analysis.get("tax_loss_harvesting_potential", 0)
        
        if tax_loss_potential > 1000:  # $1,000 threshold
            recommendations.append({
                "type": "tax_optimization",
                "priority": "low",
                "action": "tax_loss_harvesting",
                "description": f"Tax loss harvesting opportunity: ${tax_loss_potential:,.2f}",
                "specific_actions": [
                    "Consider realizing losses to offset gains",
                    "Review wash sale rules before executing",
                    "Consult tax advisor for optimal timing"
                ]
            })
        
        return recommendations
    
    def _calculate_portfolio_confidence(
        self, 
        analysis_json: Dict[str, Any], 
        portfolio_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for portfolio analysis"""
        base_confidence = 0.7
        
        # Adjust based on portfolio size
        total_value = portfolio_data.get("total_value", 0)
        if total_value > 100000:
            base_confidence += 0.1
        
        # Adjust based on number of positions
        num_positions = len(portfolio_data.get("positions", {}))
        if num_positions > 10:
            base_confidence += 0.1
        elif num_positions < 3:
            base_confidence -= 0.1
        
        # Adjust based on data completeness
        if all(key in portfolio_data for key in ["total_value", "cash", "positions"]):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0) 