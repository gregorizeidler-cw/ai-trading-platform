"""
Multi-Asset Trading Agent
Handles trading across multiple asset classes: stocks, crypto, forex, commodities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass
from loguru import logger
import requests

from .base_agent import BaseAgent
from ..llm.openai_client import OpenAIClient


@dataclass
class AssetAllocation:
    """Asset allocation recommendation"""
    asset_class: str
    symbol: str
    weight: float
    expected_return: float
    risk: float
    correlation: float
    rationale: str


class MultiAssetTrader(BaseAgent):
    """Multi-asset trading agent for diversified portfolio management"""
    
    def __init__(self, openai_client: OpenAIClient):
        super().__init__(
            name="Multi-Asset Trader",
            description="Manages trading across multiple asset classes with optimal allocation",
            openai_client=openai_client
        )
        
        self.asset_classes = {
            'stocks': ['SPY', 'QQQ', 'IWM', 'VTI', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA'],
            'crypto': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD'],
            'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X'],
            'commodities': ['GC=F', 'CL=F', 'SI=F', 'NG=F', 'HG=F'],  # Gold, Oil, Silver, Gas, Copper
            'bonds': ['TLT', 'IEF', 'SHY', 'TIP', 'HYG'],
            'reits': ['VNQ', 'SCHH', 'RWR', 'IYR', 'XLRE']
        }
        
        self.correlation_matrix = {}
        self.risk_metrics = {}
        
    async def analyze_multi_asset_opportunities(self, target_risk: float = 0.15) -> List[AssetAllocation]:
        """Analyze opportunities across all asset classes"""
        try:
            allocations = []
            
            # Get data for all asset classes
            all_data = await self._get_multi_asset_data()
            
            # Calculate returns and risks
            asset_metrics = self._calculate_asset_metrics(all_data)
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(all_data)
            
            # Optimize portfolio allocation
            optimal_weights = self._optimize_portfolio(asset_metrics, correlation_matrix, target_risk)
            
            # Generate allocations
            for asset_class, symbols in self.asset_classes.items():
                for symbol in symbols:
                    if symbol in optimal_weights and optimal_weights[symbol] > 0.01:  # Min 1% allocation
                        
                        metrics = asset_metrics.get(symbol, {})
                        
                        allocation = AssetAllocation(
                            asset_class=asset_class,
                            symbol=symbol,
                            weight=optimal_weights[symbol],
                            expected_return=metrics.get('expected_return', 0),
                            risk=metrics.get('volatility', 0),
                            correlation=self._get_avg_correlation(symbol, correlation_matrix),
                            rationale=await self._generate_allocation_rationale(symbol, asset_class, metrics)
                        )
                        
                        allocations.append(allocation)
            
            return sorted(allocations, key=lambda x: x.weight, reverse=True)
            
        except Exception as e:
            logger.error(f"Error in multi-asset analysis: {e}")
            return []
    
    async def _get_multi_asset_data(self) -> Dict[str, pd.DataFrame]:
        """Get data for all asset classes"""
        all_data = {}
        
        for asset_class, symbols in self.asset_classes.items():
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1y")
                    
                    if not data.empty:
                        all_data[symbol] = data
                        
                except Exception as e:
                    logger.warning(f"Could not get data for {symbol}: {e}")
                    continue
        
        return all_data
    
    def _calculate_asset_metrics(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Calculate risk-return metrics for each asset"""
        metrics = {}
        
        for symbol, df in data.items():
            try:
                returns = df['Close'].pct_change().dropna()
                
                if len(returns) < 30:
                    continue
                
                # Calculate metrics
                expected_return = returns.mean() * 252  # Annualized
                volatility = returns.std() * np.sqrt(252)  # Annualized
                sharpe_ratio = expected_return / volatility if volatility > 0 else 0
                
                # Downside metrics
                downside_returns = returns[returns < 0]
                downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
                sortino_ratio = expected_return / downside_deviation if downside_deviation > 0 else 0
                
                # Maximum drawdown
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                
                metrics[symbol] = {
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'max_drawdown': max_drawdown,
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis()
                }
                
            except Exception as e:
                logger.warning(f"Could not calculate metrics for {symbol}: {e}")
                continue
        
        return metrics
    
    def _calculate_correlation_matrix(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix across all assets"""
        try:
            returns_data = {}
            
            for symbol, df in data.items():
                returns = df['Close'].pct_change().dropna()
                if len(returns) > 30:
                    returns_data[symbol] = returns
            
            if not returns_data:
                return pd.DataFrame()
            
            # Align all return series
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def _optimize_portfolio(
        self, 
        asset_metrics: Dict[str, Dict[str, float]], 
        correlation_matrix: pd.DataFrame, 
        target_risk: float
    ) -> Dict[str, float]:
        """Optimize portfolio allocation using mean-variance optimization"""
        try:
            symbols = list(asset_metrics.keys())
            
            if len(symbols) < 2:
                return {}
            
            # Filter symbols that exist in correlation matrix
            valid_symbols = [s for s in symbols if s in correlation_matrix.index]
            
            if len(valid_symbols) < 2:
                return {}
            
            # Expected returns vector
            expected_returns = np.array([asset_metrics[s]['expected_return'] for s in valid_symbols])
            
            # Covariance matrix
            volatilities = np.array([asset_metrics[s]['volatility'] for s in valid_symbols])
            corr_matrix = correlation_matrix.loc[valid_symbols, valid_symbols].values
            
            # Convert correlation to covariance
            cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
            
            # Simple equal-weight as baseline (can be improved with optimization libraries)
            n_assets = len(valid_symbols)
            weights = np.ones(n_assets) / n_assets
            
            # Apply some basic optimization heuristics
            # Favor higher Sharpe ratio assets
            sharpe_ratios = np.array([asset_metrics[s]['sharpe_ratio'] for s in valid_symbols])
            sharpe_weights = np.maximum(sharpe_ratios, 0)
            
            if sharpe_weights.sum() > 0:
                sharpe_weights = sharpe_weights / sharpe_weights.sum()
                weights = 0.5 * weights + 0.5 * sharpe_weights
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Create result dictionary
            result = {}
            for i, symbol in enumerate(valid_symbols):
                result[symbol] = float(weights[i])
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return {}
    
    def _get_avg_correlation(self, symbol: str, correlation_matrix: pd.DataFrame) -> float:
        """Get average correlation of symbol with other assets"""
        try:
            if symbol not in correlation_matrix.index:
                return 0.0
            
            correlations = correlation_matrix.loc[symbol].drop(symbol)
            return float(correlations.mean())
            
        except Exception as e:
            return 0.0
    
    async def _generate_allocation_rationale(
        self, 
        symbol: str, 
        asset_class: str, 
        metrics: Dict[str, float]
    ) -> str:
        """Generate AI-powered rationale for allocation"""
        try:
            prompt = f"""
            Provide a brief rationale for allocating to {symbol} in the {asset_class} asset class.
            
            Key metrics:
            - Expected Return: {metrics.get('expected_return', 0):.2%}
            - Volatility: {metrics.get('volatility', 0):.2%}
            - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
            - Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
            
            Focus on:
            1. Risk-return profile
            2. Diversification benefits
            3. Market outlook
            4. Role in portfolio
            
            Keep it concise (2-3 sentences).
            """
            
            response = await self.openai_client.generate_completion(
                prompt=prompt,
                max_tokens=150,
                temperature=0.3
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating allocation rationale: {e}")
            return f"Allocation to {symbol} provides diversification in {asset_class} asset class."
    
    async def analyze_cross_asset_correlations(self) -> Dict[str, Any]:
        """Analyze correlations across asset classes"""
        try:
            data = await self._get_multi_asset_data()
            correlation_matrix = self._calculate_correlation_matrix(data)
            
            if correlation_matrix.empty:
                return {}
            
            # Calculate average correlations by asset class
            asset_class_correlations = {}
            
            for class1, symbols1 in self.asset_classes.items():
                for class2, symbols2 in self.asset_classes.items():
                    if class1 != class2:
                        # Get correlations between asset classes
                        valid_symbols1 = [s for s in symbols1 if s in correlation_matrix.index]
                        valid_symbols2 = [s for s in symbols2 if s in correlation_matrix.index]
                        
                        if valid_symbols1 and valid_symbols2:
                            corr_subset = correlation_matrix.loc[valid_symbols1, valid_symbols2]
                            avg_corr = corr_subset.values.mean()
                            
                            key = f"{class1}_vs_{class2}"
                            asset_class_correlations[key] = avg_corr
            
            # Find best diversification pairs
            diversification_pairs = []
            for key, corr in asset_class_correlations.items():
                if corr < 0.3:  # Low correlation threshold
                    diversification_pairs.append({
                        'pair': key,
                        'correlation': corr,
                        'diversification_benefit': 'High' if corr < 0.1 else 'Medium'
                    })
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'asset_class_correlations': asset_class_correlations,
                'diversification_opportunities': sorted(diversification_pairs, key=lambda x: x['correlation']),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cross-asset correlations: {e}")
            return {}
    
    async def generate_tactical_allocation(self, market_regime: str = "normal") -> Dict[str, Any]:
        """Generate tactical asset allocation based on market regime"""
        try:
            base_allocations = await self.analyze_multi_asset_opportunities()
            
            # Adjust allocations based on market regime
            regime_adjustments = {
                'bull_market': {
                    'stocks': 1.2,
                    'crypto': 1.3,
                    'bonds': 0.7,
                    'commodities': 1.1,
                    'forex': 0.9,
                    'reits': 1.1
                },
                'bear_market': {
                    'stocks': 0.6,
                    'crypto': 0.4,
                    'bonds': 1.4,
                    'commodities': 1.2,
                    'forex': 1.1,
                    'reits': 0.8
                },
                'high_inflation': {
                    'stocks': 0.8,
                    'crypto': 1.2,
                    'bonds': 0.5,
                    'commodities': 1.5,
                    'forex': 1.1,
                    'reits': 1.3
                },
                'recession': {
                    'stocks': 0.5,
                    'crypto': 0.3,
                    'bonds': 1.6,
                    'commodities': 0.8,
                    'forex': 1.2,
                    'reits': 0.6
                }
            }
            
            adjustments = regime_adjustments.get(market_regime, {})
            
            # Apply adjustments
            adjusted_allocations = []
            for allocation in base_allocations:
                adjustment_factor = adjustments.get(allocation.asset_class, 1.0)
                adjusted_weight = allocation.weight * adjustment_factor
                
                adjusted_allocation = AssetAllocation(
                    asset_class=allocation.asset_class,
                    symbol=allocation.symbol,
                    weight=adjusted_weight,
                    expected_return=allocation.expected_return,
                    risk=allocation.risk,
                    correlation=allocation.correlation,
                    rationale=f"[{market_regime.upper()}] {allocation.rationale}"
                )
                
                adjusted_allocations.append(adjusted_allocation)
            
            # Normalize weights
            total_weight = sum(a.weight for a in adjusted_allocations)
            if total_weight > 0:
                for allocation in adjusted_allocations:
                    allocation.weight = allocation.weight / total_weight
            
            return {
                'market_regime': market_regime,
                'allocations': adjusted_allocations,
                'total_expected_return': sum(a.weight * a.expected_return for a in adjusted_allocations),
                'total_risk': np.sqrt(sum((a.weight * a.risk) ** 2 for a in adjusted_allocations)),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating tactical allocation: {e}")
            return {} 