"""
Performance Analytics Agent - Advanced performance tracking and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import yfinance as yf
from dataclasses import dataclass
from loguru import logger

from .base_agent import BaseAgent
from ..llm.openai_client import OpenAIClient


@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float


class PerformanceAnalytics(BaseAgent):
    """Advanced performance analytics for trading strategies"""
    
    def __init__(self, openai_client: OpenAIClient):
        super().__init__(
            name="Performance Analytics",
            description="Advanced performance tracking and analysis",
            openai_client=openai_client
        )
        
    async def calculate_performance_metrics(self, returns: pd.Series) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + returns.mean()) ** 252 - 1
            volatility = returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Trade metrics
            winning_trades = returns[returns > 0]
            losing_trades = returns[returns < 0]
            
            win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
            gross_profit = winning_trades.sum()
            gross_loss = abs(losing_trades.sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)
    
    async def analyze_portfolio_performance(self, portfolio_returns: pd.Series) -> Dict[str, Any]:
        """Comprehensive portfolio performance analysis"""
        try:
            metrics = await self.calculate_performance_metrics(portfolio_returns)
            
            # Risk analysis
            var_95 = portfolio_returns.quantile(0.05)
            skewness = portfolio_returns.skew()
            kurtosis = portfolio_returns.kurtosis()
            
            # Generate AI analysis
            ai_analysis = await self._generate_performance_analysis(metrics)
            
            return {
                'metrics': {
                    'total_return': metrics.total_return,
                    'annualized_return': metrics.annualized_return,
                    'volatility': metrics.volatility,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'var_95': var_95,
                    'skewness': skewness,
                    'kurtosis': kurtosis
                },
                'ai_analysis': ai_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio performance: {e}")
            return {'error': str(e)}
    
    async def _generate_performance_analysis(self, metrics: PerformanceMetrics) -> str:
        """Generate AI-powered performance analysis"""
        try:
            prompt = f"""
            Analyze the following portfolio performance metrics:
            
            - Total Return: {metrics.total_return:.2%}
            - Annualized Return: {metrics.annualized_return:.2%}
            - Volatility: {metrics.volatility:.2%}
            - Sharpe Ratio: {metrics.sharpe_ratio:.2f}
            - Maximum Drawdown: {metrics.max_drawdown:.2%}
            - Win Rate: {metrics.win_rate:.2%}
            - Profit Factor: {metrics.profit_factor:.2f}
            
            Provide a comprehensive analysis covering:
            1. Overall performance assessment
            2. Risk-adjusted performance quality
            3. Risk characteristics
            4. Areas for improvement
            
            Focus on actionable insights.
            """
            
            response = await self.openai_client.generate_completion(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating performance analysis: {e}")
            return "Performance analysis completed. Review metrics for detailed insights." 