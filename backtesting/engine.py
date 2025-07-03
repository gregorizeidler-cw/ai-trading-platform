"""
Professional Backtesting Engine
Realistic market simulation with slippage, commissions, and market impact
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from .slippage_model import SlippageModel
from .commission_calculator import CommissionCalculator
from .market_impact_model import MarketImpactModel
from .metrics import PerformanceMetrics

@dataclass
class Trade:
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    pnl: float = 0.0
    strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    first_entry: datetime
    last_update: datetime

class BacktestEngine:
    """
    Professional Backtesting Engine
    
    Features:
    - Realistic slippage and commission modeling
    - Market impact simulation
    - Multiple strategy support
    - Advanced performance metrics
    - Risk management integration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.slippage_model = SlippageModel(self.config.get("slippage", {}))
        self.commission_calc = CommissionCalculator(self.config.get("commission", {}))
        self.market_impact = MarketImpactModel(self.config.get("market_impact", {}))
        self.performance_metrics = PerformanceMetrics()
        
        # Portfolio state
        self.initial_capital = self.config["initial_capital"]
        self.current_capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict[str, Any]] = []
        
        # Strategy management
        self.strategies: Dict[str, Callable] = {}
        self.strategy_performance: Dict[str, Dict] = {}
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            "initial_capital": 100000,
            "slippage": {
                "model": "linear",
                "base_slippage": 0.0005,  # 5 bps
                "volatility_factor": 0.5,
                "volume_factor": 0.3
            },
            "commission": {
                "type": "per_share",
                "rate": 0.005,  # $0.005 per share
                "minimum": 1.0   # $1 minimum
            },
            "market_impact": {
                "model": "sqrt",
                "impact_factor": 0.1,
                "temporary_impact": 0.5,
                "permanent_impact": 0.5
            },
            "risk_management": {
                "max_position_size": 0.1,  # 10% max per position
                "max_portfolio_leverage": 1.0,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.15
            }
        }
    
    def add_strategy(self, name: str, strategy_func: Callable):
        """Add a trading strategy to the backtest"""
        self.strategies[name] = strategy_func
        self.strategy_performance[name] = {
            "trades": [],
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0
        }
    
    def run_backtest(self, data: Dict[str, pd.DataFrame], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Run comprehensive backtest
        """
        print(f"🚀 Starting backtest from {start_date} to {end_date}")
        
        # Initialize backtest state
        self._initialize_backtest(start_date)
        
        # Get all dates in chronological order
        all_dates = self._get_trading_dates(data, start_date, end_date)
        
        # Main backtest loop
        for current_date in all_dates:
            # Get market data for current date
            current_data = self._get_market_data_for_date(data, current_date)
            
            if not current_data:
                continue
            
            # Update portfolio values
            self._update_portfolio_values(current_data, current_date)
            
            # Run all strategies
            for strategy_name, strategy_func in self.strategies.items():
                signals = strategy_func(current_data, self._get_portfolio_state())
                
                # Execute signals
                for signal in signals:
                    self._execute_signal(signal, current_data, current_date, strategy_name)
            
            # Risk management checks
            self._apply_risk_management(current_data, current_date)
            
            # Record equity curve
            self._record_equity_point(current_date)
        
        # Calculate final performance
        results = self._calculate_final_results()
        
        print(f"✅ Backtest completed. Final portfolio value: ${self.current_capital:,.2f}")
        
        return results
    
    def _initialize_backtest(self, start_date: datetime):
        """Initialize backtest state"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Initialize strategy performance tracking
        for strategy_name in self.strategies:
            self.strategy_performance[strategy_name] = {
                "trades": [],
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0
            }
    
    def _get_trading_dates(self, data: Dict[str, pd.DataFrame], start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get all trading dates in chronological order"""
        all_dates = set()
        
        for symbol, df in data.items():
            symbol_dates = df.index[(df.index >= start_date) & (df.index <= end_date)]
            all_dates.update(symbol_dates)
        
        return sorted(list(all_dates))
    
    def _get_market_data_for_date(self, data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, Dict]:
        """Get market data for specific date"""
        market_data = {}
        
        for symbol, df in data.items():
            if date in df.index:
                row = df.loc[date]
                market_data[symbol] = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'timestamp': date
                }
        
        return market_data
    
    def _execute_signal(self, signal: Dict[str, Any], market_data: Dict, timestamp: datetime, strategy_name: str):
        """Execute trading signal with realistic costs"""
        
        symbol = signal['symbol']
        side = signal['side']  # BUY/SELL
        quantity = signal['quantity']
        
        if symbol not in market_data:
            return
        
        # Get current market price
        current_price = market_data[symbol]['close']
        volume = market_data[symbol]['volume']
        
        # Calculate realistic execution price
        slippage = self.slippage_model.calculate_slippage(symbol, quantity, volume, current_price)
        market_impact = self.market_impact.calculate_impact(symbol, quantity, volume)
        commission = self.commission_calc.calculate_commission(quantity, current_price)
        
        # Adjust execution price
        if side == "BUY":
            execution_price = current_price + slippage + market_impact
        else:  # SELL
            execution_price = current_price - slippage - market_impact
        
        # Check if we have enough capital
        trade_value = quantity * execution_price
        if side == "BUY" and trade_value > self.current_capital:
            return  # Insufficient capital
        
        # Execute the trade
        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=execution_price,
            entry_time=timestamp,
            commission=commission,
            slippage=slippage,
            market_impact=market_impact,
            strategy=strategy_name
        )
        
        # Update positions
        self._update_positions(trade)
        
        # Update capital
        if side == "BUY":
            self.current_capital -= (trade_value + commission)
        else:  # SELL
            self.current_capital += (trade_value - commission)
        
        # Record trade
        self.trades.append(trade)
        self.strategy_performance[strategy_name]["trades"].append(trade)
    
    def _update_positions(self, trade: Trade):
        """Update position tracking"""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            # New position
            if trade.side == "BUY":
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=trade.quantity,
                    avg_price=trade.entry_price,
                    market_value=trade.quantity * trade.entry_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    first_entry=trade.entry_time,
                    last_update=trade.entry_time
                )
        else:
            # Update existing position
            position = self.positions[symbol]
            
            if trade.side == "BUY":
                # Add to position
                total_cost = (position.quantity * position.avg_price) + (trade.quantity * trade.entry_price)
                total_quantity = position.quantity + trade.quantity
                position.avg_price = total_cost / total_quantity
                position.quantity = total_quantity
            else:  # SELL
                # Reduce position
                if trade.quantity >= position.quantity:
                    # Close entire position
                    realized_pnl = (trade.entry_price - position.avg_price) * position.quantity
                    position.realized_pnl += realized_pnl
                    del self.positions[symbol]
                else:
                    # Partial close
                    realized_pnl = (trade.entry_price - position.avg_price) * trade.quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= trade.quantity
            
            if symbol in self.positions:
                self.positions[symbol].last_update = trade.entry_time
    
    def _update_portfolio_values(self, market_data: Dict, timestamp: datetime):
        """Update portfolio values based on current market prices"""
        total_market_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['close']
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                total_market_value += position.market_value
        
        # Update total portfolio value
        self.current_capital = self.current_capital + total_market_value
    
    def _apply_risk_management(self, market_data: Dict, timestamp: datetime):
        """Apply risk management rules"""
        risk_config = self.config["risk_management"]
        
        for symbol, position in list(self.positions.items()):
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol]['close']
            pnl_pct = (current_price - position.avg_price) / position.avg_price
            
            # Stop loss check
            if pnl_pct <= -risk_config["stop_loss_pct"]:
                self._close_position(symbol, current_price, timestamp, "STOP_LOSS")
            
            # Take profit check
            elif pnl_pct >= risk_config["take_profit_pct"]:
                self._close_position(symbol, current_price, timestamp, "TAKE_PROFIT")
    
    def _close_position(self, symbol: str, price: float, timestamp: datetime, reason: str):
        """Close a position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Create closing trade
        trade = Trade(
            symbol=symbol,
            side="SELL" if position.quantity > 0 else "BUY",
            quantity=abs(position.quantity),
            entry_price=price,
            entry_time=timestamp,
            commission=self.commission_calc.calculate_commission(abs(position.quantity), price),
            strategy=f"RISK_MANAGEMENT_{reason}"
        )
        
        # Calculate PnL
        trade.pnl = (price - position.avg_price) * position.quantity
        
        # Update capital
        trade_value = abs(position.quantity) * price
        self.current_capital += trade_value - trade.commission
        
        # Remove position
        del self.positions[symbol]
        
        # Record trade
        self.trades.append(trade)
    
    def _record_equity_point(self, timestamp: datetime):
        """Record equity curve point"""
        total_value = self.current_capital
        
        # Add unrealized PnL
        for position in self.positions.values():
            total_value += position.unrealized_pnl
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'total_value': total_value,
            'cash': self.current_capital,
            'positions_value': sum(p.market_value for p in self.positions.values()),
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values()),
            'realized_pnl': sum(t.pnl for t in self.trades if t.pnl != 0)
        })
    
    def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state for strategies"""
        return {
            'capital': self.current_capital,
            'positions': dict(self.positions),
            'total_value': self.current_capital + sum(p.market_value for p in self.positions.values())
        }
    
    def _calculate_final_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest results"""
        
        # Basic metrics
        final_value = self.equity_curve[-1]['total_value'] if self.equity_curve else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Performance metrics
        performance = self.performance_metrics.calculate_metrics(
            self.equity_curve,
            self.trades,
            self.initial_capital
        )
        
        # Strategy-specific performance
        strategy_results = {}
        for strategy_name, strategy_data in self.strategy_performance.items():
            strategy_trades = strategy_data["trades"]
            if strategy_trades:
                strategy_pnl = sum(t.pnl for t in strategy_trades if t.pnl != 0)
                strategy_wins = sum(1 for t in strategy_trades if t.pnl > 0)
                strategy_results[strategy_name] = {
                    "total_trades": len(strategy_trades),
                    "total_pnl": strategy_pnl,
                    "win_rate": strategy_wins / len(strategy_trades) if strategy_trades else 0,
                    "avg_trade": strategy_pnl / len(strategy_trades) if strategy_trades else 0
                }
        
        return {
            "summary": {
                "initial_capital": self.initial_capital,
                "final_value": final_value,
                "total_return": total_return,
                "total_trades": len(self.trades),
                "winning_trades": sum(1 for t in self.trades if t.pnl > 0),
                "losing_trades": sum(1 for t in self.trades if t.pnl < 0)
            },
            "performance_metrics": performance,
            "strategy_performance": strategy_results,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
            "final_positions": dict(self.positions)
        }
    
    def get_trade_analysis(self) -> pd.DataFrame:
        """Get detailed trade analysis"""
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'entry_price': trade.entry_price,
                'entry_time': trade.entry_time,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'market_impact': trade.market_impact,
                'pnl': trade.pnl,
                'strategy': trade.strategy
            })
        
        return pd.DataFrame(trade_data)
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame(self.equity_curve).set_index('timestamp') 