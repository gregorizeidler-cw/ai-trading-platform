"""
Order Manager - Coordinates order execution across multiple brokers.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger
from enum import Enum

from ..brokers.base_broker import BaseBroker, Order, OrderStatus, OrderSide, OrderType


class ExecutionStrategy(str, Enum):
    """Order execution strategies"""
    IMMEDIATE = "immediate"
    TWAP = "twap"  # Time Weighted Average Price
    VWAP = "vwap"  # Volume Weighted Average Price
    ICEBERG = "iceberg"  # Break large orders into smaller chunks
    SMART = "smart"  # AI-driven execution


class OrderManager:
    """Manages order execution across multiple brokers"""
    
    def __init__(self):
        self.brokers: Dict[str, BaseBroker] = {}
        self.active_orders: Dict[str, Order] = {}
        self.execution_strategies = {}
        self.risk_limits = {
            "max_position_size": 10000,  # Maximum position size per symbol
            "max_order_value": 50000,    # Maximum single order value
            "max_daily_volume": 100000,  # Maximum daily trading volume
            "max_concentration": 0.2     # Maximum concentration per symbol (20%)
        }
        self.daily_volume = 0
        self.positions: Dict[str, float] = {}
        
    def add_broker(self, broker: BaseBroker):
        """Add a broker to the order manager"""
        self.brokers[broker.name] = broker
        logger.info(f"Added broker: {broker.name}")
    
    def remove_broker(self, broker_name: str):
        """Remove a broker from the order manager"""
        if broker_name in self.brokers:
            del self.brokers[broker_name]
            logger.info(f"Removed broker: {broker_name}")
    
    async def connect_all_brokers(self) -> Dict[str, bool]:
        """Connect to all brokers"""
        results = {}
        
        for name, broker in self.brokers.items():
            try:
                success = await broker.connect()
                results[name] = success
                logger.info(f"Broker {name} connection: {'Success' if success else 'Failed'}")
            except Exception as e:
                results[name] = False
                logger.error(f"Error connecting to broker {name}: {e}")
        
        return results
    
    async def disconnect_all_brokers(self) -> Dict[str, bool]:
        """Disconnect from all brokers"""
        results = {}
        
        for name, broker in self.brokers.items():
            try:
                success = await broker.disconnect()
                results[name] = success
            except Exception as e:
                results[name] = False
                logger.error(f"Error disconnecting from broker {name}: {e}")
        
        return results
    
    async def submit_order(
        self, 
        order: Order, 
        broker_name: Optional[str] = None,
        strategy: ExecutionStrategy = ExecutionStrategy.IMMEDIATE
    ) -> str:
        """Submit an order with risk management and execution strategy"""
        try:
            # Risk management checks
            if not self._validate_risk_limits(order):
                raise ValueError("Order violates risk limits")
            
            # Select broker
            if broker_name and broker_name not in self.brokers:
                raise ValueError(f"Broker {broker_name} not found")
            
            selected_broker = self.brokers[broker_name] if broker_name else self._select_best_broker(order)
            
            if not selected_broker.connected:
                raise ValueError(f"Broker {selected_broker.name} not connected")
            
            # Execute based on strategy
            if strategy == ExecutionStrategy.IMMEDIATE:
                return await self._execute_immediate(order, selected_broker)
            elif strategy == ExecutionStrategy.TWAP:
                return await self._execute_twap(order, selected_broker)
            elif strategy == ExecutionStrategy.ICEBERG:
                return await self._execute_iceberg(order, selected_broker)
            elif strategy == ExecutionStrategy.SMART:
                return await self._execute_smart(order, selected_broker)
            else:
                return await self._execute_immediate(order, selected_broker)
                
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if order_id not in self.active_orders:
                logger.error(f"Order {order_id} not found")
                return False
            
            order = self.active_orders[order_id]
            
            # Find the broker that has this order
            for broker in self.brokers.values():
                if order_id in broker.orders:
                    success = await broker.cancel_order(order_id)
                    if success:
                        order.status = OrderStatus.CANCELLED
                        logger.info(f"Order {order_id} cancelled successfully")
                    return success
            
            logger.error(f"No broker found for order {order_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status across all brokers"""
        try:
            if order_id in self.active_orders:
                return self.active_orders[order_id].status
            
            # Check all brokers
            for broker in self.brokers.values():
                if order_id in broker.orders:
                    return await broker.get_order_status(order_id)
            
            return OrderStatus.PENDING
            
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return OrderStatus.PENDING
    
    async def get_all_positions(self) -> Dict[str, List[Any]]:
        """Get positions from all brokers"""
        all_positions = {}
        
        for name, broker in self.brokers.items():
            try:
                if broker.connected:
                    positions = await broker.get_positions()
                    all_positions[name] = positions
                else:
                    all_positions[name] = []
            except Exception as e:
                logger.error(f"Error getting positions from {name}: {e}")
                all_positions[name] = []
        
        return all_positions
    
    async def get_consolidated_portfolio(self) -> Dict[str, Any]:
        """Get consolidated portfolio across all brokers"""
        try:
            all_positions = await self.get_all_positions()
            
            # Consolidate positions by symbol
            consolidated = {}
            total_value = 0
            
            for broker_name, positions in all_positions.items():
                for position in positions:
                    symbol = position.symbol
                    
                    if symbol not in consolidated:
                        consolidated[symbol] = {
                            "symbol": symbol,
                            "total_quantity": 0,
                            "total_value": 0,
                            "avg_price": 0,
                            "unrealized_pnl": 0,
                            "brokers": []
                        }
                    
                    consolidated[symbol]["total_quantity"] += position.quantity
                    consolidated[symbol]["total_value"] += position.market_value
                    consolidated[symbol]["unrealized_pnl"] += position.unrealized_pnl
                    consolidated[symbol]["brokers"].append({
                        "broker": broker_name,
                        "quantity": position.quantity,
                        "value": position.market_value
                    })
                    
                    total_value += position.market_value
            
            # Calculate average prices
            for symbol_data in consolidated.values():
                if symbol_data["total_quantity"] > 0:
                    symbol_data["avg_price"] = symbol_data["total_value"] / symbol_data["total_quantity"]
            
            return {
                "positions": consolidated,
                "total_portfolio_value": total_value,
                "num_positions": len(consolidated),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting consolidated portfolio: {e}")
            return {}
    
    def _validate_risk_limits(self, order: Order) -> bool:
        """Validate order against risk limits"""
        try:
            # Check maximum order value
            if order.order_type == OrderType.MARKET:
                # For market orders, we need to estimate value
                estimated_value = order.quantity * 100  # Rough estimate
            else:
                estimated_value = order.quantity * (order.price or 100)
            
            if estimated_value > self.risk_limits["max_order_value"]:
                logger.error(f"Order value {estimated_value} exceeds limit {self.risk_limits['max_order_value']}")
                return False
            
            # Check daily volume limit
            if self.daily_volume + estimated_value > self.risk_limits["max_daily_volume"]:
                logger.error(f"Daily volume limit exceeded")
                return False
            
            # Check position size limit
            current_position = self.positions.get(order.symbol, 0)
            new_position = current_position + (order.quantity if order.side == OrderSide.BUY else -order.quantity)
            
            if abs(new_position) > self.risk_limits["max_position_size"]:
                logger.error(f"Position size limit exceeded for {order.symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating risk limits: {e}")
            return False
    
    def _select_best_broker(self, order: Order) -> BaseBroker:
        """Select the best broker for an order"""
        connected_brokers = [broker for broker in self.brokers.values() if broker.connected]
        
        if not connected_brokers:
            raise ValueError("No connected brokers available")
        
        # Simple selection - just use the first connected broker
        # In a real implementation, you might consider:
        # - Broker fees
        # - Execution quality
        # - Available liquidity
        # - Market data quality
        
        return connected_brokers[0]
    
    async def _execute_immediate(self, order: Order, broker: BaseBroker) -> str:
        """Execute order immediately"""
        order_id = await broker.submit_order(order)
        self.active_orders[order_id] = order
        
        # Update tracking
        self._update_position_tracking(order)
        
        return order_id
    
    async def _execute_twap(self, order: Order, broker: BaseBroker, duration_minutes: int = 30) -> str:
        """Execute order using Time Weighted Average Price strategy"""
        # Break order into smaller chunks over time
        num_chunks = min(10, duration_minutes // 3)  # Max 10 chunks, min 3 minutes apart
        chunk_size = order.quantity / num_chunks
        interval = duration_minutes * 60 / num_chunks  # seconds
        
        logger.info(f"Executing TWAP order: {num_chunks} chunks of {chunk_size} shares over {duration_minutes} minutes")
        
        parent_order_id = f"twap_{order.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Submit first chunk immediately
        chunk_order = Order(
            symbol=order.symbol,
            side=order.side,
            quantity=chunk_size,
            order_type=OrderType.MARKET  # Use market orders for TWAP
        )
        
        first_chunk_id = await broker.submit_order(chunk_order)
        self.active_orders[first_chunk_id] = chunk_order
        
        # Schedule remaining chunks
        asyncio.create_task(self._schedule_twap_chunks(
            order, broker, chunk_size, interval, num_chunks - 1
        ))
        
        return parent_order_id
    
    async def _execute_iceberg(self, order: Order, broker: BaseBroker, visible_size: float = 0.1) -> str:
        """Execute large order using iceberg strategy"""
        # Show only a small portion of the order at a time
        visible_quantity = order.quantity * visible_size
        
        logger.info(f"Executing iceberg order: {order.quantity} shares, {visible_quantity} visible")
        
        # Submit first visible portion
        iceberg_order = Order(
            symbol=order.symbol,
            side=order.side,
            quantity=visible_quantity,
            order_type=order.order_type,
            price=order.price
        )
        
        order_id = await broker.submit_order(iceberg_order)
        self.active_orders[order_id] = iceberg_order
        
        # TODO: Implement logic to monitor fills and submit next portions
        
        return order_id
    
    async def _execute_smart(self, order: Order, broker: BaseBroker) -> str:
        """Execute order using AI-driven smart execution"""
        # Analyze market conditions and choose best execution strategy
        
        # Get current market data
        quote = await broker.get_quote(order.symbol)
        
        if not quote:
            # Fallback to immediate execution
            return await self._execute_immediate(order, broker)
        
        # Simple smart logic based on spread and volume
        spread = quote.get("ask", 0) - quote.get("bid", 0)
        volume = quote.get("volume", 0)
        
        # If spread is tight and volume is high, use immediate execution
        if spread < 0.05 and volume > 1000000:
            return await self._execute_immediate(order, broker)
        
        # If large order relative to volume, use TWAP
        if order.quantity > volume * 0.01:  # More than 1% of daily volume
            return await self._execute_twap(order, broker)
        
        # Default to immediate execution
        return await self._execute_immediate(order, broker)
    
    async def _schedule_twap_chunks(
        self, 
        original_order: Order, 
        broker: BaseBroker, 
        chunk_size: float, 
        interval: float, 
        remaining_chunks: int
    ):
        """Schedule remaining TWAP chunks"""
        for i in range(remaining_chunks):
            await asyncio.sleep(interval)
            
            chunk_order = Order(
                symbol=original_order.symbol,
                side=original_order.side,
                quantity=chunk_size,
                order_type=OrderType.MARKET
            )
            
            try:
                chunk_id = await broker.submit_order(chunk_order)
                self.active_orders[chunk_id] = chunk_order
                logger.info(f"Submitted TWAP chunk {i+2}/{remaining_chunks+1} for {original_order.symbol}")
            except Exception as e:
                logger.error(f"Error submitting TWAP chunk: {e}")
    
    def _update_position_tracking(self, order: Order):
        """Update position tracking for risk management"""
        if order.symbol not in self.positions:
            self.positions[order.symbol] = 0
        
        quantity_change = order.quantity if order.side == OrderSide.BUY else -order.quantity
        self.positions[order.symbol] += quantity_change
        
        # Update daily volume
        estimated_value = order.quantity * (order.price or 100)
        self.daily_volume += estimated_value
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "total_orders": len(self.active_orders),
            "daily_volume": self.daily_volume,
            "active_positions": len(self.positions),
            "connected_brokers": sum(1 for broker in self.brokers.values() if broker.connected),
            "total_brokers": len(self.brokers),
            "risk_limits": self.risk_limits,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def reset_daily_limits(self):
        """Reset daily trading limits (call at market open)"""
        self.daily_volume = 0
        logger.info("Daily trading limits reset") 