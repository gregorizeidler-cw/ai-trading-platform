"""
Base broker interface for trading execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class OrderType(str, Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order data structure"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    side: str  # "long" or "short"


@dataclass
class Account:
    """Account data structure"""
    account_id: str
    buying_power: float
    cash: float
    portfolio_value: float
    positions: List[Position]
    day_trade_count: int = 0
    pattern_day_trader: bool = False


class BaseBroker(ABC):
    """Base class for all broker implementations"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.connected = False
        self.orders: Dict[str, Order] = {}
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker API"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from broker API"""
        pass
    
    @abstractmethod
    async def get_account(self) -> Account:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """Submit an order and return order ID"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        """Get order history"""
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote"""
        pass
    
    async def create_market_order(
        self, 
        symbol: str, 
        side: OrderSide, 
        quantity: float
    ) -> Order:
        """Create a market order"""
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET
        )
    
    async def create_limit_order(
        self, 
        symbol: str, 
        side: OrderSide, 
        quantity: float, 
        price: float
    ) -> Order:
        """Create a limit order"""
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=price
        )
    
    async def create_stop_loss_order(
        self, 
        symbol: str, 
        side: OrderSide, 
        quantity: float, 
        stop_price: float
    ) -> Order:
        """Create a stop loss order"""
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_price
        )
    
    def validate_order(self, order: Order) -> bool:
        """Validate order parameters"""
        if order.quantity <= 0:
            logger.error(f"Invalid quantity: {order.quantity}")
            return False
        
        if order.order_type == OrderType.LIMIT and order.price is None:
            logger.error("Limit order requires price")
            return False
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            logger.error("Stop order requires stop price")
            return False
        
        return True
    
    def calculate_position_size(
        self, 
        symbol: str, 
        price: float, 
        risk_amount: float, 
        stop_loss_price: Optional[float] = None
    ) -> float:
        """Calculate position size based on risk management"""
        if stop_loss_price is None:
            # Use 2% default risk per trade
            return risk_amount / (price * 0.02)
        
        risk_per_share = abs(price - stop_loss_price)
        if risk_per_share == 0:
            return 0
        
        return risk_amount / risk_per_share
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status"""
        return {
            "broker": self.name,
            "connected": self.connected,
            "timestamp": datetime.utcnow().isoformat()
        } 