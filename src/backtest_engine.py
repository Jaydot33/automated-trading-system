"""
Backtest Engine
Handles nightly backtesting and signal generation
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from data"""
        pass


class BacktestEngine:
    """Engine for running nightly backtests"""
    
    def __init__(self, strategy_name: str, lookback_days: int = 252):
        self.strategy_name = strategy_name
        self.lookback_days = lookback_days
        self.strategy = None
    
    def set_strategy(self, strategy: BaseStrategy):
        """Set the trading strategy"""
        self.strategy = strategy
    
    def run_backtest(self) -> pd.DataFrame:
        """
        Run the backtest and generate order signals
        This would typically use Zipline, Backtrader, or custom engine
        """
        logger.info(f"Running backtest for strategy: {self.strategy_name}")
        
        # Mock backtest results - replace with actual backtest logic
        orders = pd.DataFrame({
            'symbol': ['NVDA', 'AMD', 'TSM', 'AVGO', 'QCOM'],
            'desired_allocation': [0.25, 0.20, 0.15, 0.10, 0.05],
            'action': ['BUY', 'BUY', 'BUY', 'BUY', 'BUY'],
            'quantity': [100, 150, 200, 50, 75],
            'order_type': ['market', 'limit', 'market', 'limit', 'market'],
            'limit_price': [None, 150.50, None, 900.00, None],
            'confidence_score': [0.85, 0.78, 0.82, 0.75, 0.71]
        })
        
        # Add timestamp
        orders['signal_date'] = datetime.now()
        
        logger.info(f"Backtest completed. Generated {len(orders)} signals")
        return orders
    
    def save_orders(self, orders: pd.DataFrame, 
                   filepath: str = 'data/backtest_results/backtest_orders.pkl'):
        """Save backtest orders to file"""
        output = {
            'orders': orders,
            'timestamp': datetime.now(),
            'strategy': self.strategy_name
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(output, f)
        
        logger.info(f"Saved backtest orders to {filepath}")
    
    def load_orders(self, filepath: str = 'data/backtest_results/backtest_orders.pkl') -> dict:
        """Load previously saved backtest orders"""
        try:
            with open(filepath, 'rb') as f:
                output = pickle.load(f)
            logger.info(f"Loaded backtest orders from {filepath}")
            return output
        except FileNotFoundError:
            logger.warning(f"No backtest orders file found at {filepath}")
            return None
    
    def get_historical_data(self, symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical data for backtesting
        Replace with actual data source (Yahoo Finance, Alpha Vantage, etc.)
        """
        # Mock historical data - replace with actual data fetching
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []
        
        for symbol in symbols:
            for date in dates:
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'open': np.random.normal(100, 10),
                    'high': np.random.normal(105, 10),
                    'low': np.random.normal(95, 10),
                    'close': np.random.normal(100, 10),
                    'volume': np.random.randint(1000000, 10000000)
                })
        
        return pd.DataFrame(data)
    
    def calculate_metrics(self, returns: pd.Series) -> dict:
        """Calculate backtest performance metrics"""
        metrics = {
            'total_return': returns.sum(),
            'annualized_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'win_rate': (returns > 0).sum() / len(returns)
        }
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
