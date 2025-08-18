# Jay Algo Trader - Automated Trading System

A comprehensive automated trading system built with Quant Science methodology for running hedge fund strategies with Omega integration.

## Features

- **Omega Integration**: Complete wrapper for broker connectivity (IBKR, etc.)
- **Backtest Engine**: Nightly strategy backtesting with signal generation
- **Position Management**: Automated position reconciliation and liquidation
- **Order Execution**: Market and limit order execution with validation
- **Risk Management**: Kelly Criterion sizing, position limits, drawdown protection
- **Scheduling**: Automated daily execution after market close
- **Performance Tracking**: Comprehensive logging and performance metrics

## Architecture
```
├── src/
│   ├── core/
│   │   ├── omega_wrapper.py      # Omega library wrapper
│   │   ├── backtest_engine.py    # Backtesting logic
│   │   ├── position_manager.py   # Position management
│   │   ├── order_executor.py     # Order execution
│   │   └── risk_manager.py       # Risk management
│   ├── strategies/
│   │   └── ml_momentum.py        # ML momentum strategy
│   ├── utils/
│   │   ├── logger.py             # Logging utilities
│   │   └── config.py             # Configuration management
│   └── main.py                   # Main trading system
├── config/
│   ├── trading_config.yaml       # Trading configuration
│   └── risk_config.yaml          # Risk parameters
├── data/
│   ├── backtest_results/         # Backtest outputs
│   └── performance/              # Performance tracking
├── tests/
│   └── test_*.py                 # Unit tests
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Jay Algo Trader"
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
copy .env.example .env
# Edit .env with your API credentials
```

## Configuration

### Environment Variables

- `BROKER_API_KEY`: Your broker API key
- `BROKER_API_SECRET`: Your broker API secret
- `TRADING_MODE`: `paper` or `live`
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

### Trading Configuration

Edit `config/trading_config.yaml` to customize:
- Strategy parameters
- Execution timing
- Order preferences
- Broker settings

### Risk Configuration

Edit `config/risk_config.yaml` to set:
- Maximum drawdown limits
- Position size limits
- Sector allocation limits
- Kelly Criterion parameters

## Usage

### Run Once (Manual Execution)
```bash
python src/main.py --once
```

### Scheduled Execution (Default: 4:30 PM daily)
```bash
python src/main.py
```

### Backtest Only
```bash
python src/main.py --backtest-only
```

## Strategy Development

The system supports pluggable strategies. Create new strategies in `src/strategies/`:

```python
from src.core.backtest_engine import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self):
        # Your strategy logic here
        return signals_dataframe
```

## Risk Management

The system includes multiple risk management layers:

1. **Position Sizing**: Kelly Criterion-based optimal sizing
2. **Concentration Limits**: Maximum position and sector allocations
3. **Drawdown Protection**: Automatic position reduction on losses
4. **Order Validation**: Pre-execution risk checks

## Monitoring & Alerts

- Real-time logging to `trading_system.log`
- Performance tracking in CSV format
- Email/SMS alerts for critical events
- Position reconciliation reports

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Production Deployment

1. Set `TRADING_MODE=live` in environment
2. Configure proper API credentials
3. Set up monitoring and alerting
4. Schedule with task scheduler or cron
5. Implement backup and recovery procedures

## Disclaimer

This system is for educational purposes. Always test thoroughly in paper trading mode before using with real money. Trading involves risk of loss.

## Support

For issues and questions, please check the logs first, then create an issue with:
- Error messages
- Configuration details
- Steps to reproduce
- System information
