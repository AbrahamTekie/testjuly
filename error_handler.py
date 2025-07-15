import logging
import traceback
import functools
import time
from typing import Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import json
import sys
from datetime import datetime

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorInfo:
    timestamp: datetime
    error_type: str
    message: str
    severity: ErrorSeverity
    function_name: str
    traceback: str
    context: dict

class ForexBotErrorHandler:
    def __init__(self, log_file: str = "forex_bot_errors.log"):
        self.errors_log = []
        self.recovery_strategies = {}
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register recovery strategies for specific error types"""
        self.recovery_strategies[error_type] = strategy
        
    def auto_retry(self, max_retries: int = None, delay: float = None):
        """Decorator for automatic retry with exponential backoff"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                retries = max_retries or self.max_retries
                retry_delay = delay or self.retry_delay
                
                for attempt in range(retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == retries:
                            self._log_error(e, func.__name__, ErrorSeverity.HIGH)
                            raise
                        
                        self.logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        
            return wrapper
        return decorator
    
    def handle_gracefully(self, default_return: Any = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Decorator to handle errors gracefully with fallback values"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._log_error(e, func.__name__, severity)
                    
                    # Try recovery strategy if available
                    error_type = type(e).__name__
                    if error_type in self.recovery_strategies:
                        try:
                            return self.recovery_strategies[error_type](*args, **kwargs)
                        except Exception as recovery_error:
                            self._log_error(recovery_error, f"{func.__name__}_recovery", ErrorSeverity.HIGH)
                    
                    return default_return
            return wrapper
        return decorator
    
    def _log_error(self, error: Exception, function_name: str, severity: ErrorSeverity, context: dict = None):
        """Log error with detailed information"""
        error_info = ErrorInfo(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            function_name=function_name,
            traceback=traceback.format_exc(),
            context=context or {}
        )
        
        self.errors_log.append(error_info)
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR in {function_name}: {error}")
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH ERROR in {function_name}: {error}")
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM ERROR in {function_name}: {error}")
        else:
            self.logger.info(f"LOW ERROR in {function_name}: {error}")
    
    def get_error_summary(self) -> dict:
        """Get summary of all errors"""
        error_counts = {}
        for error in self.errors_log:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
        
        return {
            "total_errors": len(self.errors_log),
            "error_types": error_counts,
            "recent_errors": [
                {
                    "timestamp": error.timestamp.isoformat(),
                    "type": error.error_type,
                    "message": error.message,
                    "severity": error.severity.value,
                    "function": error.function_name
                }
                for error in self.errors_log[-10:]  # Last 10 errors
            ]
        }

# Initialize global error handler
error_handler = ForexBotErrorHandler()

# Example usage for DRL Forex Bot
class ForexBot:
    def __init__(self):
        self.balance = 10000
        self.positions = []
        
        # Register recovery strategies
        error_handler.register_recovery_strategy("ConnectionError", self._handle_connection_error)
        error_handler.register_recovery_strategy("DataError", self._handle_data_error)
        error_handler.register_recovery_strategy("ModelError", self._handle_model_error)
    
    @error_handler.auto_retry(max_retries=3, delay=2)
    def connect_to_broker(self):
        """Connect to forex broker with automatic retry"""
        # Simulate connection logic
        import random
        if random.random() < 0.3:  # 30% chance of failure
            raise ConnectionError("Failed to connect to broker")
        return "Connected successfully"
    
    @error_handler.handle_gracefully(default_return=[], severity=ErrorSeverity.MEDIUM)
    def get_market_data(self):
        """Get market data with graceful error handling"""
        import random
        if random.random() < 0.2:  # 20% chance of failure
            raise ValueError("Invalid market data received")
        return [1.2345, 1.2346, 1.2347]  # Mock data
    
    @error_handler.handle_gracefully(default_return=0, severity=ErrorSeverity.HIGH)
    def execute_trade(self, action, amount):
        """Execute trade with error handling"""
        if amount <= 0:
            raise ValueError("Invalid trade amount")
        if action not in ["buy", "sell"]:
            raise ValueError("Invalid trade action")
        
        # Simulate trade execution
        import random
        if random.random() < 0.1:  # 10% chance of failure
            raise RuntimeError("Trade execution failed")
        
        return amount * 0.01  # Mock profit
    
    def _handle_connection_error(self, *args, **kwargs):
        """Recovery strategy for connection errors"""
        self.logger.info("Attempting to reconnect...")
        time.sleep(5)
        return "Reconnection attempted"
    
    def _handle_data_error(self, *args, **kwargs):
        """Recovery strategy for data errors"""
        self.logger.info("Using cached data...")
        return [1.2340, 1.2341, 1.2342]  # Cached data
    
    def _handle_model_error(self, *args, **kwargs):
        """Recovery strategy for model errors"""
        self.logger.info("Switching to backup model...")
        return 0  # Safe default action

# Health monitoring system
class HealthMonitor:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.health_checks = []
        
    def add_health_check(self, name: str, check_func: Callable):
        """Add health check function"""
        self.health_checks.append((name, check_func))
    
    def run_health_checks(self):
        """Run all health checks"""
        results = {}
        for name, check_func in self.health_checks:
            try:
                results[name] = check_func()
            except Exception as e:
                results[name] = f"FAILED: {e}"
                error_handler._log_error(e, f"health_check_{name}", ErrorSeverity.MEDIUM)
        return results

# Example usage
if __name__ == "__main__":
    # Create bot instance
    bot = ForexBot()
    
    # Setup health monitoring
    monitor = HealthMonitor(bot)
    monitor.add_health_check("broker_connection", lambda: bot.connect_to_broker())
    monitor.add_health_check("data_feed", lambda: len(bot.get_market_data()) > 0)
    
    # Run some operations
    try:
        bot.connect_to_broker()
        data = bot.get_market_data()
        profit = bot.execute_trade("buy", 100)
        
        # Run health checks
        health_status = monitor.run_health_checks()
        print("Health Status:", health_status)
        
        # Get error summary
        error_summary = error_handler.get_error_summary()
        print("Error Summary:", json.dumps(error_summary, indent=2))
        
    except Exception as e:
        print(f"Unhandled error: {e}")
        error_handler._log_error(e, "main", ErrorSeverity.CRITICAL)