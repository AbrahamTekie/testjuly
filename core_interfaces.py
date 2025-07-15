from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Dict, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
from logging import Logger
from typing import runtime_checkable, Protocol
import asyncio

@runtime_checkable
class PluginProtocol(Protocol):
    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> None: ...
    @abstractmethod
    def get_name(self) -> str: ...

class EventType(Enum):
    DATA_LOADED = "data_loaded"
    FEATURES_GENERATED = "features_generated"
    MODEL_TRAINED = "model_trained"
    PREDICTION_MADE = "prediction_made"
    TRADE_EXECUTED = "trade_executed"
    BACKTEST_COMPLETED = "backtest_completed"
    ENVIRONMENT_INITIALIZED = "environment_initialized"
    HYPERPARAMETERS_OPTIMIZED = "hyperparameters_optimized"

@dataclass
class Event:
    type: EventType
    data: Any
    timestamp: pd.Timestamp
    metadata: Optional[dict] = None

    def __str__(self) -> str:
        return (f"Event(type={self.type.value}, data_type={type(self.data).__name__}, "
                f"ts={self.timestamp.isoformat()})")

    def __repr__(self) -> str:
        return self.__str__()

class EventHandler(ABC):
    @abstractmethod
    def handle(self, event: Event) -> None:
        """Handle an incoming event.
        
        Args:
            event: The event to process
        """
        pass

class AsyncEventHandler(ABC):
    @abstractmethod
    async def handle_async(self, event: Event) -> None:
        """Async variant of event handling."""
        pass

class EventBus:
    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._async_handlers: Dict[EventType, List[AsyncEventHandler]] = {}
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a synchronous handler.
        
        Args:
            event_type: Type of event to handle
            handler: Handler implementation
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def subscribe_async(self, event_type: EventType, handler: AsyncEventHandler) -> None:
        """Register an asynchronous handler.
        
        Args:
            event_type: Type of event to handle
            handler: Async handler implementation
        """
        if event_type not in self._async_handlers:
            self._async_handlers[event_type] = []
        self._async_handlers[event_type].append(handler)
    
    def publish(self, event: Event) -> None:
        """Publish event to all synchronous handlers.
        
        Args:
            event: Event to publish
        """
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                try:
                    handler.handle(event)
                except Exception as e:
                    logging.error(f"Error in sync handler for {event.type}: {str(e)}")

    async def publish_async(self, event: Event) -> None:
        """Publish event to all asynchronous handlers.
        
        Args:
            event: Event to publish
        """
        if event.type in self._async_handlers:
            for handler in self._async_handlers[event.type]:
                try:
                    await handler.handle_async(event)
                except Exception as e:
                    logging.error(f"Error in async handler for {event.type}: {str(e)}")

class Plugin(ABC):
    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize plugin with runtime context.
        
        Args:
            context: System context dictionary
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return unique plugin identifier.
        
        Returns:
            str: Plugin name
        """
        pass

    def validate(self) -> bool:
        """Validate plugin configuration.
        
        Returns:
            bool: True if configuration is valid
        """
        return True

class PluginManager(ABC):
    @abstractmethod
    def register(self, plugin: Plugin) -> None:
        """Register a new plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        pass
    
    @abstractmethod
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Retrieve plugin by name.
        
        Args:
            name: Plugin name to retrieve
            
        Returns:
            Optional[Plugin]: Plugin instance if found
        """
        pass
    
    @abstractmethod
    def initialize_all(self, context: Dict[str, Any]) -> None:
        """Initialize all registered plugins.
        
        Args:
            context: System context dictionary
        """
        pass

class DataProvider(Plugin, ABC):
    @abstractmethod
    def fetch(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Fetch data for given symbol.
        
        Args:
            symbol: Financial instrument symbol
            **kwargs: Additional fetch parameters
            
        Returns:
            pd.DataFrame: Retrieved data
        """
        pass

class DataProcessor(Plugin, ABC):
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process input data.
        
        Args:
            data: Input data to process
            
        Returns:
            pd.DataFrame: Processed data
        """
        pass

class FeatureExtractor(Plugin, ABC):
    @abstractmethod
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from data.
        
        Args:
            data: Input data
            
        Returns:
            pd.DataFrame: Extracted features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of all features.
        
        Returns:
            List[str]: Feature names
        """
        pass

class TradingEnvironment(Plugin, ABC):
    @property
    @abstractmethod
    def current_step(self) -> int:
        """Get current environment step.
        
        Returns:
            int: Current step index
        """
        pass

    @property
    @abstractmethod
    def state_history(self) -> List[dict]:
        """Get state history.
        
        Returns:
            List[dict]: History of states
        """
        pass
        
    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment state.
        
        Args:
            seed: Optional random seed
            options: Reset options
            
        Returns:
            Tuple[np.ndarray, dict]: Initial observation and info
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute environment step.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: 
                (observation, reward, terminated, truncated, info)
        """
        pass
    
    @abstractmethod
    def get_state(self) -> dict:
        """Get current environment state.
        
        Returns:
            dict: Current state dictionary
        """
        pass
    
    @abstractmethod
    def set_data(self, data: pd.DataFrame) -> None:
        """Set environment data.
        
        Args:
            data: Data to use for environment
        """
        pass

    @abstractmethod
    def set_reward_function(self, name: str) -> None:
        """Set reward function.
        
        Args:
            name: Name of reward function to use
        """
        pass

class RewardFunction(Plugin, ABC):
    data: Optional[pd.DataFrame] = None
    
    @abstractmethod
    def calculate(self, env: TradingEnvironment) -> float:
        """Calculate reward.
        
        Args:
            env: Environment to calculate reward from
            
        Returns:
            float: Calculated reward value
        """
        pass

class Model(Plugin, ABC):
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to path.
        
        Args:
            path: Path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from path.
        
        Args:
            path: Path to load model from
        """
        pass
    
    @abstractmethod
    def train(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, 
              env: Optional[TradingEnvironment] = None) -> None:
        """Train model.
        
        Args:
            X: Optional input features
            y: Optional target values
            env: Optional environment for RL models
        """
        pass
    
    @abstractmethod
    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate predictions.
        
        Args:
            X: Optional input features
            
        Returns:
            np.ndarray: Model predictions
        """
        pass

class Strategy(Plugin, ABC):
    @abstractmethod
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate trading signals.
        
        Args:
            features: Input features
            
        Returns:
            pd.Series: Generated signals
        """
        pass

class HyperparameterOptimizer(Plugin, ABC):
    @abstractmethod
    def optimize(self, pipeline: Any, objective: str, n_trials: Optional[int] = None) -> dict:
        """Optimize hyperparameters.
        
        Args:
            pipeline: Pipeline instance
            objective: Optimization objective
            n_trials: Number of trials
            
        Returns:
            dict: Optimized parameters
        """
        pass
    
    @abstractmethod
    def get_best_params(self) -> dict:
        """Get best found parameters.
        
        Returns:
            dict: Best parameters
        """
        pass

class Evaluator(Plugin, ABC):
    @abstractmethod
    def evaluate(self, predictions: np.ndarray, actual: Optional[np.ndarray] = None, 
                 features: Optional[pd.DataFrame] = None) -> dict:
        """Evaluate model performance.
        
        Args:
            predictions: Model predictions
            actual: Optional ground truth
            features: Optional input features
            
        Returns:
            dict: Evaluation metrics
        """
        pass