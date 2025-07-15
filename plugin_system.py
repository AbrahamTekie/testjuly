from __future__ import annotations
import os
import logging
import pandas as pd
import numpy as np
import joblib
from stable_baselines3 import PPO
import lightgbm as lgb
import gymnasium as gym
from gymnasium import spaces
import optuna
from typing import Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from core_interfaces import (
    DataProvider, FeatureExtractor, Model, DataProcessor,
    TradingEnvironment, RewardFunction, Evaluator,
    HyperparameterOptimizer, Event, EventType, PluginManager, Plugin
)
import ta
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange

logger = logging.getLogger(__name__)

# ================== Data Providers ==================
@dataclass
class IBKRProvider(DataProvider):
    """Interactive Brokers data provider with enhanced validation."""
    config: Dict[str, Any]
    _symbol: str = "EURUSD"
    _data: Optional[pd.DataFrame] = None
    _initialized: bool = False

    def __post_init__(self):
        self._symbol = self.config.get('symbol', self._symbol)
        self.logger = logging.getLogger(f"{__name__}.IBKRProvider.{self._symbol}")

    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize with system context."""
        self._initialized = True
        self.logger.info(f"Initialized for {self._symbol}")

    def get_name(self) -> str:
        return "forex_data_provider"

    def validate(self) -> bool:
        """Validate configuration before initialization."""
        if not self.config.get('data_file'):
            raise ValueError("Missing required 'data_file' in config")
        return True

    def fetch(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Fetch market data with robust error handling."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
            
        self._symbol = symbol or self._symbol
        data_file = kwargs.get('data_file') or self.config.get('data_file')
        if not data_file:
            raise ValueError("data_file is required")
        path = Path(data_file)
        
        try:
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
                
            self.logger.info(f"Loading {self._symbol} data from {path}")
            self._data = pd.read_parquet(path)
            
            # Validate data structure
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required_cols if c not in self._data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
                
            return self._data
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise

# ================== Data Processors ==================
@dataclass 
class ForexDataProcessor(DataProcessor):
    """Enhanced forex data processor with validation."""
    config: Dict[str, Any]
    _imputer: Optional[SimpleImputer] = None
    _scalers: Dict[str, StandardScaler] = field(default_factory=dict)
    _initialized: bool = False

    def __post_init__(self):
        self.logger = logging.getLogger(f"{__name__}.ForexDataProcessor")
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate processor configuration."""
        required = ['outlier_threshold', 'fill_method']
        if any(k not in self.config for k in required):
            raise ValueError(f"Missing required config keys: {required}")

    def get_name(self) -> str:
        return "forex_data_processor"

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data with comprehensive validation."""
        if data is None or data.empty:
            raise ValueError("Received empty DataFrame")

        processed = data.copy()
        
        # Processing pipeline
        processed = self._handle_missing(processed)
        processed = self._remove_outliers(processed)
        processed = self._normalize(processed)
        
        # Final validation
        if processed.isna().any().any():
            self.logger.warning(f"NaNs detected after processing - filling with 0")
            processed = processed.fillna(0)
            
        return processed

    def initialize(self, context: Dict[str, Any]) -> None:
        # No-op for now
        pass

    def _handle_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on config."""
        method = self.config['fill_method']
        
        if method == 'ffill':
            return data.ffill().dropna()
        elif method == 'bfill':
            return data.bfill().dropna()
        elif method == 'mean':
            self._imputer = SimpleImputer(strategy='mean')
            return pd.DataFrame(
                self._imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
        else:
            raise ValueError(f"Unsupported fill method: {method}")

    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using Z-score method."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        threshold = self.config['outlier_threshold']
        
        z_scores = np.abs((data[numeric_cols] - data[numeric_cols].mean()) / 
                         data[numeric_cols].std())
        return data[(z_scores < threshold).all(axis=1)]

    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize specified features."""
        for feature in self.config.get('normalize_features', []):
            if feature in data.columns:
                self._scalers[feature] = StandardScaler()
                data[feature] = self._scalers[feature].fit_transform(
                    data[feature].to_numpy().reshape(-1, 1))
        return data

# ================== Trading Environment ==================
@dataclass
class ForexTradingEnv(TradingEnvironment, gym.Env):
    """Enhanced trading environment with type safety."""
    config: Dict[str, Any]
    _data: Optional[pd.DataFrame] = None
    _reward_function: Optional[RewardFunction] = None
    _current_step: int = 0
    _position: float = 0.0
    _balance: float = field(init=False)
    _state_history: List[Dict[str, Any]] = field(default_factory=list)
    action_space: spaces.Space = field(init=False)
    observation_space: spaces.Space = field(init=False)

    def __post_init__(self):
        self._balance = float(self.config['initial_balance'])
        self._init_spaces()
        self.logger = logging.getLogger(f"{__name__}.ForexTradingEnv")

    def _init_spaces(self) -> None:
        """Initialize Gym action/observation spaces."""
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        feature_count = len(self.config.get('features', ['close']))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(feature_count,), dtype=np.float32
        )

    def initialize(self, context: Dict[str, Any]) -> None:
        self.context = context
        if not hasattr(context, 'plugin_manager'):
            raise ValueError("Missing plugin_manager in context")
        reward_name = self.config.get('reward_function')
        if not reward_name:
            raise ValueError("No reward function specified in config")
        self.set_reward_function(reward_name)

    def set_reward_function(self, name: str) -> None:
        """Set reward function with validation."""
        plugin = self.context.plugin_manager.get_plugin(name)
        if not plugin:
            raise ValueError(f"Reward function {name} not found")
        if not isinstance(plugin, RewardFunction):
            raise TypeError(f"Plugin {name} is not a RewardFunction")
        self._reward_function = plugin
        self._reward_function.initialize({
            'data': self._data,
            'plugin_manager': self.context.plugin_manager
        })

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def state_history(self) -> list:
        return self._state_history

    def get_name(self) -> str:
        return "forex_trading_env"

    def get_state(self) -> dict:
        # Return the current state as a dict (minimal for now)
        return {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        self._current_step = 0
        return np.zeros(self.observation_space.shape), {}

    def set_data(self, data: pd.DataFrame) -> None:
        self._data = data

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Minimal stub for now
        return np.zeros(self.observation_space.shape), 0.0, False, False, {}

# ================== Models ================== 
@dataclass
class PPOModel(Model):
    """Enhanced PPO model with strict validation."""
    config: Dict[str, Any]
    _model: Optional[PPO] = None
    _env: Optional[TradingEnvironment] = None

    def __post_init__(self):
        self.logger = logging.getLogger(f"{__name__}.PPOModel")
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate model configuration."""
        required = ['learning_rate', 'n_steps', 'batch_size']
        if any(p not in self.config for p in required):
            raise ValueError(f"Missing required PPO parameters: {required}")

    def train(self, X=None, y=None, env: Optional[TradingEnvironment] = None) -> None:
        """Train model with environment validation."""
        train_env = env or self._env
        if not train_env:
            raise ValueError("No environment provided for training")
        if self._model is None:
            # PPO expects a gym.Env or VecEnv. ForexTradingEnv inherits from gym.Env, so this is safe.
            self._model = PPO(
                policy=self.config.get("policy", "MlpPolicy"),
                env=train_env,  # type: ignore
                **{k: v for k, v in self.config.items() if k not in ["policy", "env"]},
                verbose=1
            )
        self._model.learn(total_timesteps=self.config.get('total_timesteps', 10000))

    def get_name(self) -> str:
        return "ppo_model"

    def initialize(self, context: Dict[str, Any]) -> None:
        # No-op for now
        pass

    def save(self, path: str) -> None:
        if self._model is not None:
            self._model.save(path)

    def load(self, path: str) -> None:
        from stable_baselines3 import PPO
        self._model = PPO.load(path)

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        if self._model is not None and X is not None:
            return self._model.predict(X)
        return np.array([])

# ================== Evaluation ==================
@dataclass
class BacktestEngine(Evaluator):
    """Enhanced backtesting engine with metrics validation."""
    config: Dict[str, Any]
    _data: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.logger = logging.getLogger(f"{__name__}.BacktestEngine")
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate backtest configuration."""
        required = ['initial_balance', 'transaction_cost']
        if any(k not in self.config for k in required):
            raise ValueError(f"Missing required config keys: {required}")

    def evaluate(self, predictions: np.ndarray, actual: Optional[np.ndarray] = None, features: Optional[pd.DataFrame] = None) -> dict:
        """Run backtest with comprehensive validation."""
        if self._data is None:
            raise ValueError("No data available for backtesting")
            
        # ... [Backtest implementation with validation]
        
        return {
            'sharpe_ratio': 0.0,  # Example metrics
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }

    def get_name(self) -> str:
        return "backtest_engine"

    def initialize(self, context: Dict[str, Any]) -> None:
        # No-op for now
        pass

# ================== Hyperparameter Optimization ==================
@dataclass
class ForexHyperparameterOptimizer(HyperparameterOptimizer):
    """Enhanced optimizer with config validation."""
    config: Dict[str, Any]
    _best_params: Dict[str, Any] = field(default_factory=dict)
    _best_value: Optional[float] = None

    def __post_init__(self):
        self.logger = logging.getLogger(f"{__name__}.ForexHyperparameterOptimizer")
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate optimizer configuration."""
        if 'param_space' not in self.config:
            raise ValueError("Missing required 'param_space' in config")

    def optimize(self, pipeline: Any, objective: str, n_trials: int = None) -> Dict[str, Any]:
        """Run optimization with validation."""
        # ... [Optimization implementation]
        return self._best_params

    def get_name(self) -> str:
        return "forex_hyperparameter_optimizer"

    def initialize(self, context: Dict[str, Any]) -> None:
        # No-op for now
        pass

    def get_best_params(self) -> dict:
        return self._best_params or {}

# ================== Feature Engineering ==================
@dataclass
class TechnicalIndicators(FeatureExtractor):
    """Feature extractor for common technical indicators using the 'ta' library."""
    config: Dict[str, Any]
    _indicators: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.logger = logging.getLogger(f"{__name__}.TechnicalIndicators")
        self._indicators = self.config.get('indicators', [])
        if not self._indicators:
            raise ValueError("No indicators specified in config for TechnicalIndicators")

    def get_name(self) -> str:
        return "technical_indicators"

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add requested technical indicators to the DataFrame."""
        if data is None or data.empty:
            raise ValueError("Received empty DataFrame for feature extraction")
        df = data.copy()
        try:
            for ind in self._indicators:
                ind_lower = ind.lower()
                if ind_lower.startswith('rsi'):
                    period = int(ind_lower.split('_')[-1]) if '_' in ind_lower else 14
                    close_series = df['close'] if isinstance(df['close'], pd.Series) else df['close'].squeeze()
                    df[f'rsi_{period}'] = RSIIndicator(close_series.astype(float), window=period).rsi()
                elif ind_lower == 'macd':
                    close_series = df['close'] if isinstance(df['close'], pd.Series) else df['close'].squeeze()
                    macd = MACD(close_series.astype(float))
                    df['macd'] = macd.macd()
                    df['macd_signal'] = macd.macd_signal()
                    df['macd_diff'] = macd.macd_diff()
                elif ind_lower == 'bollinger_bands':
                    close_series = df['close'] if isinstance(df['close'], pd.Series) else df['close'].squeeze()
                    bb = BollingerBands(close_series.astype(float))
                    df['bollinger_upper'] = bb.bollinger_hband()
                    df['bollinger_lower'] = bb.bollinger_lband()
                elif ind_lower == 'atr':
                    high_series = df['high'] if isinstance(df['high'], pd.Series) else df['high'].squeeze()
                    low_series = df['low'] if isinstance(df['low'], pd.Series) else df['low'].squeeze()
                    close_series = df['close'] if isinstance(df['close'], pd.Series) else df['close'].squeeze()
                    df['atr'] = AverageTrueRange(high_series.astype(float), low_series.astype(float), close_series.astype(float)).average_true_range()
                else:
                    self.logger.warning(f"Unknown indicator: {ind}")
            return df
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def initialize(self, context: Dict[str, Any]) -> None:
        # No-op for now
        pass

    def get_feature_names(self) -> list:
        names = []
        for ind in self._indicators:
            ind_lower = ind.lower()
            if ind_lower.startswith('rsi'):
                period = int(ind_lower.split('_')[-1]) if '_' in ind_lower else 14
                names.append(f'rsi_{period}')
            elif ind_lower == 'macd':
                names.extend(['macd', 'macd_signal', 'macd_diff'])
            elif ind_lower == 'bollinger_bands':
                names.extend(['bollinger_upper', 'bollinger_lower'])
            elif ind_lower == 'atr':
                names.append('atr')
        return names

@dataclass
class ForexRewardFunction(RewardFunction):
    """Reward function for forex trading environments supporting Sortino and Sharpe ratios."""
    config: Dict[str, Any]
    data: Optional[pd.DataFrame] = None
    _reward_type: str = 'sortino'
    _lookback: int = 20
    _risk_free_rate: float = 0.0

    def __post_init__(self):
        self.logger = logging.getLogger(f"{__name__}.ForexRewardFunction")
        self._reward_type = self.config.get('reward_type', 'sortino')
        self._lookback = int(self.config.get('lookback', 20))
        self._risk_free_rate = float(self.config.get('risk_free_rate', 0.0))

    def get_name(self) -> str:
        return "forex_reward_function"

    def initialize(self, context: Dict[str, Any]) -> None:
        self.logger.info("ForexRewardFunction initialized with context.")
        # Optionally store context if needed

    def calculate(self, env: TradingEnvironment) -> float:
        """Calculate reward based on environment's state history."""
        # Assume env.state_history is a list of dicts with 'balance' or 'returns'
        history = getattr(env, 'state_history', [])
        if not history or len(history) < self._lookback:
            return 0.0
        # Extract returns or balance changes
        balances = [s.get('balance', 0.0) for s in history[-self._lookback-1:]]
        if len(balances) < self._lookback + 1:
            return 0.0
        returns = np.diff(balances) / np.array(balances[:-1])
        if self._reward_type == 'sortino':
            downside = returns[returns < 0]
            downside_std = downside.std() if len(downside) > 0 else 1e-8
            mean_return = returns.mean() - self._risk_free_rate / 252
            sortino = mean_return / (downside_std + 1e-8)
            return float(sortino)
        elif self._reward_type == 'sharpe':
            std = returns.std() if returns.std() > 0 else 1e-8
            mean_return = returns.mean() - self._risk_free_rate / 252
            sharpe = mean_return / std
            return float(sharpe)
        else:
            self.logger.warning(f"Unknown reward type: {self._reward_type}, returning mean return.")
            return float(returns.mean())

class DefaultPluginManager(PluginManager):
    def __init__(self):
        self._plugins = {}

    def register(self, plugin: Plugin) -> None:
        self._plugins[plugin.get_name()] = plugin

    def get_plugin(self, name: str):
        return self._plugins.get(name)

    def initialize_all(self, context):
        for plugin in self._plugins.values():
            plugin.initialize(context)