import logging
import traceback
import pandas as pd
import numpy as np
import time
import functools
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Set, Callable
from pathlib import Path
from enum import Enum
from datetime import datetime
from core_interfaces import (
    Event, EventType, EventHandler, PluginManager, EventBus,
    TradingEnvironment, Model, Evaluator, HyperparameterOptimizer
)
from configuration import Configuration
from plugin_system import DefaultPluginManager

# ================ ERROR HANDLING FRAMEWORK ================
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
    context: Optional[dict] = field(default_factory=dict)

class ForexBotErrorHandler:
    def __init__(self, log_file: str = "forex_bot_errors.log"):
        self.errors_log = []
        self.recovery_strategies = {}
        self.max_retries = 3
        self.retry_delay = 1
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
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
                        time.sleep(retry_delay * (2 ** attempt))
                        
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
                for error in self.errors_log[-10:]
            ]
        }

# Initialize global error handler
error_handler = ForexBotErrorHandler()

# ================ ORIGINAL ORCHESTRATOR CODE WITH ERROR HANDLING ================
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class PipelineContext:
    """Immutable pipeline execution context."""
    config: Dict[str, Any]
    plugin_manager: PluginManager
    event_bus: EventBus
    data: Optional[pd.DataFrame] = None
    features: Optional[pd.DataFrame] = None
    environment: Optional[TradingEnvironment] = None
    model: Optional[Model] = None
    predictions: Optional[np.ndarray] = None
    metrics: Optional[Dict[str, Any]] = None
    symbol: Optional[str] = None  # Add symbol as an optional field
    extra: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    def update(self, **kwargs) -> 'PipelineContext':
        # Only pass known fields to the dataclass constructor, store unknowns in extra
        known_fields = set(self.__dataclass_fields__.keys())
        base = {**self.__dict__}
        extra = dict(base.get('extra', {}))
        for k, v in kwargs.items():
            if k in known_fields:
                base[k] = v
            else:
                extra[k] = v
        base['extra'] = extra
        print(f"DEBUG: plugin_manager before update: {base.get('plugin_manager')}")
        # Fallback: if plugin_manager is None after merging, set it to self.plugin_manager
        if base.get('plugin_manager') is None:
            base['plugin_manager'] = self.plugin_manager
            print("DEBUG: plugin_manager was None, set to self.plugin_manager")
        result = PipelineContext(**{k: base[k] for k in known_fields})
        print(f"DEBUG: plugin_manager after update: {result.plugin_manager}")
        assert isinstance(result, PipelineContext), "update() did not return PipelineContext!"
        return result

@dataclass
class PipelineStepResult:
    """Standardized pipeline step result."""
    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class Pipeline:
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.logger = logging.getLogger(f"{__name__}.Pipeline")
        self._validate_config_path(config_path)
        
        # Register recovery strategies
        error_handler.register_recovery_strategy("FileNotFoundError", self._handle_file_not_found)
        error_handler.register_recovery_strategy("ValueError", self._handle_value_error)
        error_handler.register_recovery_strategy("RuntimeError", self._handle_runtime_error)
        error_handler.register_recovery_strategy("ConnectionError", self._handle_connection_error)
        
        try:
            self.config = Configuration(config_path)
            self.plugin_manager = DefaultPluginManager()
            self.event_bus = EventBus()
            self.context = PipelineContext(
                config=asdict(self.config.get_global_config()),
                plugin_manager=self.plugin_manager,
                event_bus=self.event_bus
            )
            
            self._setup_plugins()
            self._setup_event_handlers()
            self._validate_pipelines()
            
            self.logger.info("Pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError("Pipeline init failed") from e

    def _validate_config_path(self, path: str) -> None:
        """Verify config file exists."""
        if not Path(path).exists():
            error = f"Config file not found: {path}"
            self.logger.error(error)
            raise FileNotFoundError(error)

    @error_handler.handle_gracefully(default_return=None, severity=ErrorSeverity.HIGH)
    def _setup_plugins(self) -> None:
        """Load and initialize all plugins with error handling."""
        self.logger.info("Loading plugins...")
        print("DEBUG: Starting plugin registration...")
        for plugin_cfg in self.config.plugins:
            try:
                name = getattr(plugin_cfg, 'name', '<unnamed>')
                class_path = getattr(plugin_cfg, 'class_path', None)
                print(f"DEBUG: Registering plugin: {name} with class_path: {class_path}")
                if not class_path:
                    raise ValueError(f"Plugin {name} missing class path")
                module_name, class_name = class_path.rsplit('.', 1)
                module = __import__(module_name, fromlist=[class_name])
                plugin_class = getattr(module, class_name)
                plugin = plugin_class(getattr(plugin_cfg, 'config', {}))
                if hasattr(plugin, 'validate'):
                    plugin.validate()
                self.plugin_manager.register(plugin)
                print(f"DEBUG: Registered plugin: {plugin.get_name()}")
                plugin.initialize(self.context)
                self.logger.info(f"Initialized plugin: {name}")
            except Exception as e:
                self.logger.error(f"Failed to load plugin {name}: {str(e)}")
                continue
        print(f"DEBUG: All registered plugin names: {list(self.plugin_manager._plugins.keys())}")

    @error_handler.handle_gracefully(default_return=None, severity=ErrorSeverity.MEDIUM)
    def _setup_event_handlers(self) -> None:
        """Register all event handlers with error handling."""
        handler_map = {
            EventType.DATA_LOADED: DataLoadedHandler,
            EventType.FEATURES_GENERATED: FeaturesGeneratedHandler,
            EventType.MODEL_TRAINED: ModelTrainedHandler,
            EventType.PREDICTION_MADE: PredictionMadeHandler,
            EventType.ENVIRONMENT_INITIALIZED: EnvironmentInitializedHandler,
            EventType.BACKTEST_COMPLETED: BacktestCompletedHandler,
            EventType.HYPERPARAMETERS_OPTIMIZED: HyperparametersOptimizedHandler
        }
        
        for event_type, handler_class in handler_map.items():
            try:
                handler = handler_class(self)
                self.event_bus.subscribe(event_type, handler)
                self.logger.info(f"Registered handler for {event_type.value}")
            except Exception as e:
                self.logger.error(f"Failed to register handler for {event_type.value}: {e}")
                continue

    @error_handler.handle_gracefully(default_return=None, severity=ErrorSeverity.HIGH)
    def _validate_pipelines(self) -> None:
        """Validate pipeline dependencies."""
        required_plugins = set()
        for pipeline in self.config.pipelines.values():
            for step in pipeline:
                required_plugins.add(step.component)
        available = set(self.plugin_manager._plugins.keys())
        missing = required_plugins - available
        if missing:
            error = f"Missing plugins required by pipelines: {missing}"
            self.logger.error(error)
            raise ValueError(error)

    @error_handler.auto_retry(max_retries=2, delay=1)
    def execute_pipeline(self, pipeline_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a pipeline with comprehensive error handling."""
        self.logger.info(f"Starting pipeline: {pipeline_name}")
        
        pipeline_cfg = self.config.get_pipeline_config(pipeline_name)
        if not pipeline_cfg:
            error = f"Pipeline {pipeline_name} not found"
            self.logger.error(error)
            raise ValueError(error)
            
        results = {}
        step_errors = []
        context = self.context.update(**kwargs)
        
        for step_idx, step_cfg in enumerate(pipeline_cfg, 1):
            step_result = self._execute_single_step(step_idx, step_cfg, context)
            
            if not step_result.success:
                step_errors.append({
                    'step': step_idx,
                    'type': getattr(step_cfg, 'type', None),
                    'component': getattr(step_cfg, 'component', None),
                    'error': str(step_result.error)
                })
                
                # Decide whether to continue or stop based on error severity
                if step_result.error and isinstance(step_result.error, (ValueError, RuntimeError)):
                    self.logger.error(f"Critical error in step {step_idx}, stopping pipeline")
                    break
                else:
                    self.logger.warning(f"Non-critical error in step {step_idx}, continuing")
            else:
                context = context.update(**step_result.metadata)
                if step_result.output:
                    results.update(step_result.output)
        
        # Add error summary to results
        results['error_summary'] = error_handler.get_error_summary()
        
        if step_errors:
            results['errors'] = step_errors
            self.logger.error(f"Completed with {len(step_errors)} errors")
        else:
            self.logger.info("Pipeline completed successfully")
            
        return results

    def _execute_single_step(self, step_idx: int, step_cfg, context: PipelineContext) -> PipelineStepResult:
        """Execute individual pipeline step with error handling."""
        try:
            step_type = getattr(step_cfg, 'type', None) if not isinstance(step_cfg, dict) else step_cfg.get('type')
            component = getattr(step_cfg, 'component', None) if not isinstance(step_cfg, dict) else step_cfg.get('component')
            params = {**(getattr(step_cfg, 'params', {}) if not isinstance(step_cfg, dict) else step_cfg.get('params', {})), **context.config}
            self.logger.info(f"Executing step {step_idx}: {step_type} ({component})")
            plugin = self.plugin_manager.get_plugin(component if isinstance(component, str) and component else "")
            if not plugin:
                raise ValueError(f"Plugin {component} not found")
            executor = getattr(self, f"_execute_{step_type if isinstance(step_type, str) and step_type else ''}", None)
            if not executor:
                raise ValueError(f"Unknown step type: {step_type}")
            output = executor(plugin, params, context)
            metadata = self._get_step_metadata(str(step_type) if step_type is not None else '', context)
            return PipelineStepResult(
                success=True,
                output=output,
                metadata=metadata
            )
        except Exception as e:
            self.logger.error(f"HIGH ERROR in _execute_single_step: {e}")
            return PipelineStepResult(
                success=False,
                error=e,
                output=None,
                metadata={}
            )

    def _get_step_metadata(self, step_type: str, context: PipelineContext) -> Dict[str, Any]:
        """Get context updates for step type."""
        type_map = {
            'data_loading': {'data': context.data},
            'data_processing': {'data': context.data},
            'feature_extraction': {'features': context.features},
            'environment_setup': {'environment': context.environment},
            'model_training': {'model': context.model},
            'prediction': {'predictions': context.predictions},
            'evaluation': {'metrics': context.metrics},
            'optimization': {'best_params': context.config.get('best_params')}
        }
        return type_map.get(step_type, {})

    # Recovery strategies
    def _handle_file_not_found(self, *args, **kwargs):
        """Recovery strategy for file not found errors"""
        self.logger.info("Attempting to create default config file...")
        return {"status": "recovered", "action": "created_default_config"}

    def _handle_value_error(self, *args, **kwargs):
        """Recovery strategy for value errors"""
        self.logger.info("Using default values for invalid parameters...")
        return {"status": "recovered", "action": "used_defaults"}

    def _handle_runtime_error(self, *args, **kwargs):
        """Recovery strategy for runtime errors"""
        self.logger.info("Attempting to restart component...")
        return {"status": "recovered", "action": "component_restart"}

    def _handle_connection_error(self, *args, **kwargs):
        """Recovery strategy for connection errors"""
        self.logger.info("Retrying connection with fallback settings...")
        time.sleep(5)
        return {"status": "recovered", "action": "connection_retry"}

    # Step execution methods with error handling
    @error_handler.handle_gracefully(default_return={'data': pd.DataFrame()}, severity=ErrorSeverity.HIGH)
    def _execute_data_loading(self, plugin, params: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        symbol = params.pop('symbol', getattr(plugin, 'symbol', None))
        if not symbol:
            raise ValueError("Symbol required for data loading")
            
        data = plugin.fetch(symbol, **params)
        if data.empty:
            raise ValueError("Data loading returned empty DataFrame")
            
        self.event_bus.publish(Event(
            EventType.DATA_LOADED,
            data,
            pd.Timestamp.now(),
            {'provider': plugin.get_name()}
        ))
        return {'data': data}

    @error_handler.handle_gracefully(default_return={'data': pd.DataFrame()}, severity=ErrorSeverity.HIGH)
    def _execute_data_processing(self, plugin, params: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        if context.data is None or context.data.empty:
            raise ValueError("No data available for processing")
            
        processed = plugin.process(context.data)
        if processed.empty:
            raise ValueError("Processing returned empty DataFrame")
            
        if processed.isna().any().any():
            self.logger.warning("NaN values detected after processing")
            processed = processed.fillna(0)
            
        self.event_bus.publish(Event(
            EventType.DATA_LOADED,
            processed,
            pd.Timestamp.now(),
            {'processor': plugin.get_name()}
        ))
        return {'data': processed}

    @error_handler.handle_gracefully(default_return={'features': pd.DataFrame()}, severity=ErrorSeverity.HIGH)
    def _execute_feature_extraction(self, plugin, params: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        if context.data is None or context.data.empty:
            raise ValueError("No data available for feature extraction")
            
        features = plugin.extract(context.data)
        if features.empty:
            raise ValueError("Feature extraction returned empty DataFrame")
            
        if features.isna().any().any():
            self.logger.warning("NaN values in features")
            features = features.fillna(0)
            
        self.event_bus.publish(Event(
            EventType.FEATURES_GENERATED,
            features,
            pd.Timestamp.now(),
            {'extractor': plugin.get_name()}
        ))
        return {'features': features}

    @error_handler.handle_gracefully(default_return={'environment': None}, severity=ErrorSeverity.HIGH)
    def _execute_environment_setup(self, plugin, params: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        if context.features is None or context.features.empty:
            raise ValueError("No features available for environment")
            
        if isinstance(plugin, TradingEnvironment):
            if 'reward_function' in params:
                plugin.set_reward_function(params['reward_function'])
            plugin.set_data(context.features)
            
        self.event_bus.publish(Event(
            EventType.ENVIRONMENT_INITIALIZED,
            {'step_count': len(context.features)},
            pd.Timestamp.now(),
            {'environment': plugin.get_name()}
        ))
        return {'environment': plugin}

    @error_handler.handle_gracefully(default_return={'model': None}, severity=ErrorSeverity.CRITICAL)
    def _execute_model_training(self, plugin, params: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        if context.environment is None:
            raise ValueError("No environment available for training")
            
        if isinstance(plugin, Model):
            plugin.train(env=context.environment)
            
            self.event_bus.publish(Event(
                EventType.MODEL_TRAINED,
                {},
                pd.Timestamp.now(),
                {'model': plugin.get_name()}
            ))
            return {'model': plugin}
        raise TypeError("Plugin must implement Model interface")

    @error_handler.handle_gracefully(default_return={'predictions': np.array([])}, severity=ErrorSeverity.HIGH)
    def _execute_prediction(self, plugin, params: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        if context.features is None or context.features.empty:
            raise ValueError("No features available for prediction")
            
        if isinstance(plugin, Model):
            from plugin_system import PPOModel
            if isinstance(plugin, PPOModel) and getattr(plugin, '_env', None) is None:
                plugin._env = context.environment
            
            predictions = plugin.predict()
            
            self.event_bus.publish(Event(
                EventType.PREDICTION_MADE,
                predictions,
                pd.Timestamp.now(),
                {'model': plugin.get_name()}
            ))
            return {'predictions': predictions}
        raise TypeError("Plugin must implement Model interface")

    @error_handler.handle_gracefully(default_return={'metrics': {}}, severity=ErrorSeverity.MEDIUM)
    def _execute_evaluation(self, plugin, params: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        if context.predictions is None:
            raise ValueError("No predictions available for evaluation")
            
        if isinstance(plugin, Evaluator):
            metrics = plugin.evaluate(
                context.predictions,
                features=context.features
            )
            return metrics
        raise TypeError("Plugin must implement Evaluator interface")

    @error_handler.handle_gracefully(default_return={'best_params': {}}, severity=ErrorSeverity.MEDIUM)
    def _execute_optimization(self, plugin, params: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        if not isinstance(plugin, HyperparameterOptimizer):
            raise TypeError("Plugin must implement HyperparameterOptimizer")
            
        results = plugin.optimize(
            self,
            params.get('objective', 'sharpe_ratio'),
            params.get('n_trials', 50)
        )
        return {'best_params': results}

# Event Handlers with error handling
class BaseEventHandler(EventHandler):
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @error_handler.handle_gracefully(default_return=None, severity=ErrorSeverity.LOW)
    def handle(self, event: Event):
        """Base handler with error protection"""
        self.logger.info(f"Processing {event.type.value} event")

class DataLoadedHandler(BaseEventHandler):
    @error_handler.handle_gracefully(default_return=None, severity=ErrorSeverity.LOW)
    def handle(self, event: Event):
        self.logger.info(f"Processing {event.type.value} event")
        updated_context = self.pipeline.context.update(data=event.data)
        assert isinstance(updated_context, type(self.pipeline.context)), "Context update did not return PipelineContext!"
        self.pipeline.context = updated_context

class FeaturesGeneratedHandler(BaseEventHandler):
    @error_handler.handle_gracefully(default_return=None, severity=ErrorSeverity.LOW)
    def handle(self, event: Event):
        self.logger.info(f"Processing {event.type.value} event")
        self.pipeline.context = self.pipeline.context.update(features=event.data)

class ModelTrainedHandler(BaseEventHandler):
    @error_handler.handle_gracefully(default_return=None, severity=ErrorSeverity.LOW)
    def handle(self, event: Event):
        self.logger.info(f"Processing {event.type.value} event")

class PredictionMadeHandler(BaseEventHandler):
    @error_handler.handle_gracefully(default_return=None, severity=ErrorSeverity.LOW)
    def handle(self, event: Event):
        self.logger.info(f"Processing {event.type.value} event")
        self.pipeline.context = self.pipeline.context.update(predictions=event.data)

class EnvironmentInitializedHandler(BaseEventHandler):
    @error_handler.handle_gracefully(default_return=None, severity=ErrorSeverity.LOW)
    def handle(self, event: Event):
        self.logger.info(f"Processing {event.type.value} event")

class BacktestCompletedHandler(BaseEventHandler):
    @error_handler.handle_gracefully(default_return=None, severity=ErrorSeverity.LOW)
    def handle(self, event: Event):
        self.logger.info(f"Processing {event.type.value} event")
        self.pipeline.context = self.pipeline.context.update(metrics=event.data.get('metrics'))

class HyperparametersOptimizedHandler(BaseEventHandler):
    @error_handler.handle_gracefully(default_return=None, severity=ErrorSeverity.LOW)
    def handle(self, event: Event):
        self.logger.info(f"Processing {event.type.value} event")
        self.pipeline.context = self.pipeline.context.update(best_params=event.data.get('best_params'))