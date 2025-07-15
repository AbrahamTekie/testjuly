from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set, Tuple
import yaml
import os
import logging
import importlib
from pathlib import Path
from core_interfaces import Plugin

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class GlobalConfig:
    """Immutable global configuration container."""
    data_path: str
    model_path: str
    initial_balance: float
    logging_level: str = "INFO"
    scaling_factor: int = 10000
    max_position_size: float = 1.0
    transaction_cost: float = 0.0001
    max_drawdown: float = 0.1
    risk_free_rate: float = 0.0

@dataclass(frozen=True)
class PluginConfig:
    """Validated plugin configuration."""
    name: str
    type: str
    class_path: str
    config: Dict[str, Any]

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Plugin name must be a non-empty string")
        if not self.class_path or '.' not in self.class_path:
            raise ValueError(f"Invalid class path for plugin {self.name}")

@dataclass
class PipelineStep:
    """Validated pipeline step configuration."""
    type: str
    component: str
    params: Dict[str, Any]

    def __post_init__(self):
        valid_types = {
            'data_loading', 'data_processing', 'feature_extraction',
            'environment_setup', 'model_training', 'prediction',
            'evaluation', 'optimization', 'trade_execution'
        }
        if self.type not in valid_types:
            raise ValueError(f"Invalid step type '{self.type}'. Must be one of {valid_types}")

class Configuration:
    def __init__(self, config_path: str):
        """Enhanced configuration loader with validation.
        
        Args:
            config_path: Path to YAML config file
        Raises:
            ValueError: On invalid configuration
            FileNotFoundError: If config file doesn't exist
            ImportError: If plugin classes can't be imported
        """
        self.config_path = Path(config_path)
        self.global_config: GlobalConfig
        self.plugins: List[PluginConfig] = []
        self.pipelines: Dict[str, List[PipelineStep]] = {}
        self._plugin_names: Set[str] = set()
        
        self._load_config()
        self._validate()

    def _load_config(self) -> None:
        """Load and parse the YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {str(e)}") from e

        # Load global config with defaults
        global_data = config_data.get('global', {})
        print(f"DEBUG: global_data loaded from YAML: {global_data}")
        self.global_config = GlobalConfig(
            data_path=global_data.get('data_path', 'data/'),
            model_path=global_data.get('model_path', 'models/'),
            initial_balance=float(global_data.get('initial_balance', 100000)),
            logging_level=global_data.get('logging_level', 'INFO'),
            scaling_factor=int(global_data.get('scaling_factor', 10000)),
            max_position_size=float(global_data.get('max_position_size', 1.0)),
            transaction_cost=float(global_data.get('transaction_cost', 0.0001)),
            max_drawdown=float(global_data.get('max_drawdown', 0.1)),
            risk_free_rate=float(global_data.get('risk_free_rate', 0.0))
        )
        print(f"DEBUG: self.global_config after creation: {self.global_config}")
        print(f"DEBUG: risk_free_rate from global_config: {getattr(self.global_config, 'risk_free_rate', None)}")

        # Load plugins with placeholder resolution
        self.plugins = [
            PluginConfig(
                name=plugin['name'],
                type=plugin['type'],
                class_path=plugin['class'],
                config=self._resolve_placeholders(plugin.get('config', {}))
            ) for plugin in config_data.get('plugins', []) if plugin.get('name')
        ]
        self._plugin_names = {p.name for p in self.plugins}
        # Debug print for risk_adjusted_reward config
        for p in self.plugins:
            if p.name == 'risk_adjusted_reward':
                print(f"risk_adjusted_reward config after placeholder resolution: {p.config}")

        # Load pipelines
        self.pipelines = {
            pipeline['name']: [
                PipelineStep(
                    type=step['type'],
                    component=step['component'],
                    params=self._resolve_placeholders(step.get('params', {}))
                ) for step in pipeline.get('steps', [])
            ] for pipeline in config_data.get('pipelines', []) if pipeline.get('name')
        }

    def _resolve_placeholders(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve ${global.*} placeholders in configuration."""
        resolved = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith('${global.') and value.endswith('}'):
                global_key = value[9:-1]
                resolved_value = getattr(self.global_config, global_key, None)
                if resolved_value is None:
                    logger.warning(f"Missing global key: {global_key}")
                    resolved[key] = value
                else:
                    # Convert to float if possible
                    try:
                        resolved[key] = float(resolved_value)
                    except (TypeError, ValueError):
                        resolved[key] = resolved_value
                    logger.debug(f"Resolved placeholder {value} to {resolved[key]} for key {key}")
            elif isinstance(value, dict):
                resolved[key] = self._resolve_placeholders(value)
            elif isinstance(value, list):
                resolved[key] = [self._resolve_placeholders(v) if isinstance(v, dict) else v for v in value]
            else:
                resolved[key] = value
        # Debug log for risk_adjusted_reward config
        if resolved.get('reward_type') == 'sortino' and 'risk_free_rate' in resolved:
            logger.debug(f"risk_adjusted_reward config after placeholder resolution: {resolved}")
        return resolved

    def _validate(self) -> None:
        """Validate the entire configuration."""
        # Validate required plugins exist
        required_plugins = set()
        for steps in self.pipelines.values():
            for step in steps:
                required_plugins.add(step.component)

        missing_plugins = required_plugins - self._plugin_names
        if missing_plugins:
            raise ValueError(f"Missing plugins required by pipelines: {missing_plugins}")

        # Validate plugin class paths
        for plugin in self.plugins:
            try:
                module_name, class_name = plugin.class_path.rsplit('.', 1)
                module = importlib.import_module(module_name)
                getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Invalid plugin class {plugin.class_path}: {str(e)}") from e

        # Validate pipeline dependencies
        for pipeline_name, steps in self.pipelines.items():
            for step in steps:
                if step.component not in self._plugin_names:
                    raise ValueError(
                        f"Pipeline '{pipeline_name}' references missing plugin '{step.component}'"
                    )

    def get_global_config(self) -> GlobalConfig:
        """Get the global configuration.
        
        Returns:
            GlobalConfig: Immutable global config
        """
        return self.global_config

    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Dict[str, Any]: Plugin configuration
        Raises:
            ValueError: If plugin not found
        """
        for plugin in self.plugins:
            if plugin.name == plugin_name:
                return plugin.config.copy()
        raise ValueError(f"Plugin {plugin_name} not found")

    def get_pipeline_config(self, pipeline_name: str) -> List[PipelineStep]:
        """Get configuration for a specific pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            List[PipelineStep]: Pipeline steps
        Raises:
            ValueError: If pipeline not found
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        return self.pipelines[pipeline_name].copy()

    def get_all_pipelines(self) -> Dict[str, List[PipelineStep]]:
        """Get all pipeline configurations.
        
        Returns:
            Dict[str, List[PipelineStep]]: All pipelines
        """
        return {name: steps.copy() for name, steps in self.pipelines.items()}

    def get_pipeline_names(self) -> List[str]:
        """Get names of all available pipelines.
        
        Returns:
            List[str]: Pipeline names
        """
        return list(self.pipelines.keys())

    def get_required_plugins(self) -> Set[str]:
        """Get names of all plugins required by pipelines.
        
        Returns:
            Set[str]: Required plugin names
        """
        required = set()
        for steps in self.pipelines.values():
            for step in steps:
                required.add(step.component)
        return required

    def __repr__(self) -> str:
        return (f"<Configuration: {len(self.plugins)} plugins, "
                f"{len(self.pipelines)} pipelines, globals={self.global_config}>")