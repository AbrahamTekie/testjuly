import logging
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from orchestrator import Pipeline
from configuration import Configuration
from datetime import datetime
from error_handler import error_handler, ErrorSeverity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('usage_examples.log')
    ]
)
logger = logging.getLogger(__name__)

def make_json_safe(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return obj

class TradingSystemExample:
    def __init__(self, config_path: str = 'pipeline_config.yaml'):
        """Initialize with type-checked configuration."""
        self.config_path = Path(config_path)
        self._validate_config()
        self.pipeline = Pipeline(str(self.config_path))
        self.context = self.pipeline.context

    def _validate_config(self) -> None:
        """Pre-execution configuration validation."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        try:
            Configuration(str(self.config_path))
        except Exception as e:
            logger.error(f"Invalid configuration: {str(e)}")
            raise

    async def train_model_async(self, symbol: str = 'EURUSD',
                              timeframe: str = 'H1',
                              model_path: str = 'models/ppo_model.zip') -> Dict[str, Any]:
        """Async training example with progress tracking."""
        logger.info(f"Async training started for {symbol} {timeframe}")
        
        def progress_callback(current: int, total: int):
            logger.info(f"Progress: {current}/{total} steps")
            
        try:
            # Execute pipeline asynchronously
            results = await self.pipeline.execute_pipeline_async(
                'train',
                symbol=symbol,
                timeframe=timeframe,
                model_path=model_path,
                progress_callback=progress_callback
            )
            
            # Validate outputs
            self._validate_training_outputs()
            
            # Save model
            model = self.context.model
            if model:
                model.save(model_path)
                logger.info(f"Model saved to {model_path}")
                
            return results
            
        except Exception as e:
            logger.exception("Async training failed")
            raise

    def train_model(self, **kwargs) -> Dict[str, Any]:
        """Sync wrapper for async training."""
        return self.pipeline.execute_pipeline('train', **kwargs)

    def _validate_training_outputs(self) -> None:
        """Validate pipeline outputs after training."""
        if self.context.data is not None:
            logger.info(f"Training data shape: {self.context.data.shape}")
            self._check_for_nans(self.context.data, "training data")
            
        if self.context.features is not None:
            logger.info(f"Feature matrix shape: {self.context.features.shape}")
            self._check_for_nans(self.context.features, "features")

    def _check_for_nans(self, df: pd.DataFrame, name: str) -> None:
        """Data quality check with automatic repair."""
        if df.isna().any().any():
            nan_count = df.isna().sum().sum()
            logger.warning(f"Found {nan_count} NaNs in {name}")
            df.fillna(0, inplace=True)

    def backtest(self, symbol: str = 'EURUSD',
                timeframe: str = 'H1',
                model_path: str = 'models/ppo_model.zip',
                results_path: str = 'backtest_results.json') -> Dict[str, Any]:
        """Complete backtest workflow with visualization."""
        logger.info(f"Starting backtest for {symbol} {timeframe}")
        
        try:
            # Load or train model
            model = self.pipeline.plugin_manager.get_plugin('ppo_model')
            if model and os.path.exists(model_path):
                model.load(model_path)
                logger.info(f"Loaded model from {model_path}")
            else:
                logger.warning("Model not found - training first...")
                self.train_model(symbol=symbol, timeframe=timeframe, model_path=model_path)
            
            # Execute backtest
            results = self.pipeline.execute_pipeline(
                'backtest',
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Save and visualize
            self._save_results(results, results_path)
            self._visualize_backtest(results, symbol, timeframe)
            
            return results
            
        except Exception as e:
            logger.exception("Backtest failed")
            raise

    def optimize(self, symbol: str = 'EURUSD',
                timeframe: str = 'H1',
                trials: int = 50,
                results_path: str = 'optimization_results.json') -> Dict[str, Any]:
        """Hyperparameter optimization workflow."""
        logger.info(f"Optimizing with {trials} trials")
        
        try:
            results = self.pipeline.execute_pipeline(
                'optimize',
                symbol=symbol,
                timeframe=timeframe,
                n_trials=trials
            )
            
            self._save_results(results, results_path)
            
            if 'best_params' in results:
                logger.info("Optimized Parameters:")
                for param, value in results['best_params'].items():
                    logger.info(f"{param}: {value}")
                    
            return results
            
        except Exception as e:
            logger.exception("Optimization failed")
            raise

    def run_pipeline(self, pipeline_name: str, **kwargs) -> Dict[str, Any]:
        """Generic pipeline executor."""
        logger.info(f"Running pipeline: {pipeline_name}")
        return self.pipeline.execute_pipeline(pipeline_name, **kwargs)

    def _save_results(self, results: Dict[str, Any], path: str) -> None:
        """Save results with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{Path(path).stem}_{timestamp}{Path(path).suffix}"
        
        with open(path, 'w') as f:
            json.dump(make_json_safe(results), f, indent=2)
        logger.info(f"Results saved to {path}")

    def _visualize_backtest(self, results: Dict[str, Any], 
                          symbol: str, timeframe: str) -> None:
        """Generate professional visualizations."""
        if 'equity' not in results:
            logger.warning("No equity curve to visualize")
            return
            
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity Curve
        ax1.plot(results['equity'], label='Equity', color='royalblue')
        ax1.set_title(f'{symbol} {timeframe} Backtest Results')
        ax1.set_ylabel('Balance')
        ax1.grid(True)
        ax1.legend()
        
        # Drawdown
        peak = pd.Series(results['equity']).cummax()
        drawdown = (peak - results['equity']) / peak
        ax2.fill_between(range(len(drawdown)), drawdown, color='crimson', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"backtest_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(plot_path)
        logger.info(f"Saved visualization to {plot_path}")
        plt.close()

def run_full_workflow():
    """Complete example workflow from training to trading."""
    try:
        system = TradingSystemExample()
        
        # 1. Train
        train_results = system.train_model(
            symbol='EURUSD',
            timeframe='H1',
            model_path='models/eurusd_h1_model.zip'
        )
        
        # 2. Optimize
        opt_results = system.optimize(
            symbol='EURUSD',
            timeframe='H1',
            trials=30
        )
        
        # 3. Backtest
        backtest_results = system.backtest(
            symbol='EURUSD',
            timeframe='H1',
            model_path='models/eurusd_h1_model.zip'
        )
        
        return {
            'training': train_results,
            'optimization': opt_results,
            'backtest': backtest_results
        }
        
    except Exception as e:
        logger.exception("Full workflow failed")
        raise

if __name__ == "__main__":
    try:
        # Example executions
        system = TradingSystemExample()
        
        # Run full workflow
        results = run_full_workflow()
        
        # Or run individual components
        # results = system.train_model()
        # results = system.backtest()
        # results = system.optimize(trials=20)
        # results = system.run_pipeline('custom_pipeline')
        
        logger.info("All examples completed successfully")
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")