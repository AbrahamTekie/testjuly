import click
import logging
import json
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from orchestrator import Pipeline
from configuration import Configuration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_system.log')
    ]
)
logger = logging.getLogger(__name__)

class CLIState:
    """Shared state for CLI commands."""
    def __init__(self):
        self.config_path: Optional[str] = None
        self.verbose: bool = False
        self.pipeline: Optional[Pipeline] = None

pass_state = click.make_pass_decorator(CLIState, ensure=True)

@click.group()
@click.option('--config', '-c', default='pipeline_config.yaml',
              help='Path to configuration file', show_default=True)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@pass_state
def cli(state: CLIState, config: str, verbose: bool):
    """Algorithmic Trading System CLI"""
    state.config_path = config
    state.verbose = verbose
    
    # Validate config early
    if not Path(config).exists():
        logger.error(f"Config file not found: {config}")
        click.secho(f"Error: Config file not found at {config}", fg='red')
        sys.exit(1)
        
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")

@cli.command()
@click.option('--symbol', '-s', default='EURUSD', 
              help='Trading symbol (e.g., EURUSD)', show_default=True)
@click.option('--timeframe', '-t', default='H1',
              help='Timeframe (e.g., H1, D1)', show_default=True)
@click.option('--model-path', '-m', default='models/ppo_model.zip',
              help='Path to save trained model', show_default=True)
@pass_state
def train(state: CLIState, symbol: str, timeframe: str, model_path: str):
    """Train the trading model with enhanced validation"""
    try:
        logger.info(f"Starting training for {symbol} {timeframe}")
        
        # Initialize pipeline
        state.pipeline = Pipeline(state.config_path)
        
        # Execute with progress tracking
        with click.progressbar(
            length=100, 
            label='Training progress'
        ) as bar:
            def update_progress(current: int, total: int):
                bar.pos = int((current / total) * 100)
                bar.update(0)
                
            results = state.pipeline.execute_pipeline(
                'train',
                symbol=symbol,
                timeframe=timeframe,
                model_path=model_path,
                progress_callback=update_progress
            )
            
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"train_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        click.secho(f"\nTraining completed! Results saved to {output_file}", fg='green')
        if 'errors' in results:
            click.secho(f"Warnings: {len(results['errors'])} non-critical errors", fg='yellow')
            
    except Exception as e:
        logger.exception("Training failed")
        click.secho(f"Error: {str(e)}", fg='red')
        sys.exit(1)

@cli.command()
@click.option('--symbol', '-s', default='EURUSD',
              help='Trading symbol', show_default=True)
@click.option('--timeframe', '-t', default='H1',
              help='Timeframe', show_default=True)
@click.option('--model-path', '-m', default='models/ppo_model.zip',
              help='Path to trained model', show_default=True)
@click.option('--output', '-o', default='backtest_results.json',
              help='Output file', show_default=True)
@pass_state
def backtest(state: CLIState, symbol: str, timeframe: str, 
             model_path: str, output: str):
    """Run backtest with comprehensive validation"""
    try:
        logger.info(f"Starting backtest for {symbol} {timeframe}")
        
        state.pipeline = Pipeline(state.config_path)
        
        # Model loading with fallback
        model_plugin = state.pipeline.plugin_manager.get_plugin('ppo_model')
        if model_plugin and os.path.exists(model_path):
            model_plugin.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            click.secho("Model not found - training new model...", fg='yellow')
            state.pipeline.execute_pipeline('train', symbol=symbol, timeframe=timeframe)
        
        # Execute backtest
        results = state.pipeline.execute_pipeline(
            'backtest',
            symbol=symbol,
            timeframe=timeframe
        )
        
        # Save and display results
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
            
        click.secho(f"\nBacktest results saved to {output}", fg='green')
        click.echo(json.dumps(results, indent=2))
        
    except Exception as e:
        logger.exception("Backtest failed")
        click.secho(f"Error: {str(e)}", fg='red')
        sys.exit(1)

@cli.command()
@click.option('--symbol', '-s', default='EURUSD',
              help='Trading symbol', show_default=True)
@click.option('--timeframe', '-t', default='H1',
              help='Timeframe', show_default=True)
@click.option('--trials', '-n', default=50, type=int,
              help='Optimization trials', show_default=True)
@click.option('--objective', '-o', default='sharpe_ratio',
              help='Optimization metric', show_default=True)
@click.option('--output', '-out', default='optimization_results.json',
              help='Output file', show_default=True)
@pass_state
def optimize(state: CLIState, symbol: str, timeframe: str,
             trials: int, objective: str, output: str):
    """Optimize hyperparameters with progress tracking"""
    try:
        logger.info(f"Starting optimization with {trials} trials")
        
        state.pipeline = Pipeline(state.config_path)
        
        with click.progressbar(
            length=trials,
            label='Optimization progress'
        ) as bar:
            def update_progress(current: int, total: int):
                bar.pos = current
                bar.update(0)
                
            results = state.pipeline.execute_pipeline(
                'optimize',
                symbol=symbol,
                timeframe=timeframe,
                n_trials=trials,
                objective=objective,
                progress_callback=update_progress
            )
        
        # Save and display best parameters
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
            
        click.secho(f"\nOptimization complete! Results saved to {output}", fg='green')
        if 'best_params' in results:
            click.secho("\nBest Parameters:", fg='cyan')
            for param, value in results['best_params'].items():
                click.echo(f"  {param}: {value}")
                
    except Exception as e:
        logger.exception("Optimization failed")
        click.secho(f"Error: {str(e)}", fg='red')
        sys.exit(1)

@cli.command()
@click.argument('pipeline_name')
@click.option('--symbol', '-s', default='EURUSD',
              help='Trading symbol', show_default=True)
@click.option('--timeframe', '-t', default='H1',
              help='Timeframe', show_default=True)
@click.option('--output', '-o', help='Output file')
@pass_state
def run(state: CLIState, pipeline_name: str, symbol: str,
        timeframe: str, output: Optional[str]):
    """Execute any pipeline by name"""
    try:
        logger.info(f"Running pipeline: {pipeline_name}")
        
        state.pipeline = Pipeline(state.config_path)
        results = state.pipeline.execute_pipeline(
            pipeline_name,
            symbol=symbol,
            timeframe=timeframe
        )
        
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            click.secho(f"Results saved to {output}", fg='green')
            
        click.echo(json.dumps(results, indent=2))
        
    except Exception as e:
        logger.exception("Pipeline execution failed")
        click.secho(f"Error: {str(e)}", fg='red')
        sys.exit(1)

if __name__ == "__main__":
    cli()