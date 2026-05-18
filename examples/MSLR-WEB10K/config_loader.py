"""
config_loader.py — Universal Configuration Loader
==================================================
Reads experiment_config.yaml and provides convenient access to all shared
configuration values across Phase 1, 2, and 3 experiments.

Usage:
    from config_loader import load_config
    cfg = load_config()
    
    # Access via dot notation or dict:
    dataset_name = cfg.dataset.name
    label_indices = cfg['dataset']['label_indices']
    mrl_dims = cfg.model.mrl_nesting_dims
    
    # Access optimizer/loss_weighting/scheduler configs:
    optimizers = cfg.optimizers  # Dict of all optimizer configs
"""

import os
import yaml
from typing import Dict, Any


class ConfigDict:
    """Dot-notation accessor for nested dictionaries."""
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getattr__(self, key: str):
        """Allow config.dataset.name syntax."""
        if key.startswith('_'):
            return super().__getattribute__(key)
        
        value = self._data.get(key)
        if isinstance(value, dict):
            return ConfigDict(value)
        return value
    
    def __getitem__(self, key: str):
        """Allow config['dataset']['name'] syntax."""
        value = self._data[key]
        if isinstance(value, dict):
            return ConfigDict(value)
        return value
    
    def get(self, key: str, default=None):
        """Standard dict get method."""
        value = self._data.get(key, default)
        if isinstance(value, dict):
            return ConfigDict(value)
        return value
    
    def items(self):
        """Iterate over dict items."""
        return self._data.items()
    
    def keys(self):
        """Return dict keys."""
        return self._data.keys()
    
    def values(self):
        """Return dict values."""
        return self._data.values()
    
    def __repr__(self):
        return f"ConfigDict({self._data})"
    
    def __str__(self):
        return str(self._data)


def load_config(config_path: str = None) -> ConfigDict:
    """
    Load experiment_config.yaml from the examples/MSLR-WEB30K/ directory.
    
    Args:
        config_path (str, optional): Path to YAML config file. 
                                    If None, uses default location relative to this script.
    
    Returns:
        ConfigDict: Configuration object with dot-notation access.
    
    Raises:
        FileNotFoundError: If config file not found.
        yaml.YAMLError: If YAML parsing fails.
    """
    if config_path is None:
        # Default: expect experiment_config.yaml in same directory as this script
        config_path = os.path.join(
            os.path.dirname(__file__),
            "experiment_config.yaml"
        )
    
    config_path = os.path.abspath(config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Expected location: {os.path.join(os.path.dirname(__file__), 'experiment_config.yaml')}"
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML file {config_path}: {e}")
    
    if config_dict is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    return ConfigDict(config_dict)


def get_dataset_config() -> ConfigDict:
    """Convenience function: get only dataset config."""
    cfg = load_config()
    return cfg.dataset


def get_task_config() -> ConfigDict:
    """Convenience function: get only task config."""
    cfg = load_config()
    return cfg.tasks


def get_model_config() -> ConfigDict:
    """Convenience function: get only model config."""
    cfg = load_config()
    return cfg.model


def get_training_config() -> ConfigDict:
    """Convenience function: get only training config."""
    cfg = load_config()
    return cfg.training


def get_finetuning_config() -> ConfigDict:
    """Convenience function: get only fine-tuning config."""
    cfg = load_config()
    return cfg.finetuning


def get_output_config() -> ConfigDict:
    """Convenience function: get only output config."""
    cfg = load_config()
    return cfg.output


def get_optimizer_config(optimizer_name: str = None) -> Dict[str, Any]:
    """
    Get optimizer configuration.
    
    Args:
        optimizer_name (str, optional): Specific optimizer name. 
                                       If None, returns all optimizer configs.
    
    Returns:
        Dict: Optimizer configuration (or all if name is None).
    """
    cfg = load_config()
    optimizers = dict(cfg.optimizers.items())  # Convert to plain dict
    
    if optimizer_name is None:
        return optimizers
    
    if optimizer_name not in optimizers:
        available = list(optimizers.keys())
        raise ValueError(
            f"Optimizer '{optimizer_name}' not found in config.\n"
            f"Available: {available}"
        )
    
    return optimizers[optimizer_name]


def get_loss_weighting_config(strategy_name: str = None) -> Dict[str, Any]:
    """
    Get loss weighting strategy configuration.
    
    Args:
        strategy_name (str, optional): Specific strategy name. 
                                      If None, returns all strategy configs.
    
    Returns:
        Dict: Loss weighting configuration (or all if name is None).
    """
    cfg = load_config()
    strategies = dict(cfg.loss_weighting.items())  # Convert to plain dict
    
    if strategy_name is None:
        return strategies
    
    if strategy_name not in strategies:
        available = list(strategies.keys())
        raise ValueError(
            f"Loss weighting strategy '{strategy_name}' not found in config.\n"
            f"Available: {available}"
        )
    
    return strategies[strategy_name]


def get_scheduler_config(scheduler_name: str = None) -> Dict[str, Any]:
    """
    Get LR scheduler configuration.
    
    Args:
        scheduler_name (str, optional): Specific scheduler name. 
                                       If None, returns all scheduler configs.
    
    Returns:
        Dict: Scheduler configuration (or all if name is None).
    """
    cfg = load_config()
    schedulers = dict(cfg.schedulers.items())  # Convert to plain dict
    
    if scheduler_name is None:
        return schedulers
    
    if scheduler_name not in schedulers:
        available = list(schedulers.keys())
        raise ValueError(
            f"Scheduler '{scheduler_name}' not found in config.\n"
            f"Available: {available}"
        )
    
    return schedulers[scheduler_name]


if __name__ == "__main__":
    # Quick test
    print("Loading experiment config...")
    cfg = load_config()
    
    print(f"\n✓ Dataset: {cfg.dataset.name}")
    print(f"✓ Task indices: {cfg.tasks.indices}")
    print(f"✓ MRL dims: {cfg.model.mrl_nesting_dims}")
    print(f"✓ Fine-tuning epochs: {cfg.finetuning.epochs}")
    print(f"✓ Optimizers: {list(cfg.optimizers.keys())}")
    print(f"✓ Loss weighting strategies: {list(cfg.loss_weighting.keys())}")
    print(f"✓ Schedulers: {list(cfg.schedulers.keys())}")
    print("\n✓ Config loaded successfully!")
