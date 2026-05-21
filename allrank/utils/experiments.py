import json
import os
from argparse import Namespace
from typing import Dict, Any

import numpy as np
from attr import asdict
from flatten_dict import flatten

from allrank.config import Config
from allrank.utils.ltr_logging import get_logger

logger = get_logger()


def unpack_numpy_values(dict):
    return {k: v.item() for k, v in dict.items()}


def _convert_keys_to_str(d):
    """Recursively convert all integer keys to strings in nested dicts."""
    if isinstance(d, dict):
        return {str(k): _convert_keys_to_str(v) for k, v in d.items()}
    return d


def _make_json_serializable(obj):
    """Recursively convert numpy/torch types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        # Handle torch tensors and other objects with .item() method
        try:
            return obj.item()
        except (ValueError, TypeError):
            return str(obj)
    return obj


def dump_experiment_result(args: Namespace, config: Config, output_dir: str, result: Dict[str, Any]):
    final_config_dict = asdict(config)
    flattened_experiment = flatten(final_config_dict, reducer="path")

    # Handle per-task metrics (replaces simple unpack_numpy_values)
    def process_metrics(metrics_obj):
        if not isinstance(metrics_obj, dict):
            return metrics_obj
        # Check if it's nested (Task ID -> Metrics)
        if all(isinstance(v, dict) for v in metrics_obj.values()):
            return {str(task_id): _make_json_serializable(task_metrics) 
                    for task_id, task_metrics in metrics_obj.items()}
        else:
            return _make_json_serializable(metrics_obj)

    result["train_metrics"] = process_metrics(result["train_metrics"])
    result["val_metrics"] = process_metrics(result["val_metrics"])

    if hasattr(result["num_params"], "item"):
        result["num_params"] = result["num_params"].item()

    # Convert all integer keys to strings to avoid os.path.join() error
    result = _convert_keys_to_str(result)
    flattened_result = flatten(result, reducer="path")
    flattened_experiment.update(flattened_result)
    flattened_experiment["run_id"] = args.run_id
    flattened_experiment["dir"] = output_dir
    # Make JSON serializable by converting all numpy/torch types to native Python types
    flattened_experiment = _make_json_serializable(flattened_experiment)
    with open(os.path.join(output_dir, "experiment_result.json"), "w") as json_file:
        json.dump(flattened_experiment, json_file)
        json_file.write("\n")


def assert_expected_metrics(result: Dict[str, Any], expected_metrics: Dict[str, Dict[str, float]]):
    if expected_metrics:
        for role, metrics in expected_metrics.items():
            metrics_dict = result["{}_metrics".format(role)]
            # If nested, default to task '0'
            if all(isinstance(v, dict) for v in metrics_dict.values()):
                metrics_dict = metrics_dict.get('0', metrics_dict.get(0, {}))

            for name, expected_value in metrics.items():
                actual_value = metrics_dict.get(name, 0.0)
                msg = "{} {} got {}. It was expected to be at least {}".format(
                    role, name, actual_value, expected_value)
                if actual_value < expected_value:
                    logger.info(msg)
                assert actual_value >= expected_value, msg
