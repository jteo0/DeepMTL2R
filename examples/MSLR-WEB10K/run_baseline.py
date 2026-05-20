"""
run_baseline.py — Phase 1: Baseline Comparison
================================================
Trains a Single-Task model (Relevance only) and a basic Multi-Task model
(DeepMTL2R without Matryoshka or Gating) to establish baseline metrics
and compute Δm% (Average Relative Improvement).

Usage:
  python run_baseline.py
"""

import os
import sys
import time
import random
import json
from functools import partial
from argparse import Namespace
from pprint import pformat
import numpy as np
import torch
from torch import optim
from attr import asdict
from tqdm import tqdm
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("allrank").setLevel(logging.WARNING)
logging.getLogger("allrank.utils.ltr_logging").setLevel(logging.WARNING)

import allrank.models.losses as losses
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import CustomDataParallel, get_num_params
from allrank.training.train_utils import fit, compute_metrics, loss_batch
from allrank.utils.command_executor import execute_command
from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
from allrank.utils.file_utils import create_output_dirs, PathsContainer
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from allrank.data.dataset_loading import PADDED_Y_VALUE

# Configuration loader
from config_loader import load_config

# Load configuration from experiment_config.yaml
cfg = load_config()

# Extract configuration from YAML
DATASET_NAME = getattr(cfg.dataset, 'name', '50bps')
DATASET_BASE_PATH = getattr(cfg.dataset, 'base_path', None) or getattr(cfg.dataset, 'path', '../../datasets/MSLR-WEB10K')
FOLDS = getattr(cfg.dataset, 'folds', None) or [1]

# Ensure DATASET_BASE_PATH and FOLDS are not None
if DATASET_BASE_PATH is None:
    DATASET_BASE_PATH = '../../datasets/MSLR-WEB10K'
if FOLDS is None:
    FOLDS = [1]
    
REDUCTION_METHOD = getattr(cfg.dataset, 'reduction_method', 'mean')
LABEL_INDICES = getattr(cfg.dataset, 'label_indices', [131, 132, 133, 134, 135])
OUTPUT_DIR = getattr(cfg.output, 'base_dir', 'outputs')
CHECKPOINT_DIR = getattr(cfg.output, 'checkpoint_dir', 'checkpoints')
DEBUG = getattr(cfg.experiment, 'debug', False)

# Config files
CONFIG_GATING = os.path.join(os.path.dirname(__file__), "configs", "config_gating.json")


def format_metric_value(value):
    """Format metric value, handling NaN and inf."""
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return "—"
        return f"{value:.4f}"
    return f"{float(value):.4f}" if value is not None else "—"


def format_special_metrics(special_metrics_per_fold, fold_ids):
    """Format special metrics cleanly without numpy type objects."""
    formatted = {}
    for fold_idx, fold_id in enumerate(fold_ids):
        if fold_idx < len(special_metrics_per_fold):
            special = special_metrics_per_fold[fold_idx]
            fold_formatted = {}
            for metric_type, task_data in special.items():
                if isinstance(task_data, dict):
                    fold_formatted[metric_type] = {}
                    for task_idx, values in task_data.items():
                        if isinstance(values, dict):
                            fold_formatted[metric_type][task_idx] = {
                                k: format_metric_value(v) for k, v in values.items()
                            }
                        else:
                            fold_formatted[metric_type][task_idx] = format_metric_value(values)
                else:
                    fold_formatted[metric_type] = format_metric_value(task_data)
            formatted[f"fold_{fold_id}"] = fold_formatted
    return formatted


def group_metrics_by_task(summary_dict):
    """Group metrics by task for cleaner display."""
    tasks = defaultdict(dict)
    for metric_name, metric_stats in summary_dict.items():
        # Extract task ID from metric name (e.g., "Task_0_ndcg_30" -> task "0")
        parts = metric_name.split("_")
        if len(parts) >= 2 and parts[0] == "Task":
            task_id = parts[1]
            metric_key = "_".join(parts[2:])  # e.g., "ndcg_30"
            mean_val = metric_stats.get("mean", 0.0)
            std_val = metric_stats.get("std", 0.0)
            tasks[task_id][metric_key] = {
                "mean": mean_val,
                "std": std_val,
                "formatted": f"{format_metric_value(mean_val)} ± {format_metric_value(std_val)}",
            }
    return tasks


def evaluate_baseline(model, val_dataloader, config, device, task_indices, loss_func):
    model.eval()
    results = {}
    with torch.no_grad():
        for task_idx in task_indices:
            temp_dl = []
            for xb, yb, indices in tqdm(
                val_dataloader, desc=f"Eval Task {task_idx}", leave=False
            ):
                ki = torch.arange(xb.shape[-1])
                ki = ki[~torch.isin(ki, torch.tensor(LABEL_INDICES))]
                mxb = xb[:, :, ki]
                tyb = yb if task_idx == 0 else xb[:, :, task_idx]
                tyb[yb == -1] = -1
                temp_dl.append((mxb, tyb, indices))
            metrics = compute_metrics(config.metrics, model, temp_dl, device)
            results[task_idx] = metrics
    return results


def run_training(experiment_name, task_indices, moo_method, task_weights_tensor, dataset_path, train_dl, val_dl, nf):
    print(f"\nStarting run_training: {experiment_name}, task_indices={task_indices}", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    config = Config.from_json(CONFIG_GATING)
    if DEBUG:
        config.training.epochs = 2
    config.model.use_mrl = False
    config.data.path = dataset_path
    config.loss.args["reduction"] = REDUCTION_METHOD

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True

    run_id = os.path.join("baselines", experiment_name.replace(" ", "_").lower())
    args = Namespace(
        output_dir=OUTPUT_DIR,
        run_id=run_id,
        config_file_path=CONFIG_GATING,
        task_indices=",".join(map(str, task_indices)),
        task_weights="",
        moo_method=moo_method,
        dataset_name=DATASET_NAME,
        reduction_method=REDUCTION_METHOD,
    )
    paths = PathsContainer.from_args(
        args.output_dir, args.run_id, args.config_file_path
    )
    create_output_dirs(paths.output_dir)

    model = make_model(n_features=nf, **asdict(config.model, recurse=False))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
    model.to(device)

    optimizer = getattr(optim, config.optimizer.name)(
        params=model.parameters(), **config.optimizer.args
    )
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
    scheduler = (
        getattr(optim.lr_scheduler, config.lr_scheduler.name)(
            optimizer, **config.lr_scheduler.args
        )
        if config.lr_scheduler.name
        else None
    )

    results_filename = os.path.join(paths.output_dir, "results.txt")
    print(f"\n--- Training {experiment_name} ---")
    with (
        torch.autograd.detect_anomaly()
        if config.detect_anomaly
        else dummy_context_mgr()
    ):
        fit_result = fit(
            moo_method=moo_method,
            main_task_index=0,
            task_indices=task_indices,
            label_indices=LABEL_INDICES,
            results_filename=results_filename,
            model=model,
            loss_func=loss_func,
            task_weights=task_weights_tensor,
            epsilon=None,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            config=config,
            device=device,
            output_dir=paths.output_dir,
            tensorboard_output_path=paths.tensorboard_output_path,
            use_mrl=False,
            use_gating=False,
            **asdict(config.training),
        )

    # Evaluate final NDCG and other configured metrics
    results = evaluate_baseline(model, val_dl, config, device, task_indices, loss_func)

    # Collect num_params from fit result (if available)
    num_params = None
    try:
        num_params = fit_result.get("num_params") if isinstance(fit_result, dict) else None
    except Exception:
        num_params = None

    # =========================================================================
    # Save baseline model to central checkpoint directory
    # =========================================================================
    checkpoint_subdir = os.path.join(CHECKPOINT_DIR, "deepmtl2r")
    os.makedirs(checkpoint_subdir, exist_ok=True)

    # Baseline model is saved as deepmtl2r_foldX.pkl
    fold_str = os.path.basename(dataset_path).lower()
    source_model_path = os.path.join(paths.output_dir, "model.pkl")
    destination_checkpoint_path = os.path.join(checkpoint_subdir, f"deepmtl2r_{fold_str}.pkl")

    if os.path.exists(source_model_path):
        import shutil

        shutil.copy(source_model_path, destination_checkpoint_path)
        print(
            f"\n✓ Saved baseline model (deepmtl2r_{fold_str}.pkl) to: {destination_checkpoint_path}"
        )
    else:
        print(f"\n⚠ Warning: Baseline model not found at {source_model_path}")

    # Save per-run metrics to output dir for later aggregation
    metrics_output = {
        "experiment_name": experiment_name,
        "dataset_path": dataset_path,
        "fold": fold_str,
        "per_task_metrics": results,
        "special_metrics": fit_result.get("special_metrics", {}) if isinstance(fit_result, dict) else {},
        "num_params": int(num_params) if num_params is not None else None,
    }
    try:
        with open(os.path.join(paths.output_dir, "metrics.json"), "w", encoding="utf-8") as mf:
            json.dump(metrics_output, mf, indent=2, default=float)
    except Exception as e:
        print(f"Warning: failed to write per-run metrics file: {e}")

    return metrics_output


def main():
    print("=" * 60)
    print(" Phase 1: Baseline Comparison (Single-Task vs Multi-Task) - Cross Validation")
    print("=" * 60)

    # Use task indices from YAML config (0=Relevance, 132-135=Auxiliary)
    task_indices_mt = list(map(int, cfg.tasks.indices.split(",")))
    task_indices_st = [0]  # Single task is only Relevance (task 0)

    # Aggregators for standard metrics across folds
    all_ndcg30_st = []
    all_ndcg30_mt = []
    metrics_agg_st = defaultdict(list)
    metrics_agg_mt = defaultdict(list)
    params_st = []
    params_mt = []
    special_st = []
    special_mt = []

    for fold in FOLDS:
        print(f"\n" + "=" * 60)
        print(f" Processing Fold {fold}")
        print("=" * 60)
        
        dataset_path = os.path.join(DATASET_BASE_PATH, f"Fold{fold}")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        print(f"Dataset path exists: {dataset_path}", flush=True)
        
        config = Config.from_json(CONFIG_GATING)
        
        # Calculate max_rows for debug mode
        max_rows = None
        if DEBUG:
            debug_ratio = cfg.experiment.get('debug_ratio', 0.1)
            if debug_ratio > 0:
                estimated_total_rows = 30000000
                max_rows = max(1, int(estimated_total_rows * debug_ratio))
                print(f"[DEBUG] DEBUG MODE ENABLED - will limit to approximately {max_rows} rows ({debug_ratio*100:.4f}%)", flush=True)
            else:
                print(f"[DEBUG] DEBUG MODE: debug_ratio is {debug_ratio}, loading full dataset", flush=True)
        
        print(f"Loading LibSVM dataset from {dataset_path}...", flush=True)
        train_ds, val_ds = load_libsvm_dataset(
            dataset_path, config.data.slate_length, config.data.validation_ds_role, max_rows=max_rows
        )
        print(f"Dataset loaded! Train shape: {train_ds.shape}, Val shape: {val_ds.shape}", flush=True)

        print(f"Creating data loaders...", flush=True)
        nf = train_ds.shape[-1] - len(LABEL_INDICES)
        train_dl, val_dl = create_data_loaders(
            train_ds, val_ds, config.data.num_workers, config.data.batch_size
        )
        print(f"Data loaders created! nf={nf}", flush=True)

        # 1. Single Task
        res_st = run_training(f"Single-Task-Fold{fold}", task_indices_st, "ls", torch.tensor([1.0]), dataset_path, train_dl, val_dl, nf)
        per_task_st = res_st.get("per_task_metrics", {})
        # Collect Task 0 NDCG@30 for delta_m
        ndcg30_st = per_task_st.get(0, {}).get("ndcg_30", 0.0)
        all_ndcg30_st.append(ndcg30_st)
        
        # Aggregate all tasks (though ST is usually just task 0)
        for t_idx, t_metrics in per_task_st.items():
            for k, v in t_metrics.items():
                metrics_agg_st[f"Task_{t_idx}_{k}"].append(float(v))
        
        params_st.append(res_st.get("num_params"))
        special_st.append(res_st.get("special_metrics", {}))

        # 2. Multi Task
        num_tasks = len(task_indices_mt)
        res_mt = run_training(
            f"Multi-Task-Vanilla-Fold{fold}", task_indices_mt, "ls", [1.0 / num_tasks] * num_tasks, dataset_path, train_dl, val_dl, nf
        )
        per_task_mt = res_mt.get("per_task_metrics", {})
        # Collect Task 0 NDCG@30 for delta_m
        ndcg30_mt = per_task_mt.get(0, {}).get("ndcg_30", 0.0)
        all_ndcg30_mt.append(ndcg30_mt)
        
        # Aggregate all tasks (Main + Auxiliary)
        for t_idx, t_metrics in per_task_mt.items():
            for k, v in t_metrics.items():
                metrics_agg_mt[f"Task_{t_idx}_{k}"].append(float(v))
        
        params_mt.append(res_mt.get("num_params"))
        special_mt.append(res_mt.get("special_metrics", {}))

        # Free memory at the end of each fold
        del train_ds, val_ds, train_dl, val_dl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Compute Averages for primary metric
    avg_ndcg30_st = np.mean(all_ndcg30_st) if all_ndcg30_st else 0.0
    avg_ndcg30_mt = np.mean(all_ndcg30_mt) if all_ndcg30_mt else 0.0

    # Aggregate stats for all discovered metrics
    def summarize_metrics(agg):
        summary = {}
        for k, vals in agg.items():
            arr = np.array(vals, dtype=float)
            summary[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "per_fold": [float(x) for x in arr.tolist()],
            }
        return summary

    summary_st = summarize_metrics(metrics_agg_st)
    summary_mt = summarize_metrics(metrics_agg_mt)
    avg_params_st = float(np.mean([p for p in params_st if p is not None])) if any(p is not None for p in params_st) else 0.0
    avg_params_mt = float(np.mean([p for p in params_mt if p is not None])) if any(p is not None for p in params_mt) else 0.0

    # Compute Delta m%
    if avg_ndcg30_st > 0:
        delta_m = ((avg_ndcg30_mt - avg_ndcg30_st) / avg_ndcg30_st) * 100
    else:
        delta_m = 0.0

    # Group metrics by task
    st_tasks = group_metrics_by_task(summary_st)
    mt_tasks = group_metrics_by_task(summary_mt)
    st_special = format_special_metrics(special_st, FOLDS)
    mt_special = format_special_metrics(special_mt, FOLDS)

    print("\n" + "=" * 80)
    print("BASELINE RESULTS (CROSS-VALIDATION AVERAGE)")
    print("=" * 80)
    print(f"Folds Evaluated:                {FOLDS}")
    print(f"Single-Task NDCG@30 (Relevance): {avg_ndcg30_st:.4f}")
    print(f"Multi-Task  NDCG@30 (Relevance): {avg_ndcg30_mt:.4f}")
    print(f"Δm% (Relative Improvement):     {delta_m:+.2f}%")
    print(f"Single-Task Total Params:       {avg_params_st:.0f}")
    print(f"Multi-Task  Total Params:       {avg_params_mt:.0f}")

    # Display Single-Task metrics grouped by task
    print("\n--- SINGLE-TASK METRICS ---")
    for task_id in sorted(st_tasks.keys()):
        task_label = " (Main)" if str(task_id) == "0" else " (Aux)"
        print(f"  Task {task_id}{task_label}:")
        for metric_key in sorted(st_tasks[task_id].keys()):
            formatted_val = st_tasks[task_id][metric_key]["formatted"]
            print(f"    {metric_key}: {formatted_val}")

    # Display Multi-Task metrics grouped by task
    print("\n--- MULTI-TASK METRICS ---")
    global_metrics = defaultdict(list)
    for task_id in sorted(mt_tasks.keys()):
        task_label = " (Main)" if str(task_id) == "0" else " (Aux)"
        print(f"  Task {task_id}{task_label}:")
        for metric_key in sorted(mt_tasks[task_id].keys()):
            formatted_val = mt_tasks[task_id][metric_key]["formatted"]
            print(f"    {metric_key}: {formatted_val}")
            
            mean_val = mt_tasks[task_id][metric_key]["mean"]
            if not np.isnan(mean_val):
                global_metrics[metric_key].append(mean_val)
                
    print(f"\n  --- Global Averages (All Tasks) ---")
    for metric_key in sorted(global_metrics.keys()):
        g_avg = float(np.mean(global_metrics[metric_key]))
        print(f"    {metric_key.upper()}: {g_avg:.4f}")

    # Display special metrics if available
    if any(special_st):
        agg_robustness_st = {}
        for s_metrics in special_st:
            if "noise_robustness" in s_metrics:
                for t_idx, rob in s_metrics["noise_robustness"].items():
                    for r_key, r_val in rob.items():
                        if r_key not in agg_robustness_st:
                            agg_robustness_st[r_key] = []
                        agg_robustness_st[r_key].append(float(r_val))
        if len(agg_robustness_st) > 0:
            print(f"\n--- SINGLE-TASK SPECIAL METRICS ---")
            print(f"  Robustness to Noisy Features (Global Average Drop):")
            for r_key in sorted(agg_robustness_st.keys()):
                if "ndcg" in r_key or "map" in r_key:
                    r_arr = np.array(agg_robustness_st[r_key])
                    r_mean = float(np.mean(r_arr))
                    r_std = float(np.std(r_arr, ddof=1)) if len(r_arr) > 1 else 0.0
                    print(f"    {r_key}: {r_mean:.4f} ± {r_std:.4f}")

    if any(special_mt):
        agg_robustness_mt = {}
        for s_metrics in special_mt:
            if "noise_robustness" in s_metrics:
                for t_idx, rob in s_metrics["noise_robustness"].items():
                    for r_key, r_val in rob.items():
                        if r_key not in agg_robustness_mt:
                            agg_robustness_mt[r_key] = []
                        agg_robustness_mt[r_key].append(float(r_val))
        if len(agg_robustness_mt) > 0:
            print(f"\n  Robustness to Noisy Features (Global Average Drop):")
            for r_key in sorted(agg_robustness_mt.keys()):
                if "ndcg" in r_key or "map" in r_key:
                    r_arr = np.array(agg_robustness_mt[r_key])
                    r_mean = float(np.mean(r_arr))
                    r_std = float(np.std(r_arr, ddof=1)) if len(r_arr) > 1 else 0.0
                    print(f"    {r_key}: {r_mean:.4f} ± {r_std:.4f}")

    print("\n" + "=" * 80)

    # Save summary (expanded)
    os.makedirs(os.path.join(OUTPUT_DIR, "baselines"), exist_ok=True)
    summary_out = {
        "folds": FOLDS,
        "delta_m_percent": float(delta_m),
        "single_task": {
            "ndcg30_avg": float(avg_ndcg30_st),
            "ndcg30_folds": [float(x) for x in all_ndcg30_st],
            "params_per_fold": params_st,
            "params_avg": avg_params_st,
            "metrics": summary_st,
            "special_metrics_per_fold": special_st,
        },
        "multi_task": {
            "ndcg30_avg": float(avg_ndcg30_mt),
            "ndcg30_folds": [float(x) for x in all_ndcg30_mt],
            "params_per_fold": params_mt,
            "params_avg": avg_params_mt,
            "metrics": summary_mt,
            "special_metrics_per_fold": special_mt,
        },
    }

    with open(os.path.join(OUTPUT_DIR, "baselines", "baseline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
