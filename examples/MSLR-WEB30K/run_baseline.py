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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
logging.getLogger().setLevel(logging.INFO) # Tambahkan ini untuk memunculkan log INFO ke console

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
DATASET_NAME = cfg.dataset.name
DATASET_PATH = cfg.dataset.path
REDUCTION_METHOD = cfg.dataset.reduction_method
LABEL_INDICES = cfg.dataset.label_indices
OUTPUT_DIR = cfg.output.base_dir
CHECKPOINT_DIR = cfg.output.checkpoint_dir
DEBUG = cfg.experiment.debug

# Config files
CONFIG_GATING = os.path.join(os.path.dirname(__file__), "configs", "config_gating.json")


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


def run_training(experiment_name, task_indices, moo_method, task_weights_tensor):
    print(f"\n[DEBUG] Starting run_training: {experiment_name}, task_indices={task_indices}", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    config = Config.from_json(CONFIG_GATING)
    if DEBUG:
        config.training.epochs = 2
    config.model.use_mrl = False
    config.data.path = DATASET_PATH
    config.loss.args["reduction"] = REDUCTION_METHOD

    if not os.path.exists(config.data.path):
        raise FileNotFoundError(f"Dataset path not found: {config.data.path}")

    print(f"[DEBUG] Dataset path exists: {config.data.path}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True

    # Calculate max_rows for debug mode
    max_rows = None
    if DEBUG:
        debug_ratio = cfg.experiment.get('debug_ratio', 0.1)  # Default 10% if not specified
        if debug_ratio > 0:
            # Estimate: assume ~100 documents per query for MSLR-WEB30K
            estimated_total_rows = 30000000  # MSLR-WEB30K typical size
            max_rows = max(1, int(estimated_total_rows * debug_ratio))
            print(f"[DEBUG] DEBUG MODE ENABLED - will limit to approximately {max_rows} rows ({debug_ratio*100:.4f}%)", flush=True)
        else:
            print(f"[DEBUG] DEBUG MODE: debug_ratio is {debug_ratio}, loading full dataset", flush=True)
    
    print(f"[DEBUG] Loading LibSVM dataset from {config.data.path}...", flush=True)
    train_ds, val_ds = load_libsvm_dataset(
        config.data.path, config.data.slate_length, config.data.validation_ds_role, max_rows=max_rows
    )
    print(f"[DEBUG] Dataset loaded! Train shape: {train_ds.shape}, Val shape: {val_ds.shape}", flush=True)

    print(f"[DEBUG] Creating data loaders...", flush=True)
    nf = train_ds.shape[-1] - len(LABEL_INDICES)
    train_dl, val_dl = create_data_loaders(
        train_ds, val_ds, config.data.num_workers, config.data.batch_size
    )
    print(f"[DEBUG] Data loaders created! nf={nf}", flush=True)

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
        fit(
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

    # Evaluate final NDCG
    results = evaluate_baseline(model, val_dl, config, device, task_indices, loss_func)

    # =========================================================================
    # Save baseline model to central checkpoint directory
    # =========================================================================
    checkpoint_subdir = os.path.join(CHECKPOINT_DIR, "deepmtl2r")
    os.makedirs(checkpoint_subdir, exist_ok=True)

    # Baseline model is saved as deepmtl2r.pkl
    source_model_path = os.path.join(paths.output_dir, "model.pkl")
    destination_checkpoint_path = os.path.join(checkpoint_subdir, "deepmtl2r.pkl")

    if os.path.exists(source_model_path):
        import shutil

        shutil.copy(source_model_path, destination_checkpoint_path)
        print(
            f"\n✓ Saved baseline model (deepmtl2r.pkl) to: {destination_checkpoint_path}"
        )
    else:
        print(f"\n⚠ Warning: Baseline model not found at {source_model_path}")

    return results


def main():
    print("=" * 60)
    print(" Phase 1: Baseline Comparison (Single-Task vs Multi-Task)")
    print("=" * 60)

    print("[DEBUG] About to read user input...", flush=True)
    task_input = input(
        "\nEnter task indices for Multi-Task (comma separated, default: 0,131): "
    ).strip()
    print(f"[DEBUG] User input received: '{task_input}'", flush=True)
    task_indices_mt = list(map(int, task_input.split(","))) if task_input else [0, 131]
    task_indices_st = [0]  # Single task is only Relevance (task 0)

    # 1. Single Task
    res_st = run_training("Single-Task", task_indices_st, "ls", torch.tensor([1.0]))
    ndcg30_st = res_st[0].get("ndcg_30", 0.0)

    # 2. Multi Task
    num_tasks = len(task_indices_mt)
    res_mt = run_training(
        "Multi-Task-Vanilla", task_indices_mt, "ls", [1.0 / num_tasks] * num_tasks
    )
    ndcg30_mt = res_mt[0].get("ndcg_30", 0.0)

    # Compute Delta m%
    if ndcg30_st > 0:
        delta_m = ((ndcg30_mt - ndcg30_st) / ndcg30_st) * 100
    else:
        delta_m = 0.0

    print("\n" + "=" * 60)
    print(" BASELINE RESULTS")
    print("=" * 60)
    print(f" Single-Task NDCG@30 (Relevance): {ndcg30_st:.4f}")
    print(f" Multi-Task  NDCG@30 (Relevance): {ndcg30_mt:.4f}")
    print(f" Δm% (Relative Improvement):      {delta_m:+.2f}%")
    print("=" * 60)

    # Save summary
    os.makedirs(os.path.join(OUTPUT_DIR, "baselines"), exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "baselines", "baseline_summary.json"), "w") as f:
        json.dump(
            {
                "single_task_ndcg30": float(ndcg30_st),
                "multi_task_ndcg30": float(ndcg30_mt),
                "delta_m_percent": float(delta_m),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
