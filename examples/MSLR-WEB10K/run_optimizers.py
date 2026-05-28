"""
run_optimizers.py — Final Experiment: Optimizer Comparison (Priority 1)
========================================================================
Late-stage fine-tuning experiment that loads checkpoints from the two
primary experiments (Matryoshka & Dynamic Feature Gating) and compares
the effect of different optimizers on breaking through performance plateaus.

Tested Optimizers:
    - AdamW  (baseline/default)
    - SGD + Momentum

Metrics recorded (per metrics.md Phase 3 — Priority 1):
  - NDCG@10, NDCG@20  (per task, per epoch)
  - Peak NDCG Delta (ΔNDCG@10): improvement over checkpoint baseline
  - Epochs to New Plateau (convergence speed)

Usage:
  python run_optimizers.py

The script will interactively ask:
  1. Which primary experiment checkpoint to load (Matryoshka / Gating)
  2. The path to the checkpoint .pkl file

Then it runs all 3 optimizer variants sequentially on that checkpoint.

Note: Ensure the working directory is DeepMTL2R/examples/MSLR-WEB30K/
"""

# Standard library
import os
import sys
import time
import random
import json
import copy
from functools import partial
from argparse import Namespace
from tqdm import tqdm

# Third-party
import numpy as np
import torch
from torch import optim
from attr import asdict

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Local allrank imports
import allrank.models.losses as losses
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import (
    get_torch_device,
    CustomDataParallel,
    get_num_params,
)
from allrank.training.train_utils import (
    fit,
    compute_metrics,
    loss_batch,
    log_gating_sparsity,
    get_current_lr,
    log_metrics,
)
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from allrank.models.model_utils import get_model_parameters
from allrank.methods.weight_methods import WeightMethods
from allrank.data.dataset_loading import PADDED_Y_VALUE

# Configuration loader
from config_loader import load_config, get_optimizer_config

# =============================================================================
# Load configuration from experiment_config.yaml
# =============================================================================
cfg = load_config()

# Extract configuration from YAML
FINETUNE_EPOCHS = cfg.finetuning.epochs
FINETUNE_PATIENCE = cfg.finetuning.patience
MOO_METHOD = cfg.model.moo_method
DATASET_NAME = cfg.dataset.name
try:
    DATASET_BASE_PATH = cfg.dataset.base_path
    FOLDS = cfg.dataset.folds
except AttributeError:
    # Fallback to old path variable if base_path/folds not available
    DATASET_BASE_PATH = (
        cfg.dataset.path.rsplit("/", 1)[0]
        if "Fold" in cfg.dataset.path
        else cfg.dataset.path
    )
    FOLDS = [1]
REDUCTION_METHOD = cfg.dataset.reduction_method
TASK_INDICES = cfg.tasks.indices
TASK_WEIGHTS = cfg.tasks.weights
LABEL_INDICES = cfg.dataset.label_indices
OUTPUT_DIR = cfg.output.base_dir
CHECKPOINT_DIR = cfg.output.checkpoint_dir
DEBUG = cfg.experiment.debug
OUTPUT_DIR = cfg.output.base_dir
MRL_NESTING_DIMS = cfg.model.mrl_nesting_dims
PLATEAU_THRESHOLD = cfg.finetuning.plateau_threshold
PLATEAU_WINDOW = cfg.finetuning.plateau_window
OPTIMIZER_CONFIGS = get_optimizer_config()  # Load all optimizer configs from YAML
# Remove Lion optimizer if present in YAML configs
if "Lion" in OPTIMIZER_CONFIGS:
    del OPTIMIZER_CONFIGS["Lion"]

# Adjust FINETUNE_EPOCHS for DEBUG mode
if DEBUG:
    FINETUNE_EPOCHS = 2

# Config files
CONFIG_MATRYOSHKA = os.path.join(
    os.path.dirname(__file__), "configs", "config_matryoshka.json"
)
CONFIG_GATING = os.path.join(os.path.dirname(__file__), "configs", "config_gating.json")


def create_optimizer(model, opt_config):
    """Create optimizer from config dict."""
    if opt_config.get("is_pytorch", True):
        opt_class = getattr(optim, opt_config["class"])
        return opt_class(params=model.parameters(), **opt_config.get("args", {}))
    # Non-pytorch optimizers are not supported in this script (Lion removed)
    return None


def detect_plateau_epoch(
    val_history, threshold=PLATEAU_THRESHOLD, window=PLATEAU_WINDOW
):
    """
    Detect the epoch at which a new plateau is reached.
    Returns the epoch index (0-based) where the metric stopped improving
    for `window` consecutive epochs.
    """
    if len(val_history) < window + 1:
        return None

    best_so_far = val_history[0]
    stagnation_count = 0

    for i in range(1, len(val_history)):
        if val_history[i] > best_so_far + threshold:
            best_so_far = val_history[i]
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= window:
            return i - window + 1  # Epoch where stagnation started

    return None


def evaluate_model(
    model, val_dataloader, config, device, task_indices, loss_func, use_mrl
):
    """Evaluate model and return per-task metrics and losses."""
    model.eval()
    results = {}
    all_losses = {}

    with torch.no_grad():
        for task_idx in task_indices:
            temp_dl = []
            task_losses = []
            for xb, yb, indices in tqdm(
                val_dataloader, desc=f"Evaluating Task {task_idx}", leave=False
            ):
                all_indices_t = torch.arange(xb.shape[-1])
                keep_indices = all_indices_t[
                    ~torch.isin(all_indices_t, torch.tensor(LABEL_INDICES))
                ]
                modified_xb = xb[:, :, keep_indices]
                task_yb = yb if task_idx == 0 else xb[:, :, task_idx]
                task_yb[yb == -1] = -1
                temp_dl.append((modified_xb, task_yb, indices))

                # Compute validation loss
                loss = loss_batch(
                    model,
                    loss_func,
                    modified_xb.to(device),
                    task_yb.to(device),
                    indices.to(device),
                    use_mrl=use_mrl,
                )
                task_losses.append(loss.item())

            metrics = compute_metrics(config.metrics, model, temp_dl, device)
            results[task_idx] = metrics
            all_losses[task_idx] = np.mean(task_losses)

    return results, all_losses


def finetune_with_optimizer(
    opt_name,
    opt_config,
    model_state_dict,
    model_kwargs,
    config,
    train_dataloader,
    val_dataloader,
    task_indices,
    device,
    use_mrl,
    use_gating,
    baseline_ndcg10,
    results_dir,
    experiment_tag,
    logger,
):
    """
    Perform late-stage fine-tuning with a specific optimizer.
    Returns a dict with all tracked metrics.
    """
    print(f"\n{'━'*60}")
    print(f"  Optimizer: {opt_name}")
    print(f"  Config: {opt_config['args']}")
    print(f"{'━'*60}")

    # Rebuild model and load checkpoint weights
    model = make_model(**model_kwargs)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
    model.to(device)

    # Create optimizer
    optimizer = create_optimizer(model, opt_config)
    if optimizer is None:
        print(f"  Skipping {opt_name} (optimizer not available)")
        return None

    # Loss function
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)

    # Weight method for MOO
    task_indices_parsed = list(map(int, TASK_INDICES.split(",")))
    task_weights_raw = list(map(int, TASK_WEIGHTS.split(",")))
    min_weight, max_weight = task_weights_raw

    # For fine-tuning we use a single weight combination (mid-range)
    weight_index = (min_weight + max_weight) >> 1
    import itertools

    # Use same weight logic as primary experiment
    num_tasks = len(task_indices_parsed)
    if MOO_METHOD == "ec":
        # Not typically used in fine-tuning, but keep consistent
        task_weights_tensor = torch.tensor([1] + [0])
    else:
        if num_tasks == 2:
            weight_combinations = [[x, 1 - x] for x in np.linspace(0.001, 0.991, 10)]
        else:
            weight_combinations = np.loadtxt("weights-5tasks.txt").tolist()
        task_weights_tensor = weight_combinations[weight_index]

    weight_method = WeightMethods(
        MOO_METHOD,
        n_tasks=num_tasks,
        device=device,
        task_weights=task_weights_tensor,
        epsilon=None,
    )

    # Scheduler: use ReduceLROnPlateau for fine-tuning to detect stagnation
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Results file
    results_filename = os.path.join(
        results_dir, f"optimizer_{opt_name}_{experiment_tag}.txt"
    )
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    if os.path.exists(results_filename):
        os.remove(results_filename)

    with open(results_filename, "a", encoding="utf-8") as f:
        f.write(f"=== Optimizer Comparison: {opt_name} on {experiment_tag} ===\n")
        f.write(f"Optimizer: {opt_name}\n")
        f.write(f"Optimizer Args: {opt_config['args']}\n")
        f.write(f"Baseline NDCG@10: {baseline_ndcg10:.6f}\n")
        f.write(f"Fine-tune Epochs: {FINETUNE_EPOCHS}\n")
        f.write(f"Task Indices: {task_indices}\n\n")

    # Training loop
    val_ndcg10_history = []
    best_ndcg10 = baseline_ndcg10
    best_epoch = -1

    for epoch in tqdm(
        range(FINETUNE_EPOCHS), desc=f"Fine-tuning [{opt_name}]", unit="epoch"
    ):
        model.train()
        train_loss_values = {ti: [] for ti in task_indices}
        train_nums = []

        for batch_id, batch in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch}/{FINETUNE_EPOCHS} [{opt_name}]")
        ):
            xb, yb, indices = batch
            all_idx = torch.arange(xb.shape[-1])
            keep_idx = all_idx[~torch.isin(all_idx, torch.tensor(LABEL_INDICES))]
            modified_xb = xb[:, :, keep_idx]

            batch_losses = []
            for task_idx in task_indices:
                task_yb = yb if task_idx == 0 else xb[:, :, task_idx]
                task_yb[yb == -1] = -1
                loss = loss_batch(
                    model,
                    loss_func,
                    modified_xb.to(device),
                    task_yb.to(device),
                    indices.to(device),
                    use_mrl=use_mrl,
                )
                batch_losses.append(loss)
                train_loss_values[task_idx].append(loss.item())
            train_nums.append(len(xb))

            optimizer.zero_grad()
            loss_weighted_sum, _ = weight_method.backward(
                losses=torch.stack(batch_losses),
                shared_parameters=get_model_parameters(model, "shared_parameters"),
                task_specific_parameters=get_model_parameters(
                    model, "task_specific_parameters"
                ),
                last_shared_parameters=get_model_parameters(
                    model, "last_shared_parameters"
                ),
                task_weights=task_weights_tensor,
            )
            optimizer.step()

        # Gating sparsity logging
        if use_gating:
            log_gating_sparsity(epoch, model, results_filename)

        # Validation
        val_results, val_losses = evaluate_model(
            model, val_dataloader, config, device, task_indices, loss_func, use_mrl
        )

        # Extract NDCG@10 for main task (task 0)
        current_ndcg10 = val_results[task_indices[0]].get("ndcg_10", 0.0)
        val_ndcg10_history.append(current_ndcg10)

        if current_ndcg10 > best_ndcg10:
            best_ndcg10 = current_ndcg10
            best_epoch = epoch

        # Log metrics
        lr = get_current_lr(optimizer)
        with open(results_filename, "a", encoding="utf-8") as f:
            f.write(f"epoch:{epoch}\tlr:{lr:.2e}\t")
            for task_idx in task_indices:
                f.write(f"task:{task_idx}\tVal Metrics:{val_results[task_idx]}\t")
                f.write(f"Val Loss:{val_losses[task_idx]:.6f}\t")
            f.write("\n")

        logger.info(
            f"[{opt_name}] Epoch {epoch}/{FINETUNE_EPOCHS-1} | "
            f"NDCG@10={current_ndcg10:.6f} | Best={best_ndcg10:.6f} | LR={lr:.2e}"
        )

        # Scheduler step
        scheduler.step(current_ndcg10)

    # Compute final metrics
    peak_delta_ndcg = best_ndcg10 - baseline_ndcg10
    plateau_epoch = detect_plateau_epoch(val_ndcg10_history)

    summary = {
        "optimizer": opt_name,
        "experiment": experiment_tag,
        "baseline_ndcg10": baseline_ndcg10,
        "peak_ndcg10": best_ndcg10,
        "peak_delta_ndcg10": peak_delta_ndcg,
        "best_epoch": best_epoch,
        "plateau_epoch": plateau_epoch,
        "epochs_to_plateau": (
            plateau_epoch if plateau_epoch is not None else FINETUNE_EPOCHS
        ),
        "final_ndcg10": val_ndcg10_history[-1] if val_ndcg10_history else 0.0,
        "ndcg10_history": val_ndcg10_history,
        "all_task_final_metrics": {str(ti): val_results[ti] for ti in task_indices},
    }

    # Write summary
    with open(results_filename, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"SUMMARY: {opt_name} on {experiment_tag}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Baseline NDCG@10:       {baseline_ndcg10:.6f}\n")
        f.write(f"Peak NDCG@10:           {best_ndcg10:.6f}\n")
        f.write(f"Peak ΔNDCG@10:          {peak_delta_ndcg:+.6f}\n")
        f.write(f"Best Epoch:             {best_epoch}\n")
        f.write(f"Epochs to Plateau:      {summary['epochs_to_plateau']}\n")
        f.write(f"Final NDCG@10:          {summary['final_ndcg10']:.6f}\n")
        f.write(f"NDCG@10 History:        {val_ndcg10_history}\n")

    # Save summary JSON
    json_path = os.path.join(
        results_dir, f"optimizer_{opt_name}_{experiment_tag}_summary.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Peak Delta NDCG@10 (Δ): {peak_delta_ndcg:+.6f}")
    print(f"  Epochs to Plateau: {summary['epochs_to_plateau']}")
    print(f"  Results saved to: {results_filename}")

    return summary


def run_optimizer_experiment(
    experiment_choice,
    checkpoint_path,
    dataset_path,
    train_dataloader,
    val_dataloader,
    n_features,
):
    """Run the full optimizer comparison for one primary experiment checkpoint."""
    # Set reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Determine experiment type
    if experiment_choice == "1":
        experiment_tag = "matryoshka"
        config_path = CONFIG_MATRYOSHKA
        use_mrl = True
        use_gating = False
        mrl_nesting_dims = MRL_NESTING_DIMS
    elif experiment_choice == "2":
        experiment_tag = "feature_gating"
        config_path = CONFIG_GATING
        use_mrl = False
        use_gating = True
        mrl_nesting_dims = None
    else:
        experiment_tag = "baseline"
        config_path = CONFIG_GATING
        use_mrl = False
        use_gating = False
        mrl_nesting_dims = None

    # Load config
    config = Config.from_json(config_path)
    if use_mrl:
        config.model.use_mrl = True
        config.model.mrl_nesting_dims = mrl_nesting_dims

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    # Dataset (from YAML config)
    config.data.path = dataset_path
    config.loss.args["reduction"] = REDUCTION_METHOD

    # Parse tasks
    task_indices = list(map(int, TASK_INDICES.split(",")))

    print(f"\n{'═'*60}")
    print(f"  FINAL EXPERIMENT — Priority 1: Optimizer Comparison")
    print(f"  Primary Experiment: {experiment_tag.upper()}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Fine-tune Epochs: {FINETUNE_EPOCHS}")
    print(f"  Optimizers: {list(OPTIMIZER_CONFIGS.keys())}")
    print(f"{'═'*60}\n")

    # Build model architecture (to get the right structure)
    model_kwargs = dict(
        n_features=n_features,
        **asdict(config.model, recurse=False),
    )

    # Load checkpoint state dict
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_state = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint_state, dict) and "model_state_dict" in checkpoint_state:
        model_state_dict = checkpoint_state["model_state_dict"]
    else:
        model_state_dict = checkpoint_state

    # Evaluate baseline checkpoint to get initial NDCG@10
    print("Evaluating baseline checkpoint...")
    baseline_model = make_model(**model_kwargs)
    baseline_model.load_state_dict(model_state_dict)
    baseline_model.to(device)

    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
    baseline_results, _ = evaluate_model(
        baseline_model, val_dataloader, config, device, task_indices, loss_func, use_mrl
    )
    baseline_ndcg10 = baseline_results[task_indices[0]].get("ndcg_10", 0.0)
    print(f"Baseline NDCG@10 (from checkpoint): {baseline_ndcg10:.6f}")

    # Log baseline metrics for all tasks
    fold_str = os.path.basename(dataset_path).lower()
    results_dir = os.path.join(
        "result", "final_experiment", "optimizer_comparison", experiment_tag, fold_str
    )
    os.makedirs(results_dir, exist_ok=True)

    baseline_file = os.path.join(results_dir, f"baseline_{experiment_tag}.txt")
    with open(baseline_file, "w", encoding="utf-8") as f:
        f.write(f"=== Baseline Metrics ({experiment_tag}) ===\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        for ti in task_indices:
            f.write(f"Task {ti}: {baseline_results[ti]}\n")

    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Logger
    logger = init_logger(results_dir)

    # Run each optimizer
    all_summaries = {}
    for opt_name, opt_config in tqdm(
        OPTIMIZER_CONFIGS.items(), desc="Optimizer Variants", unit="opt"
    ):
        summary = finetune_with_optimizer(
            opt_name=opt_name,
            opt_config=opt_config,
            model_state_dict=copy.deepcopy(model_state_dict),
            model_kwargs=model_kwargs,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            task_indices=task_indices,
            device=device,
            use_mrl=use_mrl,
            use_gating=use_gating,
            baseline_ndcg10=baseline_ndcg10,
            results_dir=results_dir,
            experiment_tag=experiment_tag,
            logger=logger,
        )
        if summary is not None:
            all_summaries[opt_name] = summary

        # Clear GPU memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print comparison table
    print(f"\n{'═'*70}")
    print(f"  OPTIMIZER COMPARISON RESULTS — {experiment_tag.upper()}")
    print(f"{'═'*70}")
    print(
        f"  {'Optimizer':<18} {'DeltaNDCG@10':>12} {'Peak NDCG@10':>14} {'Plateau Ep':>12}"
    )
    print(f"  {'─'*18} {'─'*12} {'─'*14} {'─'*12}")
    for name, s in all_summaries.items():
        print(
            f"  {name:<18} {s['peak_delta_ndcg10']:>+12.6f} "
            f"{s['peak_ndcg10']:>14.6f} {s['epochs_to_plateau']:>12}"
        )
    print(f"{'═'*70}\n")

    # Save combined comparison JSON
    comparison_path = os.path.join(results_dir, f"comparison_{experiment_tag}.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"Combined results saved to: {comparison_path}")


def main():
    print("\n" + "═" * 60)
    print("  DeepMTL2R Final Experiment — Priority 1")
    print("  Optimizer Comparison (AdamW vs SGD+Momentum)")
    print("═" * 60)

    print("\nSelect the PRIMARY experiment checkpoint to fine-tune:")
    print("  1 -> Matryoshka Feature Projection")
    print("  2 -> Dynamic Feature Gating")
    print("  3 -> Baseline (Vanilla Multi-Task)")
    print()

    while True:
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        if choice in ("1", "2", "3"):
            break
        print("  Invalid input. Please enter 1, 2, or 3.")

    # Try to load manifest
    task_id_string = "_task_".join(TASK_INDICES.split(","))
    if choice == "1":
        experiment_tag = "matryoshka"
    elif choice == "2":
        experiment_tag = "feature_gating"
    else:
        experiment_tag = "baseline"

    for fold in FOLDS:
        print(f"\n" + "=" * 60)
        print(f" Processing Fold {fold}")
        print("=" * 60)
        fold_str = f"fold{fold}"
        dataset_path = os.path.join(DATASET_BASE_PATH, f"Fold{fold}")

        if choice == "1":
            ckpt_name = f"matryoshka_{fold_str}_weight0.pkl"
            ckpt_dir = os.path.join(CHECKPOINT_DIR, "matryoshka")
        elif choice == "2":
            ckpt_name = f"feature_gating_{fold_str}_weight0.pkl"
            ckpt_dir = os.path.join(CHECKPOINT_DIR, "feature_gating")
        else:
            ckpt_name = f"deepmtl2r_{fold_str}.pkl"
            ckpt_dir = os.path.join(CHECKPOINT_DIR, "deepmtl2r")

        checkpoint_path = os.path.join(ckpt_dir, ckpt_name)

        if not os.path.exists(checkpoint_path):
            print(f"\n[ERROR] Checkpoint not found: {checkpoint_path}")
            print(
                "Please run run_extension.py or run_baseline.py first to generate this checkpoint."
            )
            continue

        # Load dataset once per fold
        config_tmp = Config.from_json(CONFIG_GATING)
        max_rows = None
        if DEBUG:
            debug_ratio = cfg.experiment.get("debug_ratio", 0.1)
            if debug_ratio > 0:
                estimated_total_rows = 30000000
                max_rows = max(1, int(estimated_total_rows * debug_ratio))
                print(
                    f"[DEBUG] DEBUG MODE ENABLED - will limit to approximately {max_rows} rows ({debug_ratio*100:.4f}%)",
                    flush=True,
                )
            else:
                print(
                    f"[DEBUG] DEBUG MODE: debug_ratio is {debug_ratio}, loading full dataset",
                    flush=True,
                )

        print(f"Loading MSLR-WEB30K dataset from {dataset_path}...")
        train_ds, val_ds = load_libsvm_dataset(
            input_path=dataset_path,
            slate_length=config_tmp.data.slate_length,
            validation_ds_role=config_tmp.data.validation_ds_role,
            max_rows=max_rows,
        )

        nf = train_ds.shape[-1] - len(LABEL_INDICES)
        assert nf == val_ds.shape[-1] - len(LABEL_INDICES)

        train_dl, val_dl = create_data_loaders(
            train_ds,
            val_ds,
            num_workers=config_tmp.data.num_workers,
            batch_size=config_tmp.data.batch_size,
        )

        run_optimizer_experiment(
            choice, checkpoint_path, dataset_path, train_dl, val_dl, nf
        )

        # Free memory at the end of each fold
        del train_ds, val_ds, train_dl, val_dl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()


if __name__ == "__main__":
    main()
