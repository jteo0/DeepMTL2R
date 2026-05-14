"""
run_loss_weighting.py — Final Experiment: Loss Weighting Strategies (Priority 2)
==================================================================================
Late-stage fine-tuning experiment that loads checkpoints from the two
primary experiments (Matryoshka & Dynamic Feature Gating) and compares
the effect of different loss weighting strategies on multi-task balance.

Tested Strategies:
  - Uniform Weighting (ls — Linear Scalarization with equal weights)
  - Uncertainty Weighting (uw — Kendall et al., 2018)
  - Dynamic Weight Averaging (dwa — Liu et al., 2019)

Metrics recorded (per metrics.md Phase 3 — Priority 2):
  - NDCG@10, NDCG@20 (per task, per epoch)
  - Task Performance Variance: variance of NDCG difference between
    auxiliary tasks and main task
  - Minority Task NDCG Retention Ratio: ratio of auxiliary task NDCG
    retention relative to baseline checkpoint

Usage:
  python run_loss_weighting.py

The script will interactively ask:
  1. Which primary experiment checkpoint to load (Matryoshka / Gating)
  2. The path to the checkpoint .pkl file

Then it runs all 3 loss weighting strategies on that checkpoint.

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
from config_loader import load_config, get_loss_weighting_config

# =============================================================================
# Load configuration from experiment_config.yaml
# =============================================================================
cfg = load_config()

# Extract configuration from YAML
FINETUNE_EPOCHS = cfg.finetuning.epochs
FINETUNE_PATIENCE = cfg.finetuning.patience
DATASET_NAME = cfg.dataset.name
DATASET_PATH = cfg.dataset.path
REDUCTION_METHOD = cfg.dataset.reduction_method
TASK_INDICES = cfg.tasks.indices
TASK_WEIGHTS = cfg.tasks.weights
LABEL_INDICES = cfg.dataset.label_indices
OUTPUT_DIR = cfg.output.base_dir
CHECKPOINT_DIR = cfg.output.checkpoint_dir
DEBUG = cfg.experiment.debug
MRL_NESTING_DIMS = cfg.model.mrl_nesting_dims
WEIGHTING_STRATEGIES = (
    get_loss_weighting_config()
)  # Load all loss weighting strategies from YAML

# Adjust FINETUNE_EPOCHS for DEBUG mode
if DEBUG:
    FINETUNE_EPOCHS = 2

# Config files
CONFIG_MATRYOSHKA = os.path.join(
    os.path.dirname(__file__), "configs", "config_matryoshka.json"
)
CONFIG_GATING = os.path.join(os.path.dirname(__file__), "configs", "config_gating.json")


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


def compute_task_performance_variance(val_results, task_indices, metric_key="ndcg_10"):
    """
    Compute Task Performance Variance.
    Variance of the difference between auxiliary task NDCG and main task NDCG.

    A lower variance means the loss weighting strategy successfully balanced
    the trade-off between tasks.
    """
    main_task = task_indices[0]
    main_ndcg = val_results[main_task].get(metric_key, 0.0)

    diffs = []
    for ti in task_indices[1:]:
        aux_ndcg = val_results[ti].get(metric_key, 0.0)
        diffs.append(aux_ndcg - main_ndcg)

    if len(diffs) == 0:
        return 0.0
    return float(np.var(diffs))


def compute_minority_retention_ratio(
    val_results, baseline_results, task_indices, metric_key="ndcg_10"
):
    """
    Compute Minority Task NDCG Retention Ratio.
    For each auxiliary (minority) task, compute:
        retention = current_ndcg / baseline_ndcg

    A ratio close to 1.0 means the minority task performance is retained.
    """
    retention_ratios = {}
    for ti in task_indices[1:]:  # Skip main task
        baseline_ndcg = baseline_results[ti].get(metric_key, 0.0)
        current_ndcg = val_results[ti].get(metric_key, 0.0)

        if baseline_ndcg > 0:
            retention_ratios[ti] = current_ndcg / baseline_ndcg
        else:
            retention_ratios[ti] = 1.0 if current_ndcg == 0 else float("inf")

    return retention_ratios


def finetune_with_weighting(
    strategy_name,
    strategy_config,
    model_state_dict,
    model_kwargs,
    config,
    train_dataloader,
    val_dataloader,
    task_indices,
    device,
    use_mrl,
    use_gating,
    baseline_results,
    results_dir,
    experiment_tag,
    logger,
):
    """
    Perform late-stage fine-tuning with a specific loss weighting strategy.
    Returns a dict with all tracked metrics.
    """
    moo_method = strategy_config["moo_method"]

    print(f"\n{'━'*60}")
    print(f"  Loss Weighting: {strategy_name}")
    print(f"  MOO Method: {moo_method}")
    print(f"  Description: {strategy_config['description']}")
    print(f"{'━'*60}")

    # Rebuild model and load checkpoint
    model = make_model(**model_kwargs)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
    model.to(device)

    # Optimizer: use AdamW as default (we're only varying loss weighting)
    optimizer = optim.AdamW(params=model.parameters(), lr=1e-4, weight_decay=1e-2)

    # Loss function
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)

    # Task weights
    num_tasks = len(task_indices)
    task_weights_raw = list(map(int, TASK_WEIGHTS.split(",")))
    min_weight, max_weight = task_weights_raw
    weight_index = (min_weight + max_weight) >> 1

    if moo_method == "ls":
        # Uniform: equal weights
        task_weights_tensor = [1.0 / num_tasks] * num_tasks
    else:
        # For UW and DWA, task_weights are not used directly (they learn weights)
        # But the WeightMethods init expects them for consistency
        if num_tasks == 2:
            weight_combinations = [[x, 1 - x] for x in np.linspace(0.001, 0.991, 10)]
        else:
            weight_combinations = np.loadtxt("weights-5tasks.txt").tolist()
        task_weights_tensor = weight_combinations[weight_index]

    weight_method = WeightMethods(
        moo_method,
        n_tasks=num_tasks,
        device=device,
        task_weights=task_weights_tensor,
        epsilon=None,
    )

    # Add UW learnable parameters to optimizer if uncertainty weighting
    if moo_method == "uw":
        try:
            uw_params = list(weight_method.parameters())
            if uw_params:
                optimizer.add_param_group({"params": uw_params, "lr": 1e-3})
            else:
                print("  [WARNING] No learnable parameters for UW weights found.")
        except AttributeError:
            print(
                "  [ERROR] WeightMethods.parameters() not implemented for UW. Weights won't learn."
            )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    # Baseline metrics
    baseline_ndcg10_main = baseline_results[task_indices[0]].get("ndcg_10", 0.0)

    # Results file
    results_filename = os.path.join(
        results_dir, f"weighting_{strategy_name}_{experiment_tag}.txt"
    )
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    if os.path.exists(results_filename):
        os.remove(results_filename)

    with open(results_filename, "a") as f:
        f.write(f"=== Loss Weighting: {strategy_name} on {experiment_tag} ===\n")
        f.write(f"MOO Method: {moo_method}\n")
        f.write(f"Description: {strategy_config['description']}\n")
        f.write(f"Baseline NDCG@10 (main): {baseline_ndcg10_main:.6f}\n")
        f.write(f"Fine-tune Epochs: {FINETUNE_EPOCHS}\n")
        f.write(f"Task Indices: {task_indices}\n")
        f.write(f"Baseline per-task metrics:\n")
        for ti in task_indices:
            f.write(f"  Task {ti}: {baseline_results[ti]}\n")
        f.write("\n")

    # Tracking
    variance_history = []
    retention_history = []
    val_ndcg10_history = []
    epoch_weight_logs = []

    for epoch in tqdm(
        range(FINETUNE_EPOCHS), desc=f"Fine-tuning [{strategy_name}]", unit="epoch"
    ):
        model.train()
        train_loss_values = {ti: [] for ti in task_indices}
        train_nums = []

        for batch_id, batch in enumerate(
            tqdm(
                train_dataloader,
                desc=f"Epoch {epoch}/{FINETUNE_EPOCHS} [{strategy_name}]",
            )
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
            loss_weighted_sum, extra = weight_method.backward(
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

        # Log learned weights (for UW/DWA)
        w = None
        if moo_method == "uw" and hasattr(weight_method.method, "logsigma"):
            w = torch.exp(weight_method.method.logsigma).detach().cpu().numpy().tolist()
        elif moo_method == "dwa" and hasattr(weight_method.method, "task_weights"):
            w = weight_method.method.task_weights.detach().cpu().numpy().tolist()
        elif extra and "weights" in extra:
            w = extra["weights"]
            if isinstance(w, torch.Tensor):
                w = w.detach().cpu().numpy().tolist()

        if w is not None:
            epoch_weight_logs.append({"epoch": epoch, "weights": w})

        # Gating sparsity
        if use_gating:
            log_gating_sparsity(epoch, model, results_filename)

        # Validation
        val_results, val_losses = evaluate_model(
            model, val_dataloader, config, device, task_indices, loss_func, use_mrl
        )

        # Compute Phase 3 Priority 2 metrics
        current_ndcg10 = val_results[task_indices[0]].get("ndcg_10", 0.0)
        val_ndcg10_history.append(current_ndcg10)

        task_var = compute_task_performance_variance(val_results, task_indices)
        variance_history.append(task_var)

        retention = compute_minority_retention_ratio(
            val_results, baseline_results, task_indices
        )
        retention_history.append(retention)

        # Log epoch results
        lr = get_current_lr(optimizer)
        with open(results_filename, "a") as f:
            f.write(f"epoch:{epoch}\tlr:{lr:.2e}\t")
            f.write(f"TaskPerfVariance:{task_var:.8f}\t")
            for ti in task_indices:
                f.write(f"task:{ti}\tVal Metrics:{val_results[ti]}\t")
                f.write(f"Val Loss:{val_losses[ti]:.6f}\t")
            if ti in retention:
                f.write(f"RetentionRatio:{retention[ti]:.6f}\t")
            f.write("\n")

        logger.info(
            f"[{strategy_name}] Epoch {epoch}/{FINETUNE_EPOCHS-1} | "
            f"NDCG@10={current_ndcg10:.6f} | Variance={task_var:.8f} | "
            f"Retention={retention}"
        )

        scheduler.step(current_ndcg10)

    # Final metrics
    final_variance = variance_history[-1] if variance_history else 0.0
    avg_variance = float(np.mean(variance_history))
    final_retention = retention_history[-1] if retention_history else {}

    # Compute average retention across minority tasks
    if final_retention:
        avg_retention = float(np.mean(list(final_retention.values())))
    else:
        avg_retention = 1.0

    summary = {
        "strategy": strategy_name,
        "moo_method": moo_method,
        "experiment": experiment_tag,
        "baseline_ndcg10_main": baseline_ndcg10_main,
        "final_ndcg10_main": val_ndcg10_history[-1] if val_ndcg10_history else 0.0,
        "peak_ndcg10_main": max(val_ndcg10_history) if val_ndcg10_history else 0.0,
        "final_task_variance": final_variance,
        "average_task_variance": avg_variance,
        "variance_history": variance_history,
        "final_minority_retention": {str(k): v for k, v in final_retention.items()},
        "average_minority_retention": avg_retention,
        "ndcg10_history": val_ndcg10_history,
        "weight_logs": epoch_weight_logs,
        "all_task_final_metrics": {str(ti): val_results[ti] for ti in task_indices},
    }

    # Write summary
    with open(results_filename, "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"SUMMARY: {strategy_name} on {experiment_tag}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Final NDCG@10 (main):        {summary['final_ndcg10_main']:.6f}\n")
        f.write(f"Peak NDCG@10 (main):         {summary['peak_ndcg10_main']:.6f}\n")
        f.write(f"Final Task Perf Variance:    {final_variance:.8f}\n")
        f.write(f"Average Task Perf Variance:  {avg_variance:.8f}\n")
        f.write(f"Final Minority Retention:     {final_retention}\n")
        f.write(f"Average Minority Retention:   {avg_retention:.6f}\n")
        f.write(f"Variance History:             {variance_history}\n")

    # Save summary JSON
    json_path = os.path.join(
        results_dir, f"weighting_{strategy_name}_{experiment_tag}_summary.json"
    )
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Task Perf Variance: {final_variance:.8f}")
    print(f"  Minority Retention: {avg_retention:.6f}")
    print(f"  Results saved to: {results_filename}")

    return summary


def run_loss_weighting_experiment(experiment_choice, checkpoint_path):
    """Run the full loss weighting comparison for one primary experiment checkpoint."""
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
        experiment_tag = "gating"
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
    config.data.path = DATASET_PATH
    config.loss.args["reduction"] = REDUCTION_METHOD

    # Parse tasks
    task_indices = list(map(int, TASK_INDICES.split(",")))

    print(f"\n{'═'*60}")
    print(f"  FINAL EXPERIMENT — Priority 2: Loss Weighting Strategies")
    print(f"  Primary Experiment: {experiment_tag.upper()}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Fine-tune Epochs: {FINETUNE_EPOCHS}")
    print(f"  Strategies: {list(WEIGHTING_STRATEGIES.keys())}")
    print(f"{'═'*60}\n")

    # Load dataset
    print("Loading MSLR-WEB30K dataset...")

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

    train_dataset, val_dataset = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
        max_rows=max_rows,
    )

    n_features = train_dataset.shape[-1] - len(LABEL_INDICES)
    assert n_features == val_dataset.shape[-1] - len(LABEL_INDICES)

    train_dataloader, val_dataloader = create_data_loaders(
        train_dataset,
        val_dataset,
        num_workers=config.data.num_workers,
        batch_size=config.data.batch_size,
    )

    # Build model architecture
    model_kwargs = dict(
        n_features=n_features,
        **asdict(config.model, recurse=False),
    )

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_state = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint_state, dict) and "model_state_dict" in checkpoint_state:
        model_state_dict = checkpoint_state["model_state_dict"]
    else:
        model_state_dict = checkpoint_state

    # Evaluate baseline checkpoint
    print("Evaluating baseline checkpoint...")
    baseline_model = make_model(**model_kwargs)
    baseline_model.load_state_dict(model_state_dict)
    baseline_model.to(device)

    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
    baseline_results, _ = evaluate_model(
        baseline_model, val_dataloader, config, device, task_indices, loss_func, use_mrl
    )

    print("Baseline metrics per task:")
    for ti in task_indices:
        print(f"  Task {ti}: {baseline_results[ti]}")

    results_dir = os.path.join(
        "result", "final_experiment", "loss_weighting", experiment_tag
    )
    os.makedirs(results_dir, exist_ok=True)

    # Save baseline
    baseline_file = os.path.join(results_dir, f"baseline_{experiment_tag}.txt")
    with open(baseline_file, "w") as f:
        f.write(f"=== Baseline Metrics ({experiment_tag}) ===\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        for ti in task_indices:
            f.write(f"Task {ti}: {baseline_results[ti]}\n")

    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Logger
    logger = init_logger(results_dir)

    # Run each weighting strategy
    all_summaries = {}
    for strat_name, strat_config in tqdm(
        WEIGHTING_STRATEGIES.items(), desc="Loss Weighting Strategies", unit="strat"
    ):
        summary = finetune_with_weighting(
            strategy_name=strat_name,
            strategy_config=strat_config,
            model_state_dict=copy.deepcopy(model_state_dict),
            model_kwargs=model_kwargs,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            task_indices=task_indices,
            device=device,
            use_mrl=use_mrl,
            use_gating=use_gating,
            baseline_results=baseline_results,
            results_dir=results_dir,
            experiment_tag=experiment_tag,
            logger=logger,
        )
        if summary is not None:
            all_summaries[strat_name] = summary

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print comparison table
    print(f"\n{'═'*78}")
    print(f"  LOSS WEIGHTING COMPARISON — {experiment_tag.upper()}")
    print(f"{'═'*78}")
    print(f"  {'Strategy':<16} {'NDCG@10':>10} {'TaskVar':>12} {'MinRetention':>14}")
    print(f"  {'─'*16} {'─'*10} {'─'*12} {'─'*14}")
    for name, s in all_summaries.items():
        print(
            f"  {name:<16} {s['final_ndcg10_main']:>10.6f} "
            f"{s['final_task_variance']:>12.8f} "
            f"{s['average_minority_retention']:>14.6f}"
        )
    print(f"{'═'*78}\n")

    # Save combined comparison JSON
    comparison_path = os.path.join(results_dir, f"comparison_{experiment_tag}.json")
    with open(comparison_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"Combined results saved to: {comparison_path}")


def main():
    print("\n" + "═" * 60)
    print("  DeepMTL2R Final Experiment — Priority 2")
    print("  Loss Weighting (Uniform vs Uncertainty vs DWA)")
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

    # =========================================================================
    # Auto-discover checkpoints from central directory
    # =========================================================================
    available_checkpoints = {
        "deepmtl2r.pkl": "Baseline DeepMTL2R",
        "matryoshka.pkl": "Matryoshka MRL",
        "feature_gating.pkl": "Feature Gating",
    }

    print("\nAvailable checkpoints in central directory:")
    print("=" * 60)
    found_checkpoints = {}

    for ckpt_name, description in available_checkpoints.items():
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        if os.path.exists(ckpt_path):
            found_checkpoints[ckpt_name] = ckpt_path
            print(f"  ✓ {ckpt_name:25} ({description})")
        else:
            print(f"  ✗ {ckpt_name:25} (not found)")

    if not found_checkpoints:
        print(f"\n[ERROR] No checkpoints found in {CHECKPOINT_DIR}")
        print(
            "Please run run_extension.py and run_baseline.py first to generate checkpoints."
        )
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Checkpoint selection:")

    # If user selected baseline, use deepmtl2r.pkl. Otherwise, use the selected one.
    if choice == "1":
        if "matryoshka.pkl" in found_checkpoints:
            checkpoint_path = found_checkpoints["matryoshka.pkl"]
            print(f"Selected: {checkpoint_path}")
        else:
            print("[ERROR] Matryoshka checkpoint not found.")
            sys.exit(1)
    elif choice == "2":
        if "feature_gating.pkl" in found_checkpoints:
            checkpoint_path = found_checkpoints["feature_gating.pkl"]
            print(f"Selected: {checkpoint_path}")
        else:
            print("[ERROR] Feature Gating checkpoint not found.")
            sys.exit(1)
    else:  # choice == "3" (baseline)
        if "deepmtl2r.pkl" in found_checkpoints:
            checkpoint_path = found_checkpoints["deepmtl2r.pkl"]
            print(f"Selected: {checkpoint_path}")
        else:
            print("[ERROR] Baseline checkpoint (deepmtl2r.pkl) not found.")
            sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"\n  [ERROR] Checkpoint not found: {checkpoint_path}")
        print(
            "  Make sure you have run the primary experiment first (run_extension.py)"
        )
        sys.exit(1)

    run_loss_weighting_experiment(choice, checkpoint_path)


if __name__ == "__main__":
    main()
