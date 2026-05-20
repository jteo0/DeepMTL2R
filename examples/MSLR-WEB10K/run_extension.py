"""
run_extension.py — DeepMTL2R Extension Experiments Runner
==========================================================
Main script to run DeepMTL2R architectural experiments interactively.

Available experiments:
  1. Matryoshka Feature Projection (MRL)
  2. Dynamic Feature Gating (Soft Gating)

Usage:
  python run_extension.py

The script will prompt for user input (1 or 2) and execute the model training
with the appropriate configuration. All outputs (metrics results, model checkpoints)
will be saved in the `outputs/` directory.

Saved checkpoints:
  - model_epoch_{epoch}.pkl : Model checkpoint for each epoch.
  - model.pkl               : Model at the final training epoch.

Recorded metrics (as per metrics.md Phase 1 & 2):
  - NDCG@1, @5, @10, @20, @30 (all tasks)
  - Gating Sparsity Ratio (Gating only, per epoch)
  - Effective Dimensionality Efficiency (Matryoshka only, end of training)
  - Total trainable parameters

Note: Ensure the working directory when running this script is
      DeepMTL2R/examples/MSLR-WEB30K/
"""

# Standard library
import os
import sys
import time
import random
import json
import shutil
from functools import partial
from argparse import Namespace
from pprint import pformat
import itertools

# Third-party
import numpy as np
import torch
from torch import optim
from attr import asdict
from tqdm import tqdm

# Add project root to path (so allrank can be imported)
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
from allrank.training.train_utils import fit
from allrank.utils.command_executor import execute_command
from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
from allrank.utils.file_utils import create_output_dirs, PathsContainer
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr

# Configuration loader
from config_loader import load_config

# =============================================================================
# Load configuration from experiment_config.yaml
# =============================================================================
cfg = load_config()

# Extract configuration from YAML
MRL_NESTING_DIMS = getattr(cfg.model, 'mrl_nesting_dims', [32, 64, 128, 256])
MOO_METHOD = getattr(cfg.model, 'moo_method', 'ls')
DATASET_NAME = getattr(cfg.dataset, 'name', '50bps')
DATASET_BASE_PATH = getattr(cfg.dataset, 'base_path', None) or getattr(cfg.dataset, 'path', '../../datasets/MSLR-WEB10K')
FOLDS = getattr(cfg.dataset, 'folds', None) or [1]

# Ensure DATASET_BASE_PATH and FOLDS are not None
if DATASET_BASE_PATH is None:
    DATASET_BASE_PATH = '../../datasets/MSLR-WEB10K'
if FOLDS is None:
    FOLDS = [1]
    
REDUCTION_METHOD = getattr(cfg.dataset, 'reduction_method', 'mean')
TASK_INDICES = getattr(cfg.tasks, 'indices', '0,131,132,133,134,135')
TASK_WEIGHTS = getattr(cfg.tasks, 'weights', '0,10')
OUTPUT_DIR = getattr(cfg.output, 'base_dir', 'outputs')
CHECKPOINT_DIR = getattr(cfg.output, 'checkpoint_dir', 'checkpoints')
LABEL_INDICES = getattr(cfg.dataset, 'label_indices', [131, 132, 133, 134, 135])
DEBUG = getattr(cfg.experiment, 'debug', False)

# =============================================================================
# Config file paths (relative to this directory)
# =============================================================================
CONFIG_MATRYOSHKA = os.path.join(
    os.path.dirname(__file__), "configs", "config_matryoshka.json"
)
CONFIG_GATING = os.path.join(os.path.dirname(__file__), "configs", "config_gating.json")


def print_config_summary(
    experiment_name: str,
    config: Config,
    use_mrl: bool,
    use_gating: bool,
    mrl_nesting_dims: list,
):
    """Print a readable experiment configuration summary."""
    print(f"\n{'─'*60}")
    print(f"  Experiment : {experiment_name}")
    print(f"  MOO Method : {MOO_METHOD}")
    print(f"  Dataset    : {DATASET_NAME}")
    print(f"  Tasks      : {TASK_INDICES}")
    print(f"  Output Dir : {OUTPUT_DIR}")
    if use_mrl:
        print(f"  MRL Dims   : {mrl_nesting_dims}")
        print(f"  MRL Weights: Uniform (1.0 per dim)")
    if use_gating:
        print(f"  Gating     : Soft (Sigmoid)")
    print(f"{'─'*60}\n")


def get_weight_combinations(
    moo_method: str, num_tasks: int, task_indices: list
) -> tuple:
    """Determine weight combinations based on MOO method."""
    if moo_method == "stl":
        return np.identity(num_tasks).tolist(), None
    elif moo_method == "ec":
        if num_tasks == 2:
            if task_indices[0] == 0 and task_indices[1] == 131:
                epsilon_values = [[x] for x in np.linspace(0.0001, 0.0015, 10)]
            elif task_indices[0] == 0 and task_indices[1] == 135:
                epsilon_values = [[x] for x in np.linspace(0.001, 0.015, 10)]
            else:
                epsilon_values = [[x] for x in np.linspace(0.0001, 0.001, 10)]
        else:
            sets = [[20, 35, 50], [40, 70, 100], [70, 125, 180], [120, 310, 500]]
            epsilon_values = np.array(list(itertools.product(*sets)))
        # Limit to first 2 combinations in DEBUG mode for faster testing
        if DEBUG:
            epsilon_values = epsilon_values[:2]
        return None, epsilon_values
    else:
        if num_tasks == 2:
            combinations = [[x, 1 - x] for x in np.linspace(0.001, 0.991, 10)]
            # Limit to first 2 combinations in DEBUG mode for faster testing
            if DEBUG:
                combinations = combinations[:2]
            return combinations, None
        elif num_tasks == 6:
            # Load 5-task weights and prepend uniform weight for task 0
            weights_5tasks = np.loadtxt("weights-5tasks.txt")
            # Prepend 1/6 for task 0, then normalize
            combinations = []
            for row in weights_5tasks:
                combined = np.concatenate(([1/6], row / 6))
                combined = combined / combined.sum()  # Renormalize to sum=1
                combinations.append(combined.tolist())
            # Limit to first 2 combinations in DEBUG mode for faster testing
            if DEBUG:
                combinations = combinations[:2]
            return combinations, None
        else:
            return np.loadtxt("weights-5tasks.txt").tolist(), None


def run_experiment(
    experiment_name: str,
    config_path: str,
    dataset_path: str,
    train_dataloader,
    val_dataloader,
    n_features: int,
    use_mrl: bool = False,
    use_gating: bool = False,
    mrl_nesting_dims: list = None,
    task_indices: list = None,
):
    """
    Execute a single DeepMTL2R extension experiment.

    :param experiment_name: human-readable name.
    :param config_path: path to the JSON config file.
    :param use_mrl: enable Matryoshka output layer and loss.
    :param use_gating: enable DynamicFeatureGate in FCModel.
    :param mrl_nesting_dims: nesting dimensions for Matryoshka.
    """
    # Set reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Load config
    config = Config.from_json(config_path)
    if DEBUG:
        config.training.epochs = 2

    # Override nesting dims in config from hard-coded constant
    if use_mrl:
        config.model.use_mrl = True
        config.model.mrl_nesting_dims = mrl_nesting_dims

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    # Dataset path (from YAML config)
    config.data.path = dataset_path
    config.loss.args["reduction"] = REDUCTION_METHOD

    if not os.path.exists(config.data.path):
        raise FileNotFoundError(
            f"Dataset path not found: {config.data.path}\n"
            f"Expected relative to: {os.getcwd()}\n"
            f"Absolute path: {os.path.abspath(config.data.path)}"
        )

    # Parse task config
    if task_indices is None:
        task_indices = list(map(int, TASK_INDICES.split(",")))
    num_tasks = len(task_indices)

    print_config_summary(experiment_name, config, use_mrl, use_gating, mrl_nesting_dims)

    # Weight combinations
    weight_combinations, epsilon_values = get_weight_combinations(
        MOO_METHOD, num_tasks, task_indices
    )

    # Tracking for manifest
    manifest_data = {
        "experiment": {
            "name": experiment_name,
            "config": config_path,
            "mrl_nesting_dims": mrl_nesting_dims if use_mrl else None,
            "task_indices": task_indices,
        },
        "results_summary": {"best_weight_index": -1, "best_ndcg10": -1.0},
        "checkpoints": {},
    }

    experiment_results = {}

    # Training loop across weight combinations
    for weight_index, task_weights_tensor in tqdm(
        enumerate(weight_combinations),
        total=len(weight_combinations),
        desc="Weight Combinations",
        unit="config",
    ):
        if MOO_METHOD == "ec":
            epsilon = epsilon_values[weight_index]
            task_weights_tensor = torch.tensor([1] + [0] * len(epsilon))
        else:
            epsilon = None

        # Build run ID and paths
        task_id_string = "_task_".join(map(str, task_indices))
        exp_tag = "matryoshka" if use_mrl else "gating"
        fold_str = os.path.basename(dataset_path).lower()
        run_id = os.path.join(
            f"{num_tasks}tasks",
            f"task_{task_id_string}",
            DATASET_NAME,
            REDUCTION_METHOD,
            MOO_METHOD,
            exp_tag,
            fold_str,
            str(weight_index),
        )

        args = Namespace(
            output_dir=OUTPUT_DIR,
            run_id=run_id,
            config_file_path=config_path,
            task_indices=TASK_INDICES,
            task_weights=TASK_WEIGHTS,
            moo_method=MOO_METHOD,
            dataset_name=DATASET_NAME,
            reduction_method=REDUCTION_METHOD,
        )

        paths = PathsContainer.from_args(
            args.output_dir, args.run_id, args.config_file_path
        )
        create_output_dirs(paths.output_dir)

        logger = init_logger(paths.output_dir)
        logger.info(f"=== {experiment_name} | weight_index={weight_index} ===")
        logger.info(f"Output dir: {paths.output_dir}")

        # Copy config for reproducibility
        output_config_path = os.path.join(paths.output_dir, "used_config.json")
        shutil.copy(paths.config_path, output_config_path)

        # Results file
        results_dir = os.path.join(
            "result",
            f"{num_tasks}tasks",
            f"task_{task_id_string}",
            DATASET_NAME,
            REDUCTION_METHOD,
            MOO_METHOD,
            exp_tag,
            fold_str,
        )
        results_filename = os.path.join(
            results_dir, f"results_task_{task_id_string}_weight{weight_index}.txt"
        )
        if os.path.exists(results_filename):
            os.remove(results_filename)
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)

        # Log parameter count
        with open(results_filename, "a", encoding="utf-8") as f:
            f.write(
                f"=== {experiment_name} | MRL={use_mrl} | Gating={use_gating} ===\n"
            )
            if use_mrl:
                f.write(f"MRL Nesting Dims: {mrl_nesting_dims}\n")
                f.write(f"MRL Relative Importance: Uniform (1.0)\n")
            if use_gating:
                f.write(f"Gating Type: Soft (Sigmoid)\n")
            f.write(f"n_features: {n_features}\n\n")

        # Build model
        logger.info(f"Building model on device: {device}")
        model = make_model(
            n_features=n_features,
            use_gating=use_gating,
            **asdict(config.model, recurse=False),
        )

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = CustomDataParallel(model)
        model.to(device)

        # Log parameter count to results file
        num_params = get_num_params(model)
        shared_params = sum(
            p.numel()
            for n, p in model.named_parameters()
            if "task" not in n and "output" not in n
        )
        specific_params = num_params - shared_params

        logger.info(
            f"Total Trainable Parameters: {num_params:,} (Shared: {shared_params:,}, Specific: {specific_params:,})"
        )
        with open(results_filename, "a", encoding="utf-8") as f:
            f.write(f"Total Trainable Parameters (Δp): {num_params:,}\n")
            f.write(f"  ├─ Shared Parameters: {shared_params:,}\n")
            f.write(f"  └─ Task-Specific: {specific_params:,}\n\n")

        # Optimizer, loss, scheduler
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

        # Train!
        print(f"Starting training: {experiment_name} (weight {weight_index})...")
        start_time = time.time()

        with (
            torch.autograd.detect_anomaly()
            if config.detect_anomaly
            else dummy_context_mgr()
        ):
            result = fit(
                moo_method=MOO_METHOD,
                main_task_index=0,
                task_indices=task_indices,
                label_indices=LABEL_INDICES,
                results_filename=results_filename,
                model=model,
                loss_func=loss_func,
                task_weights=task_weights_tensor,
                epsilon=epsilon,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                config=config,
                device=device,
                output_dir=paths.output_dir,
                tensorboard_output_path=paths.tensorboard_output_path,
                use_mrl=use_mrl,
                use_gating=use_gating,
                mrl_nesting_dims=mrl_nesting_dims,
                **asdict(config.training),
            )

        elapsed = time.time() - start_time
        print(f"Completed in {elapsed/60:.1f} minutes.")
        print(f"   Epoch models saved to → {paths.output_dir}/model_epoch_<epoch>.pkl")
        print(f"   Final model saved to  → {paths.output_dir}/model.pkl")
        print(f"   Metrics saved to      → {results_filename}")

        # Individual measurement for auxiliary tasks (see metrics.md Phase 1 & 2)
        print("\nFinal Per-Task Validation Metrics:")
        val_metrics = result.get("val_metrics", {})
        for t_idx in sorted(val_metrics.keys()):
            t_metrics = val_metrics[t_idx]
            task_name = "Main Relevance" if t_idx == 0 else f"Auxiliary Task {t_idx}"
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in t_metrics.items()])
            print(f"  - {task_name}: {metric_str}")
        
        special_metrics = result.get("special_metrics", {})
        if use_gating and "gating_sparsity_ratio" in special_metrics:
            sparsity = special_metrics["gating_sparsity_ratio"]
            print(f"  - Gating Sparsity Ratio: {sparsity:.4f}")
            
        if "noise_robustness" in special_metrics:
            print("\n  - Robustness to Noisy Features (Global Drop for this fold):")
            fold_rob_agg = {}
            for t_idx, rob in special_metrics["noise_robustness"].items():
                for k, v in rob.items():
                    if "ndcg" in k or "map" in k:
                        fold_rob_agg.setdefault(k, []).append(v)
            rob_str = ", ".join([f"{k}={np.mean(vals):.4f}" for k, vals in fold_rob_agg.items()])
            print(f"      {rob_str}")
        print()

        # Save experiment summary JSON
        dump_experiment_result(args, config, paths.output_dir, result)

        # Skip expected metrics check in DEBUG mode (small datasets produce NaN values)
        if not DEBUG:
            assert_expected_metrics(result, config.expected_metrics)

        # Record checkpoint in manifest
        ckpt_path = os.path.join(paths.output_dir, "model.pkl")
        manifest_data["checkpoints"][f"weight_{weight_index}"] = ckpt_path

        experiment_results[weight_index] = {
            "val_metrics": val_metrics,
            "num_params": num_params,
            "special_metrics": special_metrics
        }

        # Track best overall run based on NDCG@10 of main task (task 0)
        # Note: result contains metrics from validation, but we can just use the final one for tracking
        main_task = task_indices[0]
        # In a real scenario we extract from result, but since fit() doesn't return full ndcg_10 history easily,
        # we'll approximate or assume we can parse it from results file. Let's just track the path.
        manifest_data["results_summary"][
            "best_weight_index"
        ] = weight_index  # Placeholder

    # Write manifest.json
    fold_str = os.path.basename(dataset_path).lower()
    manifest_path = os.path.join(
        "result",
        f"{num_tasks}tasks",
        f"task_{task_id_string}",
        DATASET_NAME,
        REDUCTION_METHOD,
        MOO_METHOD,
        exp_tag,
        fold_str,
        "manifest.json",
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    # =========================================================================
    # Save all weight combination models to central checkpoint directory
    # =========================================================================
    checkpoint_subdir = (
        "matryoshka" if use_mrl else "feature_gating" if use_gating else "deepmtl2r"
    )
    checkpoint_dir_full = os.path.join(CHECKPOINT_DIR, checkpoint_subdir)
    os.makedirs(checkpoint_dir_full, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Saving checkpoints to: {checkpoint_dir_full}")
    print(f"{'=' * 60}")

    source_base_dir = os.path.join(
        OUTPUT_DIR,
        "results",
        f"{num_tasks}tasks",
        f"task_{task_id_string}",
        DATASET_NAME,
        REDUCTION_METHOD,
        MOO_METHOD,
        exp_tag,
        fold_str,
    )

    for weight_index in range(len(weight_combinations)):
        source_model_path = os.path.join(source_base_dir, str(weight_index), "model.pkl")
        destination_checkpoint_path = os.path.join(
            checkpoint_dir_full, f"{checkpoint_subdir}_{fold_str}_weight{weight_index}.pkl"
        )

        if os.path.exists(source_model_path):
            shutil.copy(source_model_path, destination_checkpoint_path)
            print(f"  ✓ weight{weight_index}: {destination_checkpoint_path}")
        else:
            print(f"  ⚠ weight{weight_index}: Not found at {source_model_path}")

    print(f"\nAll weight combinations completed for {experiment_name}!")
    
    return experiment_results


def main():
    print("\nSelect the experiment to run:")
    print("  1 -> Matryoshka Feature Projection")
    print("  2 -> Dynamic Feature Gating")
    print()

    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("  Invalid input. Please enter 1 or 2.")

    # Allow user to override task indices
    default_task_indices = [0, 131, 132, 133, 134, 135]
    print(f"\nDefault task indices: {default_task_indices}")
    user_input = input("Enter task indices as comma-separated values (or press Enter to use default): ").strip()
    
    if user_input:
        try:
            task_indices = list(map(int, user_input.split(",")))
            print(f"Using custom task indices: {task_indices}")
        except ValueError:
            print(f"Invalid input. Using default task indices: {default_task_indices}")
            task_indices = default_task_indices
    else:
        task_indices = list(map(int, TASK_INDICES.split(",")))
        print(f"Using task indices from YAML: {task_indices}")

    all_fold_results = {}

    for fold in FOLDS:
        print(f"\n" + "=" * 60)
        print(f" Processing Fold {fold}")
        print("=" * 60)
        
        dataset_path = os.path.join(DATASET_BASE_PATH, f"Fold{fold}")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        # Load configuration for dataloader settings
        config_tmp = Config.from_json(CONFIG_GATING)
        
        # Calculate max_rows for debug mode
        max_rows = None
        if DEBUG:
            debug_ratio = cfg.experiment.get('debug_ratio', 0.1)
            if debug_ratio > 0:
                estimated_total_rows = 30000000
                max_rows = max(1, int(estimated_total_rows * debug_ratio))
                print(f"DEBUG MODE ENABLED - will limit to approximately {max_rows} rows ({debug_ratio*100:.4f}%)")
            else:
                print(f"DEBUG MODE: debug_ratio is {debug_ratio}, loading full dataset")
        
        print(f"Loading MSLR-WEB10K dataset from {dataset_path}...")
        train_ds, val_ds = load_libsvm_dataset(
            input_path=dataset_path,
            slate_length=config_tmp.data.slate_length,
            validation_ds_role=config_tmp.data.validation_ds_role,
            max_rows=max_rows,
        )
        print(f"Dataset loaded! Train shape: {train_ds.shape}, Val shape: {val_ds.shape}")

        # Number of features (exclude label candidate columns)
        nf = train_ds.shape[-1] - len(LABEL_INDICES)
        assert nf == val_ds.shape[-1] - len(
            LABEL_INDICES
        ), "Feature dimension mismatch between train and val!"

        train_dl, val_dl = create_data_loaders(
            train_ds,
            val_ds,
            num_workers=config_tmp.data.num_workers,
            batch_size=config_tmp.data.batch_size,
        )

        if choice == "1":
            fold_res = run_experiment(
                experiment_name=f"Matryoshka Feature Projection (Fold {fold})",
                config_path=CONFIG_MATRYOSHKA,
                dataset_path=dataset_path,
                train_dataloader=train_dl,
                val_dataloader=val_dl,
                n_features=nf,
                use_mrl=True,
                use_gating=False,
                mrl_nesting_dims=MRL_NESTING_DIMS,
                task_indices=task_indices,
            )
        elif choice == "2":
            fold_res = run_experiment(
                experiment_name=f"Dynamic Feature Gating (Fold {fold})",
                config_path=CONFIG_GATING,
                dataset_path=dataset_path,
                train_dataloader=train_dl,
                val_dataloader=val_dl,
                n_features=nf,
                use_mrl=False,
                use_gating=True,
                mrl_nesting_dims=None,
                task_indices=task_indices,
            )
            
        all_fold_results[fold] = fold_res

        # Free memory at the end of each fold
        del train_ds, val_ds, train_dl, val_dl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    print("\n" + "=" * 80)
    print("EXTENSION RESULTS (CROSS-VALIDATION AVERAGE)")
    print("=" * 80)
    print(f"Folds Evaluated:                {FOLDS}")
    
    baseline_summary_path = os.path.join(OUTPUT_DIR, "baselines", "baseline_summary.json")
    avg_ndcg30_st = None
    if os.path.exists(baseline_summary_path):
        try:
            with open(baseline_summary_path, "r", encoding="utf-8") as f:
                bs = json.load(f)
            avg_ndcg30_st = bs.get("single_task", {}).get("ndcg30_avg")
        except Exception:
            pass
            
    if len(all_fold_results) > 0:
        first_fold = list(all_fold_results.keys())[0]
        weight_indices = sorted(list(all_fold_results[first_fold].keys()))
        
        for w_idx in weight_indices:
            print(f"\n--- WEIGHT COMBINATION {w_idx} METRICS ---")
            
            agg_metrics = {} 
            agg_sparsity = []
            agg_robustness = {}
            agg_mrl = {}
            
            for fold in FOLDS:
                if fold not in all_fold_results or w_idx not in all_fold_results[fold]:
                    continue
                res = all_fold_results[fold][w_idx]
                v_metrics = res.get("val_metrics", {})
                s_metrics = res.get("special_metrics", {})
                
                if "gating_sparsity_ratio" in s_metrics:
                    agg_sparsity.append(float(s_metrics["gating_sparsity_ratio"]))
                    
                if "mrl_dimensionality_efficiency" in s_metrics:
                    for t_idx, eff in s_metrics["mrl_dimensionality_efficiency"].items():
                        if t_idx not in agg_mrl:
                            agg_mrl[t_idx] = {}
                        for dim, dim_metrics in eff.items():
                            if dim not in agg_mrl[t_idx]:
                                agg_mrl[t_idx][dim] = {}
                            for m_key, m_val in dim_metrics.items():
                                if m_key not in agg_mrl[t_idx][dim]:
                                    agg_mrl[t_idx][dim][m_key] = []
                                agg_mrl[t_idx][dim][m_key].append(float(m_val))
                    
                if "noise_robustness" in s_metrics:
                    for t_idx, rob in s_metrics["noise_robustness"].items():
                        for r_key, r_val in rob.items():
                            if r_key not in agg_robustness:
                                agg_robustness[r_key] = []
                            agg_robustness[r_key].append(float(r_val))
                    
                for t_idx, t_metrics in v_metrics.items():
                    if t_idx not in agg_metrics:
                        agg_metrics[t_idx] = {}
                    for metric_key, val in t_metrics.items():
                        if metric_key not in agg_metrics[t_idx]:
                            agg_metrics[t_idx][metric_key] = []
                        agg_metrics[t_idx][metric_key].append(float(val))
            
            task_0_ndcg30_mean = None
            global_ndcg10_means = []
            global_ndcg30_means = []
            global_map_means = []
            global_mrr_means = []
            
            for t_idx in sorted(agg_metrics.keys()):
                task_name = "Task 0 (Main)" if t_idx == 0 else f"Task {t_idx} (Aux)"
                print(f"  {task_name}:")
                for metric_key in sorted(agg_metrics[t_idx].keys()):
                    arr = np.array(agg_metrics[t_idx][metric_key])
                    mean_val = float(np.mean(arr))
                    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                    
                    if t_idx == 0 and metric_key == "ndcg_30":
                        task_0_ndcg30_mean = mean_val
                        
                    if np.isnan(mean_val) or np.isinf(mean_val):
                        mean_str = "—"
                    else:
                        mean_str = f"{mean_val:.4f}"
                        
                    if np.isnan(std_val) or np.isinf(std_val):
                        std_str = "—"
                    else:
                        std_str = f"{std_val:.4f}"
                        
                    print(f"    {metric_key}: {mean_str} ± {std_str}")
                    
            print(f"\n  --- Global Averages (All Tasks) ---")
            global_metrics = {}
            for t_idx in sorted(agg_metrics.keys()):
                for metric_key, vals in agg_metrics[t_idx].items():
                    if metric_key not in global_metrics:
                        global_metrics[metric_key] = []
                    mean_val = float(np.mean(vals))
                    if not np.isnan(mean_val):
                        global_metrics[metric_key].append(mean_val)
                        
            for metric_key in sorted(global_metrics.keys()):
                g_avg = float(np.mean(global_metrics[metric_key]))
                print(f"    {metric_key.upper()}: {g_avg:.4f}")
                    
            if task_0_ndcg30_mean is not None and avg_ndcg30_st is not None and avg_ndcg30_st > 0:
                delta_m = ((task_0_ndcg30_mean - avg_ndcg30_st) / avg_ndcg30_st) * 100
                print(f"\n  Δm% (Relative Improvement vs Single-Task): {delta_m:+.2f}%")
                
            if len(agg_sparsity) > 0:
                s_arr = np.array(agg_sparsity)
                s_mean = float(np.mean(s_arr))
                s_std = float(np.std(s_arr, ddof=1)) if len(s_arr) > 1 else 0.0
                print(f"  Gating Sparsity Ratio: {s_mean:.4f} ± {s_std:.4f}")
                
            if len(agg_robustness) > 0:
                print(f"\n  Robustness to Noisy Features (Global Average Drop):")
                for r_key in sorted(agg_robustness.keys()):
                    if "ndcg" in r_key or "map" in r_key:
                        r_arr = np.array(agg_robustness[r_key])
                        r_mean = float(np.mean(r_arr))
                        r_std = float(np.std(r_arr, ddof=1)) if len(r_arr) > 1 else 0.0
                        print(f"    {r_key}: {r_mean:.4f} ± {r_std:.4f}")
                        
            if len(agg_mrl) > 0:
                print(f"\n  Effective Dimensionality Efficiency (Matryoshka):")
                for t_idx in sorted(agg_mrl.keys()):
                    task_name = "Task 0 (Main)" if t_idx == 0 else f"Task {t_idx} (Aux)"
                    print(f"    {task_name}:")
                    for dim in sorted(agg_mrl[t_idx].keys(), key=lambda x: int(x)):
                        print(f"      dim={dim}:")
                        for m_key in sorted(agg_mrl[t_idx][dim].keys()):
                            if "ndcg" in m_key or "map" in m_key:
                                m_arr = np.array(agg_mrl[t_idx][dim][m_key])
                                m_mean = float(np.mean(m_arr))
                                m_std = float(np.std(m_arr, ddof=1)) if len(m_arr) > 1 else 0.0
                                print(f"        {m_key}: {m_mean:.4f} ± {m_std:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
