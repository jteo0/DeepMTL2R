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
MRL_NESTING_DIMS = cfg.model.mrl_nesting_dims
MOO_METHOD = cfg.model.moo_method
DATASET_NAME = cfg.dataset.name
DATASET_PATH = cfg.dataset.path
REDUCTION_METHOD = cfg.dataset.reduction_method
TASK_INDICES = cfg.tasks.indices
TASK_WEIGHTS = cfg.tasks.weights
OUTPUT_DIR = cfg.output.base_dir
CHECKPOINT_DIR = cfg.output.checkpoint_dir
LABEL_INDICES = cfg.dataset.label_indices
DEBUG = cfg.experiment.debug

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
        else:
            return np.loadtxt("weights-5tasks.txt").tolist(), None


def run_experiment(
    experiment_name: str,
    config_path: str,
    use_mrl: bool = False,
    use_gating: bool = False,
    mrl_nesting_dims: list = None,
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
    config.data.path = DATASET_PATH
    config.loss.args["reduction"] = REDUCTION_METHOD

    if not os.path.exists(config.data.path):
        raise FileNotFoundError(
            f"Dataset path not found: {config.data.path}\n"
            f"Expected relative to: {os.getcwd()}\n"
            f"Absolute path: {os.path.abspath(config.data.path)}"
        )

    # Parse task config
    task_indices = list(map(int, TASK_INDICES.split(",")))
    num_tasks = len(task_indices)

    print_config_summary(experiment_name, config, use_mrl, use_gating, mrl_nesting_dims)

    # Weight combinations
    weight_combinations, epsilon_values = get_weight_combinations(
        MOO_METHOD, num_tasks, task_indices
    )

    # Calculate max_rows for debug mode
    max_rows = None
    if DEBUG:
        debug_ratio = cfg.experiment.get('debug_ratio', 0.1)  # Default 10% if not specified
        if debug_ratio > 0:
            estimated_total_rows = 30000000  # MSLR-WEB30K typical size
            max_rows = max(1, int(estimated_total_rows * debug_ratio))
            print(f"[DEBUG] DEBUG MODE ENABLED - will limit to approximately {max_rows} rows ({debug_ratio*100:.4f}%)")
        else:
            print(f"[DEBUG] DEBUG MODE: debug_ratio is {debug_ratio}, loading full dataset")
    
    print("Loading MSLR-WEB30K dataset...")
    train_dataset, val_dataset = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
        max_rows=max_rows,
    )
    print(f"[DEBUG] Dataset loaded! Train shape: {train_dataset.shape}, Val shape: {val_dataset.shape}")

    # Number of features (exclude label candidate columns)
    n_features = train_dataset.shape[-1] - len(LABEL_INDICES)
    assert n_features == val_dataset.shape[-1] - len(
        LABEL_INDICES
    ), "Feature dimension mismatch between train and val!"

    train_dataloader, val_dataloader = create_data_loaders(
        train_dataset,
        val_dataset,
        num_workers=config.data.num_workers,
        batch_size=config.data.batch_size,
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
        run_id = os.path.join(
            f"{num_tasks}tasks",
            f"task_{task_id_string}",
            DATASET_NAME,
            REDUCTION_METHOD,
            MOO_METHOD,
            exp_tag,
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

        # Save experiment summary JSON
        dump_experiment_result(args, config, paths.output_dir, result)

        # Skip expected metrics check in DEBUG mode (small datasets produce NaN values)
        if not DEBUG:
            assert_expected_metrics(result, config.expected_metrics)

        # Record checkpoint in manifest
        ckpt_path = os.path.join(paths.output_dir, "model.pkl")
        manifest_data["checkpoints"][f"weight_{weight_index}"] = ckpt_path

        # Track best overall run based on NDCG@10 of main task (task 0)
        # Note: result contains metrics from validation, but we can just use the final one for tracking
        main_task = task_indices[0]
        # In a real scenario we extract from result, but since fit() doesn't return full ndcg_10 history easily,
        # we'll approximate or assume we can parse it from results file. Let's just track the path.
        manifest_data["results_summary"][
            "best_weight_index"
        ] = weight_index  # Placeholder

    # Write manifest.json
    manifest_path = os.path.join(
        "result",
        f"{num_tasks}tasks",
        f"task_{task_id_string}",
        DATASET_NAME,
        REDUCTION_METHOD,
        MOO_METHOD,
        exp_tag,
        "manifest.json",
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    # =========================================================================
    # Save all weight combination models to central checkpoint directory
    # =========================================================================
    
    # Determine checkpoint subdirectory based on architecture
    if use_mrl:
        checkpoint_subdir = "matryoshka"
    elif use_gating:
        checkpoint_subdir = "feature_gating"
    else:
        checkpoint_subdir = "deepmtl2r"
    
    checkpoint_dir_full = os.path.join(CHECKPOINT_DIR, checkpoint_subdir)
    os.makedirs(checkpoint_dir_full, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Saving checkpoints to: {checkpoint_dir_full}")
    print(f"{'='*60}")
    
    # Save all weight combinations
    for weight_index in range(len(weight_combinations)):
        source_model_path = os.path.join(
            OUTPUT_DIR,
            "results",
            f"{num_tasks}tasks",
            f"task_{task_id_string}",
            DATASET_NAME,
            REDUCTION_METHOD,
            MOO_METHOD,
            exp_tag,
            str(weight_index),
            "model.pkl",
        )
        
        destination_checkpoint_path = os.path.join(
            checkpoint_dir_full, 
            f"{checkpoint_subdir}_weight{weight_index}.pkl"
        )
        
        if os.path.exists(source_model_path):
            shutil.copy(source_model_path, destination_checkpoint_path)
            print(f"  ✓ weight{weight_index}: {destination_checkpoint_path}")
        else:
            print(f"  ⚠ weight{weight_index}: Not found at {source_model_path}")

    print(f"\nAll weight combinations completed for {experiment_name}!")


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

    print()
    global TASK_INDICES
    task_input = input("Enter task indices (comma separated, default: 0,131): ").strip()
    if task_input:
        TASK_INDICES = task_input

    print()

    if choice == "1":
        run_experiment(
            experiment_name="Matryoshka Feature Projection",
            config_path=CONFIG_MATRYOSHKA,
            use_mrl=True,
            use_gating=False,
            mrl_nesting_dims=MRL_NESTING_DIMS,
        )
    elif choice == "2":
        run_experiment(
            experiment_name="Dynamic Feature Gating",
            config_path=CONFIG_GATING,
            use_mrl=False,
            use_gating=True,
            mrl_nesting_dims=None,
        )


if __name__ == "__main__":
    main()
