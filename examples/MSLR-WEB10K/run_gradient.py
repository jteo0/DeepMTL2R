"""
run_gradient.py — Final Experiment: Gradient Dynamics & LR Schedule (Priority 3)
==================================================================================
Late-stage fine-tuning with Cosine Annealing / OneCycleLR and gradient
dynamics analysis (gradient conflict norm, gradient sparsity tracking).

Metrics (metrics.md Phase 3 — Priority 3):
  - NDCG@10, NDCG@20 per task per epoch
  - Gradient Conflict Norm (cosine similarity between task gradients)
  - Gradient Sparsity Tracking (% near-zero gradient elements)

Usage:  python run_gradient.py
"""

import os, sys, time, random, json, copy
from functools import partial
from tqdm import tqdm

import numpy as np
import torch
from torch import optim
from attr import asdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import allrank.models.losses as losses
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import (
    CustomDataParallel,
    get_num_params,
    get_model_parameters,
)
from allrank.training.train_utils import (
    compute_metrics,
    loss_batch,
    log_gating_sparsity,
    get_current_lr,
)
from allrank.utils.ltr_logging import init_logger
from allrank.methods.weight_methods import WeightMethods
from allrank.data.dataset_loading import PADDED_Y_VALUE

# Configuration loader
from config_loader import load_config, get_scheduler_config

# ── Load configuration from experiment_config.yaml ────────────────────────────────────────────
cfg = load_config()

FINETUNE_EPOCHS = cfg.finetuning.epochs
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
MRL_NESTING_DIMS = cfg.model.mrl_nesting_dims
MOO_METHOD = cfg.model.moo_method
GRAD_SPARSITY_THRESHOLD = cfg.gradient.sparsity_threshold
SCHEDULER_CONFIGS = get_scheduler_config()  # Load all scheduler configs from YAML

# Adjust FINETUNE_EPOCHS for DEBUG mode
if DEBUG:
    FINETUNE_EPOCHS = 2

CONFIG_MATRYOSHKA = os.path.join(
    os.path.dirname(__file__), "configs", "config_matryoshka.json"
)
CONFIG_GATING = os.path.join(os.path.dirname(__file__), "configs", "config_gating.json")


def evaluate_model(model, val_dl, config, device, task_indices, loss_func, use_mrl):
    model.eval()
    results, all_losses = {}, {}
    with torch.no_grad():
        for ti in task_indices:
            tmp, tl = [], []
            for xb, yb, idx in tqdm(val_dl, desc=f"Evaluating Task {ti}", leave=False):
                ki = torch.arange(xb.shape[-1])
                ki = ki[~torch.isin(ki, torch.tensor(LABEL_INDICES))]
                mxb = xb[:, :, ki]
                tyb = yb if ti == 0 else xb[:, :, ti]
                tyb[yb == -1] = -1
                tmp.append((mxb, tyb, idx))
                l = loss_batch(
                    model,
                    loss_func,
                    mxb.to(device),
                    tyb.to(device),
                    idx.to(device),
                    use_mrl=use_mrl,
                )
                tl.append(l.item())
            results[ti] = compute_metrics(config.metrics, model, tmp, device)
            all_losses[ti] = np.mean(tl)
    return results, all_losses


def compute_gradient_conflict(model, loss_func, batch, task_indices, device, use_mrl):
    """Compute cosine similarity between task gradient vectors (shared params only)."""
    xb, yb, indices = batch
    ki = torch.arange(xb.shape[-1])
    ki = ki[~torch.isin(ki, torch.tensor(LABEL_INDICES))]
    mxb = xb[:, :, ki].to(device)

    base = model.module if hasattr(model, "module") else model
    shared = list(base.input_layer.parameters()) + list(base.encoder.parameters())

    task_grads = {}
    for ti in task_indices:
        tyb = (yb if ti == 0 else xb[:, :, ti]).clone()
        tyb[yb == -1] = -1
        tyb = tyb.to(device)

        model.zero_grad()
        l = loss_batch(model, loss_func, mxb, tyb, indices.to(device), use_mrl=use_mrl)
        grads = torch.autograd.grad(l, shared, retain_graph=True, allow_unused=True)
        flat = torch.cat(
            [
                g.flatten() if g is not None else torch.zeros(p.numel(), device=device)
                for g, p in zip(grads, shared)
            ]
        )
        task_grads[ti] = flat

    # Pairwise cosine similarities
    keys = list(task_grads.keys())
    conflicts = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            g1, g2 = task_grads[keys[i]], task_grads[keys[j]]
            cos = torch.nn.functional.cosine_similarity(
                g1.unsqueeze(0), g2.unsqueeze(0)
            ).item()
            conflicts[f"{keys[i]}_vs_{keys[j]}"] = cos

    return conflicts, task_grads


def compute_gradient_sparsity(task_grads, threshold=GRAD_SPARSITY_THRESHOLD):
    """Fraction of gradient elements with absolute value < threshold."""
    sparsities = {}
    for ti, g in task_grads.items():
        sparsities[ti] = float((g.abs() < threshold).float().mean().item())
    return sparsities


def finetune_with_scheduler(
    sched_name,
    sched_config,
    model_state_dict,
    model_kwargs,
    config,
    train_dl,
    val_dl,
    task_indices,
    device,
    use_mrl,
    use_gating,
    baseline_results,
    results_dir,
    exp_tag,
    logger,
):
    print(f"\n{'━'*60}")
    print(f"  LR Schedule: {sched_name}")
    print(f"{'━'*60}")

    model = make_model(**model_kwargs)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)

    num_tasks = len(task_indices)
    tw_raw = list(map(int, TASK_WEIGHTS.split(",")))
    wi = (tw_raw[0] + tw_raw[1]) >> 1
    wc = (
        [[x, 1 - x] for x in np.linspace(0.001, 0.991, 10)]
        if num_tasks == 2
        else np.loadtxt("weights-5tasks.txt").tolist()
    )
    task_wt = wc[wi]

    wm = WeightMethods(
        MOO_METHOD, n_tasks=num_tasks, device=device, task_weights=task_wt, epsilon=None
    )

    # Scheduler
    sargs = dict(sched_config["args"])
    if sched_config.get("needs_steps_per_epoch"):
        sargs["steps_per_epoch"] = len(train_dl)
    scheduler = getattr(optim.lr_scheduler, sched_config["class"])(optimizer, **sargs)

    rf = os.path.join(results_dir, f"gradient_{sched_name}_{exp_tag}.txt")
    os.makedirs(os.path.dirname(rf), exist_ok=True)
    if os.path.exists(rf):
        os.remove(rf)

    bl_n10 = baseline_results[task_indices[0]].get("ndcg_10", 0.0)
    with open(rf, "a") as f:
        f.write(f"=== Gradient Dynamics: {sched_name} on {exp_tag} ===\n")
        f.write(f"Baseline NDCG@10: {bl_n10:.6f}\n\n")

    conflict_history, sparsity_history, ndcg_history = [], [], []

    for epoch in tqdm(
        range(FINETUNE_EPOCHS), desc=f"Fine-tuning [{sched_name}]", unit="epoch"
    ):
        model.train()
        batches_for_grad_analysis = []
        sample_interval = max(1, len(train_dl) >> 2)

        for batch_idx, batch in enumerate(
            tqdm(train_dl, desc=f"Epoch {epoch}/{FINETUNE_EPOCHS} [{sched_name}]")
        ):
            xb, yb, indices = batch
            if batch_idx % sample_interval == 0 and len(batches_for_grad_analysis) < 5:
                batches_for_grad_analysis.append(batch)
            ki = torch.arange(xb.shape[-1])
            ki = ki[~torch.isin(ki, torch.tensor(LABEL_INDICES))]
            mxb = xb[:, :, ki]

            bl = []
            for ti in task_indices:
                tyb = yb if ti == 0 else xb[:, :, ti]
                tyb[yb == -1] = -1
                l = loss_batch(
                    model,
                    loss_func,
                    mxb.to(device),
                    tyb.to(device),
                    indices.to(device),
                    use_mrl=use_mrl,
                )
                bl.append(l)

            optimizer.zero_grad()
            wm.backward(
                losses=torch.stack(bl),
                shared_parameters=get_model_parameters(model, "shared_parameters"),
                task_specific_parameters=get_model_parameters(
                    model, "task_specific_parameters"
                ),
                last_shared_parameters=get_model_parameters(
                    model, "last_shared_parameters"
                ),
                task_weights=task_wt,
            )
            optimizer.step()
            if sched_config["class"] == "OneCycleLR":
                scheduler.step()

        if sched_config["class"] != "OneCycleLR":
            scheduler.step()

        # Gradient analysis on sampled batches
        if batches_for_grad_analysis:
            model.train()
            all_c = []
            all_s = []
            for b in batches_for_grad_analysis:
                c, tg = compute_gradient_conflict(
                    model, loss_func, b, task_indices, device, use_mrl
                )
                s = compute_gradient_sparsity(tg)
                all_c.append(c)
                all_s.append(s)

            conflicts = {k: np.mean([x[k] for x in all_c]) for k in all_c[0].keys()}
            sparsity = {k: np.mean([x[k] for x in all_s]) for k in all_s[0].keys()}

            conflict_history.append(conflicts)
            sparsity_history.append(sparsity)
        else:
            conflicts, sparsity = {}, {}

        if use_gating:
            log_gating_sparsity(epoch, model, rf)

        vr, vl = evaluate_model(
            model, val_dl, config, device, task_indices, loss_func, use_mrl
        )
        cn10 = vr[task_indices[0]].get("ndcg_10", 0.0)
        ndcg_history.append(cn10)

        lr = get_current_lr(optimizer)
        with open(rf, "a") as f:
            f.write(f"epoch:{epoch}\tlr:{lr:.2e}\tNDCG@10:{cn10:.6f}\t")
            f.write(f"GradConflict:{conflicts}\tGradSparsity:{sparsity}\t")
            for ti in task_indices:
                f.write(f"task:{ti} {vr[ti]}\t")
            f.write("\n")

        logger.info(
            f"[{sched_name}] Ep {epoch} | NDCG@10={cn10:.6f} | Conflict={conflicts} | Sparsity={sparsity}"
        )

    summary = {
        "scheduler": sched_name,
        "experiment": exp_tag,
        "baseline_ndcg10": bl_n10,
        "peak_ndcg10": max(ndcg_history) if ndcg_history else 0.0,
        "final_ndcg10": ndcg_history[-1] if ndcg_history else 0.0,
        "ndcg10_history": ndcg_history,
        "conflict_history": conflict_history,
        "sparsity_history": [
            {str(k): v for k, v in s.items()} for s in sparsity_history
        ],
        "all_task_final_metrics": {str(ti): vr[ti] for ti in task_indices},
    }

    with open(rf, "a") as f:
        f.write(f"\n{'='*50}\nSUMMARY: {sched_name} on {exp_tag}\n{'='*50}\n")
        f.write(f"Peak NDCG@10:  {summary['peak_ndcg10']:.6f}\n")
        f.write(f"Final NDCG@10: {summary['final_ndcg10']:.6f}\n")
        f.write(
            f"Avg Grad Conflict (last 3): {conflict_history[-3:] if len(conflict_history)>=3 else conflict_history}\n"
        )
        f.write(
            f"Avg Grad Sparsity (last 3): {sparsity_history[-3:] if len(sparsity_history)>=3 else sparsity_history}\n"
        )

    jp = os.path.join(results_dir, f"gradient_{sched_name}_{exp_tag}_summary.json")
    with open(jp, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Results saved to: {rf}")
    return summary


def run_gradient_experiment(
    choice, ckpt, dataset_path, train_dataloader, val_dataloader, n_features
):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if choice == "1":
        exp_tag, cfg_path, use_mrl, use_gating, mrl_dims = (
            "matryoshka",
            CONFIG_MATRYOSHKA,
            True,
            False,
            MRL_NESTING_DIMS,
        )
    elif choice == "2":
        exp_tag, cfg_path, use_mrl, use_gating, mrl_dims = (
            "gating",
            CONFIG_GATING,
            False,
            True,
            None,
        )
    else:
        exp_tag, cfg_path, use_mrl, use_gating, mrl_dims = (
            "baseline",
            CONFIG_GATING,
            False,
            False,
            None,
        )

    config = Config.from_json(cfg_path)
    if use_mrl:
        config.model.use_mrl = True
        config.model.mrl_nesting_dims = mrl_dims

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True

    config.data.path = dataset_path
    config.loss.args["reduction"] = REDUCTION_METHOD
    task_indices = list(map(int, TASK_INDICES.split(",")))

    print(f"\n{'═'*60}")
    print(f"  FINAL EXPERIMENT — Priority 3: Gradient Dynamics & LR")
    print(f"  Primary: {exp_tag.upper()} | Checkpoint: {ckpt}")
    print(f"  Schedulers: {list(SCHEDULER_CONFIGS.keys())}")
    print(f"{'═'*60}\n")

    print("Loading dataset...")

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

    train_ds, val_ds = load_libsvm_dataset(
        config.data.path,
        config.data.slate_length,
        config.data.validation_ds_role,
        max_rows=max_rows,
    )

    nf = train_ds.shape[-1] - len(LABEL_INDICES)
    train_dl, val_dl = create_data_loaders(
        train_ds, val_ds, config.data.num_workers, config.data.batch_size
    )

    mk = dict(n_features=nf, **asdict(config.model, recurse=False))

    print(f"Loading checkpoint: {ckpt}")
    cs = torch.load(ckpt, map_location=device)
    msd = (
        cs["model_state_dict"]
        if isinstance(cs, dict) and "model_state_dict" in cs
        else cs
    )

    print("Evaluating baseline...")
    bm = make_model(**mk)
    bm.load_state_dict(msd)
    bm.to(device)
    lf = partial(getattr(losses, config.loss.name), **config.loss.args)
    br, _ = evaluate_model(bm, val_dl, config, device, task_indices, lf, use_mrl)
    for ti in task_indices:
        print(f"  Task {ti}: {br[ti]}")
    del bm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fold_str = os.path.basename(dataset_path).lower()
    rd = os.path.join(
        "result", "final_experiment", "gradient_dynamics", exp_tag, fold_str
    )
    os.makedirs(rd, exist_ok=True)
    logger = init_logger(rd)

    all_sum = {}
    for sn, sc in tqdm(SCHEDULER_CONFIGS.items(), desc="LR Schedulers", unit="sched"):
        s = finetune_with_scheduler(
            sn,
            sc,
            copy.deepcopy(msd),
            mk,
            config,
            train_dl,
            val_dl,
            task_indices,
            device,
            use_mrl,
            use_gating,
            br,
            rd,
            exp_tag,
            logger,
        )
        if s:
            all_sum[sn] = s
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'═'*70}")
    print(f"  GRADIENT DYNAMICS RESULTS — {exp_tag.upper()}")
    print(f"{'═'*70}")
    print(f"  {'Scheduler':<20} {'Peak NDCG@10':>14} {'Final NDCG@10':>14}")
    print(f"  {'─'*20} {'─'*14} {'─'*14}")
    for n, s in all_sum.items():
        print(f"  {n:<20} {s['peak_ndcg10']:>14.6f} {s['final_ndcg10']:>14.6f}")
    print(f"{'═'*70}\n")

    cp = os.path.join(rd, f"comparison_{exp_tag}.json")
    with open(cp, "w") as f:
        json.dump(all_sum, f, indent=2, default=str)
    print(f"Combined results saved to: {cp}")


def main():
    print("\n" + "═" * 60)
    print("  DeepMTL2R Final Experiment — Priority 3")
    print("  Gradient Dynamics & LR Schedule Analysis")
    print("═" * 60)

    print("\nSelect the PRIMARY experiment checkpoint:")
    print("  1 -> Matryoshka Feature Projection")
    print("  2 -> Dynamic Feature Gating")
    print("  3 -> Baseline (Vanilla Multi-Task)\n")

    while True:
        c = input("Enter your choice (1, 2, or 3): ").strip()
        if c in ("1", "2", "3"):
            break
        print("  Invalid input.")

    for fold in FOLDS:
        print(f"\n" + "=" * 60)
        print(f" Processing Fold {fold}")
        print("=" * 60)
        fold_str = f"fold{fold}"
        dataset_path = os.path.join(DATASET_BASE_PATH, f"Fold{fold}")

        if c == "1":
            ckpt_name = f"matryoshka_{fold_str}_weight0.pkl"
            ckpt_dir = os.path.join(CHECKPOINT_DIR, "matryoshka")
        elif c == "2":
            ckpt_name = f"feature_gating_{fold_str}_weight0.pkl"
            ckpt_dir = os.path.join(CHECKPOINT_DIR, "feature_gating")
        else:
            ckpt_name = f"deepmtl2r_{fold_str}.pkl"
            ckpt_dir = os.path.join(CHECKPOINT_DIR, "deepmtl2r")

        ckpt = os.path.join(ckpt_dir, ckpt_name)

        if not os.path.exists(ckpt):
            print(f"\n[ERROR] Checkpoint not found: {ckpt}")
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

        run_gradient_experiment(c, ckpt, dataset_path, train_dl, val_dl, nf)

        # Free memory at the end of each fold
        del train_ds, val_ds, train_dl, val_dl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()


if __name__ == "__main__":
    main()
