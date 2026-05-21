import os
from functools import partial

import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
import allrank.models.metrics as metrics_module
from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_num_params, log_num_params, get_model_parameters
from allrank.models.losses.matryoshka_loss import MatryoshkaRankingLoss
from allrank.training.early_stop import EarlyStop
from allrank.utils.ltr_logging import get_logger
from allrank.utils.tensorboard_utils import TensorboardSummaryWriter
from itertools import zip_longest
from allrank.methods.weight_methods import WeightMethods
from tqdm import tqdm

logger = get_logger()


def _is_matryoshka_output(output) -> bool:
    """Check whether model output is a Matryoshka tuple (Matryoshka) or a plain tensor (baseline/Feature Gating)."""
    return isinstance(output, (tuple, list))


def loss_batch(model, loss_func, xb, yb, indices, use_mrl: bool = False):
    """
    Calculate loss for a single batch.

    For Matryoshka (Matryoshka): model returns a tuple of score tensors (one per nesting dim).
    The loss_func is expected to be the base ranking loss; MatryoshkaRankingLoss wraps it.

    For baseline / Feature Gating: model returns a plain tensor, loss computed directly.

    :param model: LTRModel instance.
    :param loss_func: base ranking loss callable with signature f(y_pred, y_true).
    :param xb: input features [batch_size, slate_length, n_features].
    :param yb: ground truth labels [batch_size, slate_length].
    :param indices: positional indices [batch_size, slate_length].
    :param use_mrl: if True, wrap loss_func in MatryoshkaRankingLoss.
    :return: scalar loss tensor.
    """
    xb = xb.detach().requires_grad_(True)
    yb = yb.detach().requires_grad_(True)
    mask = (yb == PADDED_Y_VALUE)
    output = model(xb, mask, indices)

    if use_mrl and _is_matryoshka_output(output):
        # Matryoshka: wrap loss with MatryoshkaRankingLoss (uniform weights)
        mrl_loss = MatryoshkaRankingLoss(base_loss_func=loss_func, relative_importance=None)
        return mrl_loss(output, yb)
    else:
        # Baseline or Feature Gating: standard single-output loss
        return loss_func(output, yb)


def metric_on_batch(metric, model, xb, yb, indices):
    """Calculate a ranking metric for a single batch."""
    mask = (yb == PADDED_Y_VALUE)
    return metric(model.score(xb, mask, indices), yb)


def metric_on_epoch(metric, model, dl_single, dev):
    metric_values = torch.mean(
        torch.cat(
            [metric_on_batch(metric, model, xb.to(device=dev), yb_single.to(device=dev), indices.to(device=dev))
             for xb, yb_single, indices in dl_single]
        ), dim=0
    ).cpu().numpy()
    return metric_values


def compute_metrics(metrics, model, dl_single, dev):
    metric_values_dict = {}
    batch_results = {metric_name: [] for metric_name in metrics.keys()}
    
    with torch.no_grad():
        for xb, yb, indices in dl_single:
            xb = xb.to(device=dev)
            yb = yb.to(device=dev)
            indices = indices.to(device=dev)
            mask = (yb == PADDED_Y_VALUE)
            
            # Forward pass ONCE per batch to save GPU memory and prevent fragmentation
            scores = model.score(xb, mask, indices)
            
            for metric_name, ats in metrics.items():
                metric_func = getattr(metrics_module, metric_name)
                metric_ats = ats if ats else None
                metric_func_with_ats = partial(metric_func, ats=metric_ats)
                
                res = metric_func_with_ats(scores, yb)
                batch_results[metric_name].append(res.cpu())
                
            del scores
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for metric_name, ats in metrics.items():
        metric_ats = ats if ats else None
        metrics_values = torch.mean(torch.cat(batch_results[metric_name]), dim=0).numpy()
        
        metrics_names = (
            [metric_name]
            if metric_ats is None
            else ["{metric_name}_{at}".format(metric_name=metric_name, at=at) for at in metric_ats]
        )
        metric_values_dict.update(dict(zip(metrics_names, metrics_values)))
        
    return metric_values_dict


def epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics):
    summary = "Epoch : {epoch} Train loss: {train_loss} Val loss: {val_loss}".format(
        epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss)
    for metric_name, metric_value in train_metrics.items():
        summary += " Train {metric_name} {metric_value}".format(
            metric_name=metric_name, metric_value=metric_value)
    for metric_name, metric_value in val_metrics.items():
        summary += " Val {metric_name} {metric_value}".format(
            metric_name=metric_name, metric_value=metric_value)
    return summary


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def log_metrics(epoch, phase, metrics_dict, loss_values, batch_sizes, task_idx, results_filename):
    """Helper function to log per-task metrics and loss to a results file."""
    with open(results_filename, "a") as file:
        loss_result = np.sum([a * b for a, b in zip(loss_values, batch_sizes)]) / np.sum(batch_sizes)
        file.write(f"epoch:{epoch + 1}\ttask:{task_idx}\t{phase} Loss:{loss_result}\t")
        if metrics_dict:
            file.write(f"{phase} Metrics:{metrics_dict}\t")
        file.write("\n")


def log_gating_sparsity(epoch, model, results_filename, threshold: float = 0.1):
    """
    Log the gating sparsity ratio for Dynamic Feature Gating.
    Writes the fraction of input features suppressed by the gate.

    :param epoch: current epoch number.
    :param model: LTRModel with DynamicFeatureGate inside FCModel.
    :param results_filename: path to log file.
    :param threshold: gate value threshold below which a feature is considered suppressed.
    """
    base_model = model.module if hasattr(model, 'module') else model
    input_layer = base_model.input_layer

    if hasattr(input_layer, 'gating_layer') and input_layer.gating_layer is not None:
        sparsity = input_layer.gating_layer.get_sparsity_ratio(threshold=threshold)
        with open(results_filename, "a") as file:
            file.write(f"epoch:{epoch + 1}\tGating Sparsity Ratio (threshold={threshold}): {sparsity:.4f}\n")
        logger.info(f"Epoch {epoch + 1} | Gating Sparsity Ratio: {sparsity:.4f}")


def compute_mrl_dimensionality_efficiency(model, dl_single, dev, metrics, mrl_nesting_dims):
    """
    Compute Effective Dimensionality Efficiency for Matryoshka Feature Projection.
    Evaluates NDCG at each nesting dimension level by temporarily swapping
    the evaluation head to the appropriate nesting head.

    :param model: LTRModel with MatryoshkaOutputLayer.
    :param dl_single: dataloader yielding (xb, yb, indices).
    :param dev: torch device.
    :param metrics: dict of metric names and at-values from config.
    :param mrl_nesting_dims: list of nesting dimensions, e.g. [32, 64, 128, 256].
    :return: dict mapping nesting_dim -> metrics_dict.
    """
    base_model = model.module if hasattr(model, 'module') else model
    output_layer = base_model.output_layer

    if not hasattr(output_layer, 'nesting_dims'):
        return {}

    efficiency_results = {}
    original_score = output_layer.score  # save original

    for i, dim in enumerate(output_layer.nesting_dims):
        # Temporarily override score to use head i
        head = getattr(output_layer, f"nesting_head_{i}")

        def make_score_fn(h, d, act):
            def score_fn(x):
                return act(h(x[:, :, :d])).squeeze(dim=-1)
            return score_fn

        output_layer.score = make_score_fn(head, dim, output_layer.activation)
        metrics_dict = compute_metrics(metrics, base_model, dl_single, dev)
        efficiency_results[dim] = metrics_dict

    # Restore original score function
    output_layer.score = original_score
    return efficiency_results


def compute_noise_robustness(model, dl_single, dev, metrics, label_indices, noise_fraction=0.2, noise_std=1.0):
    """
    Compute robustness to noisy features by injecting Gaussian noise into a fraction of features
    and measuring the drop in NDCG.
    
    :param model: LTRModel instance.
    :param dl_single: dataloader yielding (xb, yb, indices).
    :param dev: torch device.
    :param metrics: dict of metric names and at-values from config.
    :param label_indices: indices of label columns to exclude from feature noise.
    :param noise_fraction: fraction of features to inject noise into.
    :param noise_std: standard deviation of the Gaussian noise.
    :return: dict of metric drops (original - noisy).
    """
    # First, get baseline metrics on clean data
    baseline_metrics = compute_metrics(metrics, model, dl_single, dev)
    
    # Create a noisy dataloader
    noisy_dl = []
    for xb, yb, indices in dl_single:
        noisy_xb = xb.clone()
        n_features = xb.shape[-1]
        
        # Determine features to corrupt (excluding labels)
        all_feats = list(range(n_features))
        valid_feats = [f for f in all_feats if f not in label_indices]
        
        num_noisy_feats = int(len(valid_feats) * noise_fraction)
        if num_noisy_feats > 0:
            feats_to_corrupt = np.random.choice(valid_feats, num_noisy_feats, replace=False)
            noise = torch.randn(noisy_xb[:, :, feats_to_corrupt].shape) * noise_std
            noisy_xb[:, :, feats_to_corrupt] += noise.to(noisy_xb.device)
            
        noisy_dl.append((noisy_xb, yb, indices))
        
    # Get metrics on noisy data
    noisy_metrics = compute_metrics(metrics, model, noisy_dl, dev)
    
    # Calculate robustness (drop in performance)
    robustness_results = {}
    for k in baseline_metrics:
        # We record both the absolute drop and relative drop
        drop = baseline_metrics[k] - noisy_metrics[k]
        rel_drop = (drop / baseline_metrics[k]) * 100 if baseline_metrics[k] > 0 else 0
        robustness_results[f"{k}_drop"] = drop
        robustness_results[f"{k}_rel_drop_pct"] = rel_drop
        
    return robustness_results


def fit(epochs, moo_method, main_task_index, task_indices, label_indices,
        results_filename, model, loss_func,
        task_weights, optimizer, scheduler,
        train_dataloader, val_dataloader, config,
        gradient_clipping_norm, early_stopping_patience,
        device, output_dir, tensorboard_output_path,
        epsilon=None, compute_delta_m=False, stl_delta_m=None,
        use_mrl: bool = False, use_gating: bool = False,
        mrl_nesting_dims=None):
    """
    Main training loop for multi-task learning with ranking tasks.

    Extended for Matryoshka Feature Projection and Dynamic Feature Gating.

    Key additions:
    - Matryoshka loss wrapping via use_mrl flag.
    - Gating sparsity logging via use_gating flag.
    - Model checkpoint saving for all epochs.
    - Dimensionality efficiency logging per nesting dim (Matryoshka, logged at end).

    :param use_mrl: if True, use MatryoshkaRankingLoss (Matryoshka).
    :param use_gating: if True, log gating sparsity each epoch (Feature Gating).
    :param mrl_nesting_dims: nesting dimensions for Matryoshka logging (Matryoshka).
    """
    tensorboard_writer = TensorboardSummaryWriter(tensorboard_output_path)
    early_stop = EarlyStop(early_stopping_patience)
    weight_method = WeightMethods(moo_method,
                                  n_tasks=len(task_indices),
                                  device=device,
                                  task_weights=task_weights,
                                  epsilon=epsilon)

    epoch = -1
    train_metrics = {}
    valid_metrics = {}

    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch + 1}/{epochs + 1}, Current learning rate: {get_current_lr(optimizer)}")
        model.train()

        train_loss_values = {task_idx: [] for task_idx in task_indices}
        train_nums = []

        for batch_id, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs + 1} [Train]")):
            xb, yb, indices = batch
            all_indices = torch.arange(xb.shape[-1])
            keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(label_indices))]
            modified_xb = xb[:, :, keep_indices]

            losses = []
            for task_idx in task_indices:
                task_yb = yb if task_idx == 0 else xb[:, :, task_idx]
                task_yb[yb == -1] = -1
                loss = loss_batch(
                    model, loss_func,
                    modified_xb.to(device), task_yb.to(device), indices.to(device),
                    use_mrl=use_mrl
                )
                losses.append(loss)
                train_loss_values[task_idx].append(loss.item())
            train_nums.append(len(xb))

            if optimizer:
                optimizer.zero_grad()
                loss_weighted_sum, _ = weight_method.backward(
                    losses=torch.stack(losses),
                    shared_parameters=get_model_parameters(model, 'shared_parameters'),
                    task_specific_parameters=get_model_parameters(model, 'task_specific_parameters'),
                    last_shared_parameters=get_model_parameters(model, 'last_shared_parameters'),
                    task_weights=task_weights
                )
                if gradient_clipping_norm:
                    clip_grad_norm_(model.parameters(), gradient_clipping_norm)
                optimizer.step()

        # --- Feature Gating: Log gating sparsity each epoch ---
        if use_gating:
            log_gating_sparsity(epoch, model, results_filename)

        # Log training metrics
        train_result = {}
        for task_idx in task_indices:
            temp_dl = []
            for xb, yb, indices in train_dataloader:
                all_indices = torch.arange(xb.shape[-1])
                keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(label_indices))]
                modified_xb = xb[:, :, keep_indices]
                task_yb = yb if task_idx == 0 else xb[:, :, task_idx]
                task_yb[yb == -1] = -1
                temp_dl.append((modified_xb, task_yb, indices))

            train_metrics = compute_metrics(config.metrics, model, temp_dl, device)
            train_result[task_idx] = train_metrics
            log_metrics(epoch, "Train", train_metrics,
                        train_loss_values[task_idx], train_nums, task_idx, results_filename)

        # Validation loop
        model.eval()
        with torch.no_grad():
            valid_loss_values = {task_idx: [] for task_idx in task_indices}
            valid_nums = []

            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs + 1} [Valid]", leave=False):
                xb, yb, indices = batch
                all_indices = torch.arange(xb.shape[-1])
                keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(label_indices))]
                modified_xb = xb[:, :, keep_indices]

                for task_idx in task_indices:
                    task_yb = yb if task_idx == 0 else xb[:, :, task_idx]
                    task_yb[yb == -1] = -1
                    loss = loss_batch(
                        model, loss_func,
                        modified_xb.to(device), task_yb.to(device), indices.to(device),
                        use_mrl=use_mrl
                    )
                    valid_loss_values[task_idx].append(loss.item())
                valid_nums.append(len(xb))

            valid_result = {}
            for task_idx in task_indices:
                temp_dl = []
                for xb, yb, indices in val_dataloader:
                    all_indices = torch.arange(xb.shape[-1])
                    keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(label_indices))]
                    modified_xb = xb[:, :, keep_indices]
                    task_yb = yb if task_idx == 0 else xb[:, :, task_idx]
                    task_yb[yb == -1] = -1
                    temp_dl.append((modified_xb, task_yb, indices))

                valid_metrics = compute_metrics(config.metrics, model, temp_dl, device)
                valid_result[task_idx] = valid_metrics
                log_metrics(epoch, "Valid", valid_metrics,
                            valid_loss_values[task_idx], valid_nums, task_idx, results_filename)

            if compute_delta_m:
                metric_name = 'get_deltam'
                metric_func = getattr(metrics_module, metric_name)
                delta_m = metric_func(valid_result, stl_delta_m)

        # --- Scheduler and early stopping ---
        current_val_metric = valid_result[task_indices[0]].get(config.val_metric)
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_val_metric)
            else:
                scheduler.step()

        # --- Save model checkpoint for the current epoch ---
        epoch_checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch + 1}.pkl")
        torch.save(model.state_dict(), epoch_checkpoint_path)
        logger.info(
            f"Epoch {epoch + 1} | Model saved to {epoch_checkpoint_path}"
        )

        early_stop.step(current_val_metric, epoch + 1)
        if early_stop.stop_training(epoch + 1):
            logger.info(
                "early stopping at epoch {} since {} didn't improve from epoch no {}. "
                "Best value {}, current value {}".format(
                    epoch + 1, config.val_metric, early_stop.best_epoch,
                    early_stop.best_value, current_val_metric
                ))
            break

    # --- Save final (last-epoch) model ---
    final_checkpoint_path = os.path.join(output_dir, "model.pkl")
    torch.save(model.state_dict(), final_checkpoint_path)
    logger.info(f"Final model saved to {final_checkpoint_path}")

    # --- Matryoshka: Log Effective Dimensionality Efficiency at end of training ---
    special_metrics = {}
    base_model = model.module if hasattr(model, 'module') else model

    if use_gating:
        input_layer = getattr(base_model, 'input_layer', None)
        gating_layer = getattr(input_layer, 'gating_layer', None) if input_layer is not None else None
        if gating_layer is not None:
            special_metrics['gating_sparsity_ratio'] = gating_layer.get_sparsity_ratio()

    if use_mrl and mrl_nesting_dims:
        logger.info("Computing Effective Dimensionality Efficiency (Matryoshka)...")
        model.eval()
        with torch.no_grad():
            for task_idx in task_indices:
                temp_dl = []
                for xb, yb, indices in val_dataloader:
                    all_indices = torch.arange(xb.shape[-1])
                    keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(label_indices))]
                    modified_xb = xb[:, :, keep_indices]
                    task_yb = yb if task_idx == 0 else xb[:, :, task_idx]
                    task_yb[yb == -1] = -1
                    temp_dl.append((modified_xb, task_yb, indices))

                efficiency = compute_mrl_dimensionality_efficiency(
                    model, temp_dl, device, config.metrics, mrl_nesting_dims
                )
                with open(results_filename, "a", encoding="utf-8") as f:
                    f.write(f"\n=== Effective Dimensionality Efficiency (task {task_idx}) ===\n")
                    for dim, dim_metrics in efficiency.items():
                        f.write(f"  dim={dim}:\n")
                        for metric_name, metric_val in dim_metrics.items():
                            f.write(f"    {metric_name}: {metric_val}\n")
                # Pretty print to logger for console readability
                import json
                efficiency_str = json.dumps(efficiency, indent=2, default=str)
                logger.info(f"Task {task_idx} Dimensionality Efficiency:\n{efficiency_str}")
                special_metrics.setdefault('mrl_dimensionality_efficiency', {})[task_idx] = efficiency

    # --- Phase 1: Robustness to Noisy Features ---
    logger.info("Computing robustness to noisy features (all tasks)...")
    noise_robustness_results = {}
    with torch.no_grad():
        for task_idx in task_indices:
            robustness_dl = []
            for xb, yb, indices in val_dataloader:
                all_indices = torch.arange(xb.shape[-1])
                keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(label_indices))]
                modified_xb = xb[:, :, keep_indices]
                task_yb = yb if task_idx == 0 else xb[:, :, task_idx]
                task_yb[yb == -1] = -1
                robustness_dl.append((modified_xb, task_yb, indices))

            noise_robustness_results[task_idx] = compute_noise_robustness(
                base_model,
                robustness_dl,
                device,
                config.metrics,
                label_indices,
            )
            
    special_metrics['noise_robustness'] = noise_robustness_results

    tensorboard_writer.close_all_writers()

    return {
        "epochs": epoch,
        "train_metrics": train_result,
        "val_metrics": valid_result,
        "num_params": get_num_params(model),
        "special_metrics": special_metrics,
    }
