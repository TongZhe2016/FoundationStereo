from typing import *
import fnmatch

import sympy
import torch
import torch.nn as nn
import numpy as np


def any_match(s: str, patterns: List[str]) -> bool:
    """Check if string matches any of the given patterns"""
    return any(fnmatch.fnmatch(s, pat) for pat in patterns)


def build_optimizer(model: nn.Module, optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Build optimizer with parameter groups based on configuration
    
    Args:
        model: PyTorch model
        optimizer_config: Configuration dictionary with optimizer settings
    """
    named_param_groups = [
        {
            k: p for k, p in model.named_parameters() 
            if any_match(k, param_group_config['params']['include']) 
            and not any_match(k, param_group_config['params'].get('exclude', []))
        } for param_group_config in optimizer_config['params']
    ]
    
    excluded_params = [
        k for k, p in model.named_parameters() 
        if p.requires_grad and not any(k in named_params for named_params in named_param_groups)
    ]
    assert len(excluded_params) == 0, f'The following parameters require grad but are excluded from the optimizer: {excluded_params}'
    
    optimizer_cls = getattr(torch.optim, optimizer_config['type'])
    optimizer = optimizer_cls([
        {
            **param_group_config,
            'params': list(params.values()), 
        } for param_group_config, params in zip(optimizer_config['params'], named_param_groups)
    ])
    return optimizer


def parse_lr_lambda(s: str) -> Callable[[int], float]:
    """Parse learning rate lambda function from string"""
    epoch = sympy.symbols('epoch')
    lr_lambda = sympy.sympify(s)
    return sympy.lambdify(epoch, lr_lambda, 'math')


def build_lr_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build learning rate scheduler based on configuration
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_config: Configuration dictionary with scheduler settings
    """
    if scheduler_config['type'] == "SequentialLR":
        child_schedulers = [
            build_lr_scheduler(optimizer, child_scheduler_config)
                for child_scheduler_config in scheduler_config['params']['schedulers']
        ]
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=child_schedulers, 
            milestones=scheduler_config['params']['milestones']
        )
    elif scheduler_config['type'] == "LambdaLR":
        lr_lambda = scheduler_config['params']['lr_lambda']
        if isinstance(lr_lambda, str):
            lr_lambda = parse_lr_lambda(lr_lambda)
        elif isinstance(lr_lambda, list):
            lr_lambda = [parse_lr_lambda(l) for l in lr_lambda]
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )
    else:
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config['type'])
        scheduler = scheduler_cls(optimizer, **scheduler_config.get('params', {}))
    return scheduler


def compute_stereo_metrics(
    pred_disparity: torch.Tensor,
    gt_disparity: torch.Tensor,
    mask: torch.Tensor,
    thresholds: List[float] = [1.0, 3.0, 5.0]
) -> Dict[str, float]:
    """
    Compute comprehensive stereo evaluation metrics
    
    Args:
        pred_disparity: Predicted disparity [B, H, W] or [H, W]
        gt_disparity: Ground truth disparity [B, H, W] or [H, W]
        mask: Valid disparity mask [B, H, W] or [H, W]
        thresholds: Error thresholds for computing error rates
    
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # Handle batch dimension
        if pred_disparity.dim() == 3:
            batch_size = pred_disparity.shape[0]
            metrics = {}
            for i in range(batch_size):
                batch_metrics = compute_stereo_metrics(
                    pred_disparity[i], gt_disparity[i], mask[i], thresholds
                )
                for k, v in batch_metrics.items():
                    if k not in metrics:
                        metrics[k] = []
                    metrics[k].append(v)
            # Average across batch
            return {k: np.mean(v) for k, v in metrics.items()}
        
        if mask.sum() == 0:
            return {f'd{int(t)}_error': 0.0 for t in thresholds} | {'epe': 0.0, 'rmse': 0.0, 'mae': 0.0}
        
        # Compute absolute error
        abs_error = torch.abs(pred_disparity - gt_disparity)[mask]
        
        # End-point error (mean absolute error)
        epe = abs_error.mean().item()
        mae = epe  # Same as EPE for disparity
        
        # Root mean square error
        rmse = torch.sqrt((abs_error ** 2).mean()).item()
        
        # Error rates at different thresholds
        metrics = {'epe': epe, 'rmse': rmse, 'mae': mae}
        for threshold in thresholds:
            error_rate = (abs_error > threshold).float().mean().item()
            metrics[f'd{int(threshold)}_error'] = error_rate
        
        return metrics


def setup_parameter_groups(model: nn.Module, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Setup parameter groups for different parts of the stereo model
    
    Args:
        model: FoundationStereo model
        config: Configuration for parameter groups
    
    Returns:
        List of parameter group dictionaries
    """
    param_groups = []
    
    # Default parameter group
    all_params = set(model.parameters())
    grouped_params = set()
    
    # Create specific parameter groups
    for group_config in config.get('param_groups', []):
        group_params = []
        for name, param in model.named_parameters():
            if any_match(name, group_config['include']) and not any_match(name, group_config.get('exclude', [])):
                group_params.append(param)
                grouped_params.add(param)
        
        if group_params:
            param_group = {
                'params': group_params,
                'lr': group_config.get('lr', config.get('lr', 1e-4)),
                'weight_decay': group_config.get('weight_decay', config.get('weight_decay', 0.0)),
            }
            # Add any additional parameters from config
            for key, value in group_config.items():
                if key not in ['include', 'exclude', 'params']:
                    param_group[key] = value
            
            param_groups.append(param_group)
    
    # Add remaining parameters to default group
    remaining_params = [p for p in all_params if p not in grouped_params and p.requires_grad]
    if remaining_params:
        default_group = {
            'params': remaining_params,
            'lr': config.get('lr', 1e-4),
            'weight_decay': config.get('weight_decay', 0.0),
        }
        param_groups.append(default_group)
    
    return param_groups


def freeze_parameters(model: nn.Module, patterns: List[str]):
    """
    Freeze parameters matching the given patterns
    
    Args:
        model: PyTorch model
        patterns: List of parameter name patterns to freeze
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if any_match(name, patterns):
            param.requires_grad = False
            frozen_count += 1
    
    print(f"Frozen {frozen_count} parameters matching patterns: {patterns}")


def unfreeze_parameters(model: nn.Module, patterns: List[str]):
    """
    Unfreeze parameters matching the given patterns
    
    Args:
        model: PyTorch model
        patterns: List of parameter name patterns to unfreeze
    """
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if any_match(name, patterns):
            param.requires_grad = True
            unfrozen_count += 1
    
    print(f"Unfrozen {unfrozen_count} parameters matching patterns: {patterns}")


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    return optimizer.param_groups[0]['lr']


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float):
    """Set learning rate for all parameter groups"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    base_lr: float,
    current_step: int
) -> float:
    """
    Linear warmup learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        base_lr: Base learning rate after warmup
        current_step: Current training step
    
    Returns:
        Current learning rate
    """
    if current_step < warmup_steps:
        lr = base_lr * (current_step + 1) / warmup_steps
    else:
        lr = base_lr
    
    set_learning_rate(optimizer, lr)
    return lr


def cosine_annealing_lr(
    base_lr: float,
    min_lr: float,
    current_step: int,
    total_steps: int,
    warmup_steps: int = 0
) -> float:
    """
    Cosine annealing learning rate schedule with optional warmup
    
    Args:
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        current_step: Current training step
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
    
    Returns:
        Current learning rate
    """
    if current_step < warmup_steps:
        return base_lr * (current_step + 1) / warmup_steps
    
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)
    
    lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
    return lr


def polynomial_lr_decay(
    base_lr: float,
    current_step: int,
    total_steps: int,
    power: float = 0.9,
    min_lr: float = 0.0
) -> float:
    """
    Polynomial learning rate decay
    
    Args:
        base_lr: Base learning rate
        current_step: Current training step
        total_steps: Total training steps
        power: Polynomial power
        min_lr: Minimum learning rate
    
    Returns:
        Current learning rate
    """
    progress = min(current_step / total_steps, 1.0)
    lr = (base_lr - min_lr) * (1 - progress) ** power + min_lr
    return lr


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module):
    """Print a summary of the model parameters"""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"Model Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Print parameter groups
    param_groups = {}
    for name, param in model.named_parameters():
        module_name = '.'.join(name.split('.')[:-1]) if '.' in name else 'root'
        if module_name not in param_groups:
            param_groups[module_name] = {'total': 0, 'trainable': 0}
        param_groups[module_name]['total'] += param.numel()
        if param.requires_grad:
            param_groups[module_name]['trainable'] += param.numel()
    
    print(f"\nParameter breakdown by module:")
    for module_name, counts in sorted(param_groups.items()):
        print(f"  {module_name}: {counts['trainable']:,} / {counts['total']:,} trainable")