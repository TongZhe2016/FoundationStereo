from typing import *
import math

import torch
import torch.nn.functional as F
import numpy as np


def _smooth_l1_loss(input: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Smooth L1 loss as defined in Fast R-CNN
    """
    diff = torch.abs(input - target)
    if beta == 0:
        return diff
    else:
        return torch.where(diff < beta, 0.5 * diff.square() / beta, diff - 0.5 * beta)


def disparity_l1_loss(
    pred_disparity: torch.Tensor,
    gt_disparity: torch.Tensor,
    mask: torch.Tensor,
    max_disparity: float = 192.0,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    L1 loss for disparity estimation
    
    Args:
        pred_disparity: Predicted disparity [H, W]
        gt_disparity: Ground truth disparity [H, W]
        mask: Valid disparity mask [H, W]
        max_disparity: Maximum disparity value for clamping
    """
    # Handle resolution mismatch by resizing prediction to match ground truth
    if pred_disparity.shape != gt_disparity.shape:
        # Ensure pred_disparity has batch and channel dimensions for interpolation
        if pred_disparity.dim() == 2:
            pred_disparity = pred_disparity.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
        elif pred_disparity.dim() == 3:
            pred_disparity = pred_disparity.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        
        pred_disparity = F.interpolate(
            pred_disparity,
            size=gt_disparity.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Squeeze back to match gt_disparity dimensions
        while pred_disparity.dim() > gt_disparity.dim():
            pred_disparity = pred_disparity.squeeze(0)
    
    # Clamp predictions to valid range
    pred_disparity = torch.clamp(pred_disparity, 0, max_disparity)
    
    # Compute L1 loss only on valid pixels
    diff = torch.abs(pred_disparity - gt_disparity)
    loss = diff[mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=pred_disparity.device)
    
    # Compute metrics
    with torch.no_grad():
        if mask.sum() > 0:
            epe = diff[mask].mean().item()
            d1_error = (diff[mask] > 3.0).float().mean().item()
            d3_error = (diff[mask] > 1.0).float().mean().item()
        else:
            epe = 0.0
            d1_error = 0.0
            d3_error = 0.0
    
    misc = {
        'epe': epe,
        'd1_error': d1_error,
        'd3_error': d3_error,
    }
    
    return loss, misc


def disparity_smooth_l1_loss(
    pred_disparity: torch.Tensor,
    gt_disparity: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
    max_disparity: float = 192.0,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Smooth L1 loss for disparity estimation
    
    Args:
        pred_disparity: Predicted disparity [H, W]
        gt_disparity: Ground truth disparity [H, W]
        mask: Valid disparity mask [H, W]
        beta: Smooth L1 beta parameter
        max_disparity: Maximum disparity value for clamping
    """
    # Handle resolution mismatch by resizing prediction to match ground truth
    if pred_disparity.shape != gt_disparity.shape:
        # Ensure pred_disparity has batch and channel dimensions for interpolation
        if pred_disparity.dim() == 2:
            pred_disparity = pred_disparity.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
        elif pred_disparity.dim() == 3:
            pred_disparity = pred_disparity.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        
        pred_disparity = F.interpolate(
            pred_disparity,
            size=gt_disparity.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Squeeze back to match gt_disparity dimensions
        while pred_disparity.dim() > gt_disparity.dim():
            pred_disparity = pred_disparity.squeeze(0)
    
    # Clamp predictions to valid range
    pred_disparity = torch.clamp(pred_disparity, 0, max_disparity)
    
    # Compute smooth L1 loss only on valid pixels
    smooth_l1 = _smooth_l1_loss(pred_disparity, gt_disparity, beta)
    loss = smooth_l1[mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=pred_disparity.device)
    
    # Compute metrics
    with torch.no_grad():
        if mask.sum() > 0:
            diff = torch.abs(pred_disparity - gt_disparity)
            epe = diff[mask].mean().item()
            d1_error = (diff[mask] > 3.0).float().mean().item()
            d3_error = (diff[mask] > 1.0).float().mean().item()
        else:
            epe = 0.0
            d1_error = 0.0
            d3_error = 0.0
    
    misc = {
        'epe': epe,
        'd1_error': d1_error,
        'd3_error': d3_error,
    }
    
    return loss, misc


def disparity_epe_loss(
    pred_disparity: torch.Tensor,
    gt_disparity: torch.Tensor,
    mask: torch.Tensor,
    max_disparity: float = 192.0,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    End-Point-Error (EPE) loss for disparity estimation
    
    Args:
        pred_disparity: Predicted disparity [H, W]
        gt_disparity: Ground truth disparity [H, W]
        mask: Valid disparity mask [H, W]
        max_disparity: Maximum disparity value for clamping
    """
    # Clamp predictions to valid range
    pred_disparity = torch.clamp(pred_disparity, 0, max_disparity)
    
    # Compute EPE loss (same as L1 for disparity)
    diff = torch.abs(pred_disparity - gt_disparity)
    loss = diff[mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=pred_disparity.device)
    
    # Compute metrics
    with torch.no_grad():
        if mask.sum() > 0:
            epe = diff[mask].mean().item()
            d1_error = (diff[mask] > 3.0).float().mean().item()
            d3_error = (diff[mask] > 1.0).float().mean().item()
        else:
            epe = 0.0
            d1_error = 0.0
            d3_error = 0.0
    
    misc = {
        'epe': epe,
        'd1_error': d1_error,
        'd3_error': d3_error,
    }
    
    return loss, misc


def multi_scale_loss(
    pred_disparity_pyramid: List[torch.Tensor],
    gt_disparity: torch.Tensor,
    mask: torch.Tensor,
    weights: List[float] = None,
    loss_type: str = 'smooth_l1',
    beta: float = 1.0,
    max_disparity: float = 192.0,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Multi-scale loss for disparity pyramid
    
    Args:
        pred_disparity_pyramid: List of predicted disparities at different scales
        gt_disparity: Ground truth disparity [H, W]
        mask: Valid disparity mask [H, W]
        weights: Weights for each scale (default: equal weights)
        loss_type: Type of loss ('l1', 'smooth_l1', 'epe')
        beta: Smooth L1 beta parameter
        max_disparity: Maximum disparity value for clamping
    """
    if weights is None:
        weights = [1.0] * len(pred_disparity_pyramid)
    
    if len(weights) != len(pred_disparity_pyramid):
        raise ValueError(f"Number of weights ({len(weights)}) must match pyramid levels ({len(pred_disparity_pyramid)})")
    
    total_loss = torch.tensor(0.0, device=gt_disparity.device)
    total_epe = 0.0
    total_d1_error = 0.0
    total_d3_error = 0.0
    
    for i, (pred_disp, weight) in enumerate(zip(pred_disparity_pyramid, weights)):
        # Ensure pred_disp has the right dimensions
        while pred_disp.dim() > gt_disparity.dim():
            pred_disp = pred_disp.squeeze(0)
        
        # Resize ground truth to match prediction scale
        scale_factor = pred_disp.shape[-1] / gt_disparity.shape[-1]
        if scale_factor != 1.0:
            gt_disp_scaled = F.interpolate(
                gt_disparity.unsqueeze(0).unsqueeze(0),
                size=pred_disp.shape[-2:],
                mode='nearest'
            ).squeeze(0).squeeze(0) * scale_factor
            
            mask_scaled = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=pred_disp.shape[-2:],
                mode='nearest'
            ).squeeze(0).squeeze(0) > 0.5
        else:
            gt_disp_scaled = gt_disparity
            mask_scaled = mask
        
        # Clamp predictions
        pred_disp = torch.clamp(pred_disp, 0, max_disparity * scale_factor)
        
        # Compute loss for this scale
        if loss_type == 'l1':
            diff = torch.abs(pred_disp - gt_disp_scaled)
            scale_loss = diff[mask_scaled].mean() if mask_scaled.sum() > 0 else torch.tensor(0.0, device=pred_disp.device)
        elif loss_type == 'smooth_l1':
            smooth_l1 = _smooth_l1_loss(pred_disp, gt_disp_scaled, beta)
            scale_loss = smooth_l1[mask_scaled].mean() if mask_scaled.sum() > 0 else torch.tensor(0.0, device=pred_disp.device)
        elif loss_type == 'epe':
            diff = torch.abs(pred_disp - gt_disp_scaled)
            scale_loss = diff[mask_scaled].mean() if mask_scaled.sum() > 0 else torch.tensor(0.0, device=pred_disp.device)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        total_loss += weight * scale_loss
        
        # Compute metrics for this scale
        with torch.no_grad():
            if mask_scaled.sum() > 0:
                diff = torch.abs(pred_disp - gt_disp_scaled)
                total_epe += diff[mask_scaled].mean().item() * weight
                total_d1_error += (diff[mask_scaled] > 3.0 * scale_factor).float().mean().item() * weight
                total_d3_error += (diff[mask_scaled] > 1.0 * scale_factor).float().mean().item() * weight
    
    # Normalize by total weight
    total_weight = sum(weights)
    total_epe /= total_weight
    total_d1_error /= total_weight
    total_d3_error /= total_weight
    
    misc = {
        'multi_scale_epe': total_epe,
        'multi_scale_d1_error': total_d1_error,
        'multi_scale_d3_error': total_d3_error,
    }
    
    return total_loss, misc


def gradient_loss(
    pred_disparity: torch.Tensor,
    gt_disparity: torch.Tensor,
    mask: torch.Tensor,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Gradient loss for disparity smoothness
    
    Args:
        pred_disparity: Predicted disparity [H, W]
        gt_disparity: Ground truth disparity [H, W]
        mask: Valid disparity mask [H, W]
    """
    # Compute gradients
    pred_grad_x = torch.abs(pred_disparity[:, 1:] - pred_disparity[:, :-1])
    pred_grad_y = torch.abs(pred_disparity[1:, :] - pred_disparity[:-1, :])
    
    gt_grad_x = torch.abs(gt_disparity[:, 1:] - gt_disparity[:, :-1])
    gt_grad_y = torch.abs(gt_disparity[1:, :] - gt_disparity[:-1, :])
    
    # Compute masks for gradients
    mask_grad_x = mask[:, 1:] & mask[:, :-1]
    mask_grad_y = mask[1:, :] & mask[:-1, :]
    
    # Compute gradient loss
    loss_x = torch.abs(pred_grad_x - gt_grad_x)[mask_grad_x].mean() if mask_grad_x.sum() > 0 else torch.tensor(0.0, device=pred_disparity.device)
    loss_y = torch.abs(pred_grad_y - gt_grad_y)[mask_grad_y].mean() if mask_grad_y.sum() > 0 else torch.tensor(0.0, device=pred_disparity.device)
    
    loss = (loss_x + loss_y) / 2
    
    misc = {
        'gradient_loss_x': loss_x.item(),
        'gradient_loss_y': loss_y.item(),
    }
    
    return loss, misc


def monitoring_stereo(disparity: torch.Tensor) -> Dict[str, float]:
    """
    Monitor stereo prediction statistics
    
    Args:
        disparity: Predicted disparity [H, W]
    """
    with torch.no_grad():
        return {
            'disparity_mean': disparity.mean().item(),
            'disparity_std': disparity.std().item(),
            'disparity_min': disparity.min().item(),
            'disparity_max': disparity.max().item(),
        }


def compute_stereo_metrics(
    pred_disparity: torch.Tensor,
    gt_disparity: torch.Tensor,
    mask: torch.Tensor,
    thresholds: List[float] = [1.0, 3.0, 5.0]
) -> Dict[str, float]:
    """
    Compute comprehensive stereo metrics
    
    Args:
        pred_disparity: Predicted disparity [H, W]
        gt_disparity: Ground truth disparity [H, W]
        mask: Valid disparity mask [H, W]
        thresholds: Error thresholds for computing error rates
    """
    with torch.no_grad():
        if mask.sum() == 0:
            return {f'd{int(t)}_error': 0.0 for t in thresholds} | {'epe': 0.0, 'rmse': 0.0}
        
        # Compute absolute error
        abs_error = torch.abs(pred_disparity - gt_disparity)[mask]
        
        # End-point error (mean absolute error)
        epe = abs_error.mean().item()
        
        # Root mean square error
        rmse = torch.sqrt((abs_error ** 2).mean()).item()
        
        # Error rates at different thresholds
        metrics = {'epe': epe, 'rmse': rmse}
        for threshold in thresholds:
            error_rate = (abs_error > threshold).float().mean().item()
            metrics[f'd{int(threshold)}_error'] = error_rate
        
        return metrics


def foundation_stereo_loss(
    pred_disparity_initial: torch.Tensor,
    pred_disparity_pyramid: List[torch.Tensor],
    gt_disparity: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 0.9,
    max_disparity: float = 192.0,
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    FoundationStereo loss function as described in the paper:
    L = |d₀ - d̄|_smooth + Σ(k=1 to K) γ^(K-k) ||d_k - d̄||₁
    
    Args:
        pred_disparity_initial: Initial disparity prediction d₀ [H, W]
        pred_disparity_pyramid: List of refined disparities [d₁, d₂, ..., d_K]
        gt_disparity: Ground truth disparity d̄ [H, W]
        mask: Valid disparity mask [H, W]
        gamma: Exponential weight factor (default: 0.9)
        max_disparity: Maximum disparity value for clamping
    """
    # Handle resolution mismatch for initial disparity
    if pred_disparity_initial.shape != gt_disparity.shape:
        if pred_disparity_initial.dim() == 2:
            pred_disparity_initial = pred_disparity_initial.unsqueeze(0).unsqueeze(0)
        elif pred_disparity_initial.dim() == 3:
            pred_disparity_initial = pred_disparity_initial.unsqueeze(0)
        
        pred_disparity_initial = F.interpolate(
            pred_disparity_initial,
            size=gt_disparity.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        while pred_disparity_initial.dim() > gt_disparity.dim():
            pred_disparity_initial = pred_disparity_initial.squeeze(0)
    
    # Clamp initial prediction
    pred_disparity_initial = torch.clamp(pred_disparity_initial, 0, max_disparity)
    
    # Compute smooth L1 loss for initial disparity: |d₀ - d̄|_smooth
    smooth_l1_initial = _smooth_l1_loss(pred_disparity_initial, gt_disparity, beta=1.0)
    loss_initial = smooth_l1_initial[mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=pred_disparity_initial.device)
    
    # Compute iterative refinement loss: Σ(k=1 to K) γ^(K-k) ||d_k - d̄||₁
    loss_iterative = torch.tensor(0.0, device=gt_disparity.device)
    K = len(pred_disparity_pyramid)
    
    total_epe = 0.0
    total_d1_error = 0.0
    total_d3_error = 0.0
    
    for k, pred_disp_k in enumerate(pred_disparity_pyramid):
        # Handle resolution mismatch
        if pred_disp_k.shape != gt_disparity.shape:
            if pred_disp_k.dim() == 2:
                pred_disp_k = pred_disp_k.unsqueeze(0).unsqueeze(0)
            elif pred_disp_k.dim() == 3:
                pred_disp_k = pred_disp_k.unsqueeze(0)
            
            pred_disp_k = F.interpolate(
                pred_disp_k,
                size=gt_disparity.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            while pred_disp_k.dim() > gt_disparity.dim():
                pred_disp_k = pred_disp_k.squeeze(0)
        
        # Clamp prediction
        pred_disp_k = torch.clamp(pred_disp_k, 0, max_disparity)
        
        # Compute weight: γ^(K-k) where k is 1-indexed
        weight = gamma ** (K - (k + 1))
        
        # Compute L1 loss for this iteration
        diff = torch.abs(pred_disp_k - gt_disparity)
        loss_k = diff[mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=pred_disp_k.device)
        loss_iterative += weight * loss_k
        
        # Compute metrics for this iteration
        with torch.no_grad():
            if mask.sum() > 0:
                epe_k = diff[mask].mean().item()
                d1_error_k = (diff[mask] > 3.0).float().mean().item()
                d3_error_k = (diff[mask] > 1.0).float().mean().item()
                
                total_epe += weight * epe_k
                total_d1_error += weight * d1_error_k
                total_d3_error += weight * d3_error_k
    
    # Total loss
    total_loss = loss_initial + loss_iterative
    
    # Compute metrics for initial disparity
    with torch.no_grad():
        if mask.sum() > 0:
            diff_initial = torch.abs(pred_disparity_initial - gt_disparity)
            epe_initial = diff_initial[mask].mean().item()
            d1_error_initial = (diff_initial[mask] > 3.0).float().mean().item()
            d3_error_initial = (diff_initial[mask] > 1.0).float().mean().item()
        else:
            epe_initial = 0.0
            d1_error_initial = 0.0
            d3_error_initial = 0.0
    
    misc = {
        'epe_initial': epe_initial,
        'd1_error_initial': d1_error_initial,
        'd3_error_initial': d3_error_initial,
        'epe_iterative': total_epe,
        'd1_error_iterative': total_d1_error,
        'd3_error_iterative': total_d3_error,
        'loss_initial': loss_initial.item(),
        'loss_iterative': loss_iterative.item(),
    }
    
    return total_loss, misc