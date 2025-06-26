"""
FoundationStereo Training Package

This package contains training utilities, data loaders, loss functions, and other
components needed for training the FoundationStereo model.
"""

from .dataloader import StereoTrainDataLoaderPipeline
from .losses import (
    disparity_l1_loss,
    disparity_smooth_l1_loss,
    disparity_epe_loss,
    multi_scale_loss,
    gradient_loss,
    monitoring_stereo,
    compute_stereo_metrics,
)
from .utils import (
    build_optimizer,
    build_lr_scheduler,
    compute_stereo_metrics,
    setup_parameter_groups,
    freeze_parameters,
    unfreeze_parameters,
    count_parameters,
    print_model_summary,
)

__all__ = [
    # Dataloader
    'StereoTrainDataLoaderPipeline',
    
    # Loss functions
    'disparity_l1_loss',
    'disparity_smooth_l1_loss',
    'disparity_epe_loss',
    'multi_scale_loss',
    'gradient_loss',
    'monitoring_stereo',
    'compute_stereo_metrics',
    
    # Training utilities
    'build_optimizer',
    'build_lr_scheduler',
    'setup_parameter_groups',
    'freeze_parameters',
    'unfreeze_parameters',
    'count_parameters',
    'print_model_summary',
]