import os
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[1])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
import time
import random
from typing import *
import itertools
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import io

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.version
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import click
from tqdm import tqdm, trange
import mlflow
torch.backends.cudnn.benchmark = False      # Varying input size, make sure cudnn benchmark is disabled

from train.dataloader import StereoTrainDataLoaderPipeline
from train.losses import (
    disparity_l1_loss,
    disparity_smooth_l1_loss,
    disparity_epe_loss,
    multi_scale_loss,
    monitoring_stereo,
)
from train.utils import build_optimizer, build_lr_scheduler, compute_stereo_metrics
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder
from Utils import vis_disparity, depth_uint8_decoding


class SimpleNamespace:
    """Simple namespace class to convert dict to object with attributes"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def get(self, key, default=None):
        """Support dict-like get method for compatibility"""
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        """Support dict-like access with square brackets"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Support dict-like assignment with square brackets"""
        setattr(self, key, value)
    
    def __contains__(self, key):
        """Support 'in' operator"""
        return hasattr(self, key)


def flatten_nested_dict(d, parent_key='', sep='.'):
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def key_average(records):
    """Average metrics across records"""
    if not records:
        return {}
    
    if isinstance(records, list):
        # Average across list of records
        avg_record = {}
        for record in records:
            for k, v in record.items():
                if isinstance(v, (int, float)):
                    if k not in avg_record:
                        avg_record[k] = []
                    avg_record[k].append(v)
        return {k: np.mean(v) for k, v in avg_record.items()}
    else:
        return records


@click.command()
@click.option('--config', 'config_path', type=str, default='configs/train/stereo_v1.json')
@click.option('--workspace', type=str, default='workspace/stereo_train', help='Path to the workspace')
@click.option('--checkpoint', 'checkpoint_path', type=str, default=None, help='Path to the checkpoint to load')
@click.option('--batch_size_forward', type=int, default=4, help='Batch size for each forward pass on each device')
@click.option('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
@click.option('--enable_gradient_checkpointing', type=bool, default=True, help='Use gradient checkpointing in backbone')
@click.option('--enable_mixed_precision', type=bool, default=False, help='Use mixed precision training. Backbone is converted to FP16')
@click.option('--enable_ema', type=bool, default=True, help='Maintain an exponential moving average of the model weights')
@click.option('--num_iterations', type=int, default=100000, help='Number of iterations to train the model')
@click.option('--save_every', type=int, default=5000, help='Save checkpoint every n iterations')
@click.option('--log_every', type=int, default=500, help='Log metrics every n iterations')
@click.option('--vis_every', type=int, default=0, help='Visualize every n iterations')
@click.option('--num_vis_images', type=int, default=16, help='Number of images to visualize, must be a multiple of divided batch size')
@click.option('--enable_mlflow', type=bool, default=True, help='Log metrics to MLFlow')
@click.option('--seed', type=int, default=0, help='Random seed')
def main(
    config_path: str,
    workspace: str,
    checkpoint_path: str,
    batch_size_forward: int,
    gradient_accumulation_steps: int,
    enable_gradient_checkpointing: bool,
    enable_mixed_precision: bool,
    enable_ema: bool,
    num_iterations: int,
    save_every: int,
    log_every: int,
    vis_every: int,
    num_vis_images: int,
    enable_mlflow: bool,
    seed: Optional[int],
):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='fp16' if enable_mixed_precision else None,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ]
    )
    device = accelerator.device
    batch_size_total = batch_size_forward * gradient_accumulation_steps * accelerator.num_processes

    # Log config
    if accelerator.is_main_process:
        if enable_mlflow:
            try:
                mlflow.log_params({
                    **click.get_current_context().params,
                    'batch_size_total': batch_size_total,
                })
            except:
                print('Failed to log config to MLFlow')
        Path(workspace).mkdir(parents=True, exist_ok=True)
        with Path(workspace).joinpath('config.json').open('w') as f:
            json.dump(config, f, indent=4)

    # Set seed
    if seed is not None:
        set_seed(seed, device_specific=True)

    # Initialize model
    print('Initialize model')
    with accelerator.local_main_process_first():
        # Convert model config dict to namespace object for FoundationStereo
        model_args = SimpleNamespace(**config['model'])
        model = FoundationStereo(model_args)
    count_total_parameters = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {count_total_parameters}')

    # Set up EMA model
    if enable_ema and accelerator.is_main_process:
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: 0.999 * averaged_model_parameter + 0.001 * model_parameter
        ema_model = torch.optim.swa_utils.AveragedModel(model, device=accelerator.device, avg_fn=ema_avg_fn)

    # Set gradient checkpointing
    if enable_gradient_checkpointing:
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
    
    # Initialize optimizer & lr scheduler
    optimizer = build_optimizer(model, config['optimizer'])
    lr_scheduler = build_lr_scheduler(optimizer, config['lr_scheduler'])

    count_grouped_parameters = [sum(p.numel() for p in param_group['params'] if p.requires_grad) for param_group in optimizer.param_groups]
    for i, count in enumerate(count_grouped_parameters):
        print(f'- Group {i}: {count} parameters')

    # Attempt to load checkpoint
    checkpoint: Dict[str, Any]
    with accelerator.local_main_process_first():
        if checkpoint_path and checkpoint_path.endswith('.pt'):
            # - Load specific checkpoint file
            print(f'Load checkpoint: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        elif checkpoint_path == "latest": 
            # - Load latest
            checkpoint_path = Path(workspace, 'checkpoint', 'latest.pt')
            if checkpoint_path.exists():
                print(f'Load checkpoint: {checkpoint_path}')
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                i_step = checkpoint['step']
                if 'model' not in checkpoint and (checkpoint_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}.pt')).exists():
                    print(f'Load model checkpoint: {checkpoint_model_path}')
                    checkpoint['model'] = torch.load(checkpoint_model_path, map_location='cpu', weights_only=True)['model']
                if 'optimizer' not in checkpoint and (checkpoint_optimizer_path := Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt')).exists():
                    print(f'Load optimizer checkpoint: {checkpoint_optimizer_path}')
                    checkpoint.update(torch.load(checkpoint_optimizer_path, map_location='cpu', weights_only=True))
                if enable_ema and accelerator.is_main_process:
                    if 'ema_model' not in checkpoint and (checkpoint_ema_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt')).exists():
                        print(f'Load EMA model checkpoint: {checkpoint_ema_model_path}')
                        checkpoint['ema_model'] = torch.load(checkpoint_ema_model_path, map_location='cpu', weights_only=True)['model']
            else:
                checkpoint = None
        elif checkpoint_path is not None:
            # - Load by step number
            i_step = int(checkpoint_path)
            checkpoint = {'step': i_step}
            if (checkpoint_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}.pt')).exists():
                print(f'Load model checkpoint: {checkpoint_model_path}')
                checkpoint['model'] = torch.load(checkpoint_model_path, map_location='cpu', weights_only=True)['model']
            if (checkpoint_optimizer_path := Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt')).exists():
                print(f'Load optimizer checkpoint: {checkpoint_optimizer_path}')
                checkpoint.update(torch.load(checkpoint_optimizer_path, map_location='cpu', weights_only=True))
            if enable_ema and accelerator.is_main_process:
                if (checkpoint_ema_model_path := Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt')).exists():
                    print(f'Load EMA model checkpoint: {checkpoint_ema_model_path}')
                    checkpoint['ema_model'] = torch.load(checkpoint_ema_model_path, map_location='cpu', weights_only=True)['model']
        else:
            checkpoint = None

    if checkpoint is None:
        # Initialize model weights
        print('Initialize model weights')
        with accelerator.local_main_process_first():
            if hasattr(model, 'init_weights'):
                model.init_weights()
        initial_step = 0
    else:
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'step' in checkpoint:
            initial_step = checkpoint['step'] + 1
        else:
            initial_step = 0
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if enable_ema and accelerator.is_main_process and 'ema_model' in checkpoint:
            ema_model.module.load_state_dict(checkpoint['ema_model'], strict=False)
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        del checkpoint
    
    model, optimizer = accelerator.prepare(model, optimizer)

    # Initialize training data pipeline
    with accelerator.local_main_process_first():
        train_data_pipe = StereoTrainDataLoaderPipeline(config['data'], batch_size_forward)

    def _write_bytes_retry_loop(save_path: Path, data: bytes):
        while True:
            try:
                save_path.write_bytes(data)
                break
            except Exception as e:
                print('Error while saving checkpoint, retrying in 1 minute: ', e)
                time.sleep(60)

    # Ready to train
    records = []
    model.train()
    with (
        train_data_pipe,
        tqdm(initial=initial_step, total=num_iterations, desc='Training', disable=not accelerator.is_main_process) as pbar,
        ThreadPoolExecutor(max_workers=1) as save_checkpoint_executor,
    ):  
        # Get some batches for visualization
        if accelerator.is_main_process:
            batches_for_vis: List[Dict[str, torch.Tensor]] = []
            num_vis_images = num_vis_images // batch_size_forward * batch_size_forward
            for _ in range(num_vis_images // batch_size_forward):
                batch = train_data_pipe.get()
                batches_for_vis.append(batch)

        # Visualize GT
        if vis_every > 0 and accelerator.is_main_process and initial_step == 0:
            save_dir = Path(workspace).joinpath('vis/gt')
            for i_batch, batch in enumerate(tqdm(batches_for_vis, desc='Visualize GT', leave=False)):
                left_image, right_image, gt_disparity, gt_mask, info = batch['left_image'], batch['right_image'], batch['disparity'], batch['disparity_mask'], batch['info']
                for i_instance in range(batch['left_image'].shape[0]):
                    idx = i_batch * batch_size_forward + i_instance
                    left_image_i = (left_image[i_instance].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    right_image_i = (right_image[i_instance].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    gt_disparity_i = gt_disparity[i_instance].numpy()
                    gt_mask_i = gt_mask[i_instance].numpy()
                    save_dir.joinpath(f'{idx:04d}').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/left_image.jpg')), cv2.cvtColor(left_image_i, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/right_image.jpg')), cv2.cvtColor(right_image_i, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/mask.png')), gt_mask_i * 255)
                    disparity_vis = vis_disparity(gt_disparity_i, invalid_thres=np.inf)
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/disparity_vis.png')), cv2.cvtColor(disparity_vis, cv2.COLOR_RGB2BGR))
                    with save_dir.joinpath(f'{idx:04d}/info.json').open('w') as f:
                        json.dump(info[i_instance], f)

        # Reset seed to avoid training on the same data when resuming training
        if seed is not None:
            set_seed(seed + initial_step, device_specific=True)   

        # Training loop
        for i_step in range(initial_step, num_iterations):

            i_accumulate, weight_accumulate = 0, 0
            while i_accumulate < gradient_accumulation_steps:
                # Load batch
                batch = train_data_pipe.get()
                left_image, right_image, gt_disparity, gt_mask, label_type = batch['left_image'], batch['right_image'], batch['disparity'], batch['disparity_mask'], batch['label_type']
                left_image, right_image, gt_disparity, gt_mask = left_image.to(device), right_image.to(device), gt_disparity.to(device), gt_mask.to(device)
                current_batch_size = left_image.shape[0]
                if all(label == 'invalid' for label in label_type):
                    continue            # NOTE: Skip all-invalid batches to avoid messing up the optimizer.
                
                # Pad images to ensure divisibility
                padder = InputPadder(left_image.shape, divis_by=32)
                left_image, right_image = padder.pad(left_image, right_image)

                with accelerator.accumulate(model):
                    # Forward
                    with torch.autocast(device_type=accelerator.device.type, dtype=torch.float16, enabled=enable_mixed_precision):
                        # FoundationStereo returns disparity directly or in a dict
                        output = model(left_image, right_image)
                    
                    # Handle different output formats
                    if isinstance(output, dict):
                        if 'disparity' in output:
                            pred_disparity = padder.unpad(output['disparity'])
                        elif 'flow_up' in output:  # FoundationStereo might use flow_up for final disparity
                            pred_disparity = padder.unpad(output['flow_up'])
                        else:
                            # Take the last item if it's a list/tuple
                            pred_disparity = padder.unpad(list(output.values())[-1])
                        
                        # Check for pyramid outputs
                        if 'disparity_pyramid' in output:
                            pred_disparity_pyramid = [padder.unpad(disp) for disp in output['disparity_pyramid']]
                        elif 'flow_predictions' in output:
                            pred_disparity_pyramid = [padder.unpad(disp) for disp in output['flow_predictions']]
                        else:
                            pred_disparity_pyramid = None
                    elif isinstance(output, (list, tuple)):
                        # Handle tuple/list output - take the first element as main disparity
                        pred_disparity = padder.unpad(output[0])
                        # If there are multiple outputs, treat the rest as pyramid
                        if len(output) > 1 and isinstance(output[1], (list, tuple)):
                            # Second element is a list of disparities (pyramid)
                            pred_disparity_pyramid = [padder.unpad(disp) for disp in output[1]]
                        elif len(output) > 1:
                            # Multiple tensor outputs
                            pred_disparity_pyramid = [padder.unpad(disp) for disp in output[1:] if hasattr(disp, 'ndim')]
                        else:
                            pred_disparity_pyramid = None
                    else:
                        # Direct tensor output
                        pred_disparity = padder.unpad(output)
                        pred_disparity_pyramid = None

                    # Compute loss (per instance)
                    loss_list, weight_list = [], []
                    for i in range(current_batch_size):
                        loss_dict, weight_dict, misc_dict = {}, {}, {}
                        misc_dict['monitoring'] = monitoring_stereo(pred_disparity[i])
                        for k, v in config['loss'][label_type[i]].items():
                            weight_dict[k] = v['weight']
                            if v['function'] == 'disparity_l1_loss':
                                loss_dict[k], misc_dict[k] = disparity_l1_loss(pred_disparity[i], gt_disparity[i], gt_mask[i], **v['params'])
                            elif v['function'] == 'disparity_smooth_l1_loss':
                                loss_dict[k], misc_dict[k] = disparity_smooth_l1_loss(pred_disparity[i], gt_disparity[i], gt_mask[i], **v['params'])
                            elif v['function'] == 'disparity_epe_loss':
                                loss_dict[k], misc_dict[k] = disparity_epe_loss(pred_disparity[i], gt_disparity[i], gt_mask[i], **v['params'])
                            elif v['function'] == 'multi_scale_loss' and pred_disparity_pyramid is not None:
                                loss_dict[k], misc_dict[k] = multi_scale_loss(pred_disparity_pyramid, gt_disparity[i], gt_mask[i], **v['params'])
                            else:
                                raise ValueError(f'Undefined loss function: {v["function"]}')
                        
                        # Flatten nested dictionaries
                        weight_dict = {'.'.join(k) if isinstance(k, tuple) else k: v for k, v in weight_dict.items()}
                        loss_dict = {'.'.join(k) if isinstance(k, tuple) else k: v for k, v in loss_dict.items()}
                        loss_ = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=torch.tensor(0.0, device=device))
                        loss_list.append(loss_)
                        
                        if torch.isnan(loss_).item():
                            pbar.write(f'NaN loss in process {accelerator.process_index}')
                            pbar.write(str(loss_dict))

                        misc_dict = {'.'.join(k) if isinstance(k, tuple) else k: v for k, v in misc_dict.items()}
                        records.append({
                            **{k: v.item() for k, v in loss_dict.items()},
                            **misc_dict,
                        })

                    loss = sum(loss_list) / len(loss_list)
                    
                    # Backward & update
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        if not enable_mixed_precision and any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
                            if accelerator.is_main_process:
                                pbar.write(f'NaN gradients, skip update')
                            optimizer.zero_grad()
                            continue
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                            
                    optimizer.step()
                    optimizer.zero_grad()

                i_accumulate += 1

            lr_scheduler.step()

            # EMA update            
            if enable_ema and accelerator.is_main_process and accelerator.sync_gradients:
                ema_model.update_parameters(model)

            # Log metrics
            if i_step == initial_step or i_step % log_every == 0:
                # records is already a list of dictionaries, no need to convert
                records = records if records else [{}]
                records = accelerator.gather_for_metrics(records, use_gather_object=True)
                if accelerator.is_main_process:
                    # Average metrics across all records
                    avg_records = {}
                    for record in records:
                        for k, v in record.items():
                            if k not in avg_records:
                                avg_records[k] = []
                            # Convert tensor to scalar if needed
                            if hasattr(v, 'item'):
                                avg_records[k].append(v.item())
                            elif isinstance(v, (int, float)):
                                avg_records[k].append(v)
                            else:
                                # Skip non-numeric values
                                continue
                    avg_records = {k: np.mean(v) for k, v in avg_records.items() if len(v) > 0}
                    
                    if enable_mlflow:
                        try:
                            mlflow.log_metrics(avg_records, step=i_step)
                        except Exception as e:
                            print(f'Error while logging metrics to mlflow: {e}')
                records = []

            # Save model weight checkpoint
            if accelerator.is_main_process and (i_step % save_every == 0):
                # NOTE: Writing checkpoint is done in a separate thread to avoid blocking the main process
                pbar.write(f'Save checkpoint: {i_step:08d}')
                Path(workspace, 'checkpoint').mkdir(parents=True, exist_ok=True)

                # Model checkpoint
                with io.BytesIO() as f:
                    torch.save({
                        'model_config': config['model'],
                        'model': accelerator.unwrap_model(model).state_dict(),
                    }, f)
                    checkpoint_bytes = f.getvalue()
                save_checkpoint_executor.submit(
                    _write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}.pt'), checkpoint_bytes
                )

                # Optimizer checkpoint
                with io.BytesIO() as f:
                    torch.save({
                        'model_config': config['model'],
                        'step': i_step,
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                    }, f)
                    checkpoint_bytes = f.getvalue()
                save_checkpoint_executor.submit(
                    _write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}_optimizer.pt'), checkpoint_bytes
                )
                
                # EMA model checkpoint
                if enable_ema:
                    with io.BytesIO() as f:
                        torch.save({
                            'model_config': config['model'],
                            'model': ema_model.module.state_dict(),
                        }, f)
                        checkpoint_bytes = f.getvalue()
                    save_checkpoint_executor.submit(
                        _write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}_ema.pt'), checkpoint_bytes
                    )

                # Latest checkpoint
                with io.BytesIO() as f:
                    torch.save({
                        'model_config': config['model'],
                        'step': i_step,
                    }, f)
                    checkpoint_bytes = f.getvalue()
                save_checkpoint_executor.submit(
                    _write_bytes_retry_loop, Path(workspace, 'checkpoint', 'latest.pt'), checkpoint_bytes
                )
            
            # Visualize
            if vis_every > 0 and accelerator.is_main_process and (i_step == initial_step or i_step % vis_every == 0):
                unwrapped_model = accelerator.unwrap_model(model)
                save_dir = Path(workspace).joinpath(f'vis/step_{i_step:08d}')
                save_dir.mkdir(parents=True, exist_ok=True)
                with torch.inference_mode():
                    for i_batch, batch in enumerate(tqdm(batches_for_vis, desc=f'Visualize: {i_step:08d}', leave=False)):
                        left_image, right_image, gt_disparity, gt_mask = batch['left_image'], batch['right_image'], batch['disparity'], batch['disparity_mask']
                        left_image, right_image, gt_disparity, gt_mask = left_image.to(device), right_image.to(device), gt_disparity.to(device), gt_mask.to(device)
                        
                        # Pad images
                        padder = InputPadder(left_image.shape, divis_by=32)
                        left_image_padded, right_image_padded = padder.pad(left_image, right_image)
                        
                        output = unwrapped_model(left_image_padded, right_image_padded)
                        
                        # Handle different output formats for visualization
                        if isinstance(output, dict):
                            if 'disparity' in output:
                                pred_disparity = padder.unpad(output['disparity']).cpu().numpy()
                            elif 'flow_up' in output:
                                pred_disparity = padder.unpad(output['flow_up']).cpu().numpy()
                            else:
                                pred_disparity = padder.unpad(list(output.values())[-1]).cpu().numpy()
                        else:
                            pred_disparity = padder.unpad(output).cpu().numpy()
                        left_image = left_image.cpu().numpy()
                        right_image = right_image.cpu().numpy()

                        for i_instance in range(left_image.shape[0]):
                            idx = i_batch * batch_size_forward + i_instance
                            left_image_i = (left_image[i_instance].transpose(1, 2, 0) * 255).astype(np.uint8)
                            right_image_i = (right_image[i_instance].transpose(1, 2, 0) * 255).astype(np.uint8)
                            pred_disparity_i = pred_disparity[i_instance]
                            save_dir.joinpath(f'{idx:04d}').mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/left_image.jpg')), cv2.cvtColor(left_image_i, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/right_image.jpg')), cv2.cvtColor(right_image_i, cv2.COLOR_RGB2BGR))
                            disparity_vis = vis_disparity(pred_disparity_i, invalid_thres=np.inf)
                            cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/pred_disparity_vis.png')), cv2.cvtColor(disparity_vis, cv2.COLOR_RGB2BGR))

            pbar.set_postfix({'loss': loss.item()}, refresh=False)
            pbar.update(1)


if __name__ == '__main__':
    main()