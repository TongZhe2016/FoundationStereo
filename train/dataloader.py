import os
from pathlib import Path
import json
import time
import random
from typing import *
import traceback
import itertools
from numbers import Number
import io

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms.v2.functional as TF
from tqdm import tqdm

import sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from Utils import depth_uint8_decoding, get_resize_keep_aspect_ratio
from core.utils.utils import InputPadder


class StereoTrainDataLoaderPipeline:
    def __init__(self, config: dict, batch_size: int, num_load_workers: int = 4, num_process_workers: int = 8, buffer_size: int = 8):
        self.config = config
        self.batch_size = batch_size
        self.max_disparity = config.get('max_disparity', 192)
        self.image_augmentation = config.get('image_augmentation', [])
        self.stereo_augmentation = config.get('stereo_augmentation', True)
        
        if 'image_sizes' in config:
            self.image_size_strategy = 'fixed'
            self.image_sizes = config['image_sizes']
        elif 'aspect_ratio_range' in config and 'area_range' in config:
            self.image_size_strategy = 'aspect_area'
            self.aspect_ratio_range = config['aspect_ratio_range']
            self.area_range = config['area_range']
        else:
            raise ValueError('Invalid image size configuration')

        # Load datasets
        self.datasets = {}
        for dataset in tqdm(config['datasets'], desc='Loading datasets'):
            name = dataset['name']
            dataset_path = Path(dataset['path'])
            
            # Find all stereo pairs
            left_rgb_path = dataset_path / 'left' / 'rgb'
            right_rgb_path = dataset_path / 'right' / 'rgb'
            left_disp_path = dataset_path / 'left' / 'disparity'
            
            if not all([left_rgb_path.exists(), right_rgb_path.exists(), left_disp_path.exists()]):
                raise ValueError(f"Dataset {name} missing required directories")
            
            # Get list of available files
            left_files = sorted([f.stem for f in left_rgb_path.glob('*.jpg')])
            right_files = sorted([f.stem for f in right_rgb_path.glob('*.jpg')])
            disp_files = sorted([f.stem for f in left_disp_path.glob('*.png')])
            
            # Find intersection of all three
            available_files = list(set(left_files) & set(right_files) & set(disp_files))
            available_files.sort()
            
            if len(available_files) == 0:
                raise ValueError(f"No matching stereo pairs found in dataset {name}")
            
            self.datasets[name] = {
                **dataset,
                'path': dataset_path,
                'filenames': available_files,
            }
            print(f"Dataset {name}: {len(available_files)} stereo pairs")
            
        self.dataset_names = [dataset['name'] for dataset in config['datasets']]
        self.dataset_weights = [dataset['weight'] for dataset in config['datasets']]

        # Build simple pipeline for stereo data
        self.invalid_instance = {
            'left_image': torch.zeros((3, 256, 256), dtype=torch.float32),
            'right_image': torch.zeros((3, 256, 256), dtype=torch.float32),
            'disparity': torch.ones((256, 256), dtype=torch.float32),
            'disparity_mask': torch.zeros((256, 256), dtype=torch.bool),
            'label_type': 'invalid',
        }

    def _sample_batch(self):
        batch_id = 0
        while True:
            batch_id += 1
            batch = []
            
            # Sample instances
            for _ in range(self.batch_size):
                dataset_name = random.choices(self.dataset_names, weights=self.dataset_weights)[0]
                filename = random.choice(self.datasets[dataset_name]['filenames'])

                instance = {
                    'batch_id': batch_id,
                    'seed': random.randint(0, 2 ** 32 - 1),
                    'dataset': dataset_name,
                    'filename': filename,
                    'label_type': self.datasets[dataset_name]['label_type'],
                }
                batch.append(instance)

            # Decide the image size for this batch
            if self.image_size_strategy == 'fixed':
                width, height = random.choice(self.config['image_sizes'])
            elif self.image_size_strategy == 'aspect_area':
                area = random.uniform(*self.area_range)
                aspect_ratio_ranges = [self.datasets[instance['dataset']].get('aspect_ratio_range', self.aspect_ratio_range) for instance in batch]
                aspect_ratio_range = (min(r[0] for r in aspect_ratio_ranges), max(r[1] for r in aspect_ratio_ranges))
                aspect_ratio = random.uniform(*aspect_ratio_range)
                width, height = int((area * aspect_ratio) ** 0.5), int((area / aspect_ratio) ** 0.5)
            else:
                raise ValueError('Invalid image size strategy')
            
            for instance in batch:
                instance['width'], instance['height'] = width, height
            
            yield batch

    def _load_instance(self, instance: dict):
        try:
            dataset_path = self.datasets[instance['dataset']]['path']
            filename = instance['filename']
            
            # Load left and right images
            left_image_path = dataset_path / 'left' / 'rgb' / f'{filename}.jpg'
            right_image_path = dataset_path / 'right' / 'rgb' / f'{filename}.jpg'
            disparity_path = dataset_path / 'left' / 'disparity' / f'{filename}.png'
            
            left_image = np.array(Image.open(left_image_path).convert('RGB'))
            right_image = np.array(Image.open(right_image_path).convert('RGB'))
            
            # Load disparity using FoundationStereo's method
            disparity_uint8 = np.array(Image.open(disparity_path))
            if len(disparity_uint8.shape) == 3:
                disparity = depth_uint8_decoding(disparity_uint8)
            else:
                # Fallback for single channel disparity
                disparity = disparity_uint8.astype(np.float32)
            
            # Create disparity mask (valid where disparity > 0)
            disparity_mask = disparity > 0
            
            # Ensure images have the same size
            if left_image.shape[:2] != right_image.shape[:2]:
                raise ValueError(f"Left and right images have different sizes: {left_image.shape[:2]} vs {right_image.shape[:2]}")
            
            if left_image.shape[:2] != disparity.shape[:2]:
                raise ValueError(f"Image and disparity have different sizes: {left_image.shape[:2]} vs {disparity.shape[:2]}")
            
            instance.update({
                'left_image': left_image,
                'right_image': right_image,
                'disparity': disparity,
                'disparity_mask': disparity_mask,
            })
            
        except Exception as e:
            print(f"Failed to load instance {instance['dataset']}/{instance['filename']} because of exception:", e)
            instance.update(self.invalid_instance)
        return instance

    def _process_instance(self, instance: Dict[str, Union[np.ndarray, str, float, bool]]):
        if instance['label_type'] == 'invalid':
            return instance
            
        left_image, right_image, disparity, disparity_mask = instance['left_image'], instance['right_image'], instance['disparity'], instance['disparity_mask']
        
        raw_height, raw_width = left_image.shape[:2]
        tgt_width, tgt_height = instance['width'], instance['height']
        
        rng = np.random.default_rng(instance['seed'])

        # 1. Resize images and disparity
        # Use keep aspect ratio resize to maintain stereo geometry
        resize_height, resize_width = get_resize_keep_aspect_ratio(raw_height, raw_width, max_H=tgt_height, max_W=tgt_width)
        
        # Resize images
        left_image = cv2.resize(left_image, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        right_image = cv2.resize(right_image, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        
        # Resize disparity (scale by width ratio)
        width_ratio = resize_width / raw_width
        disparity = cv2.resize(disparity, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST) * width_ratio
        disparity_mask = cv2.resize(disparity_mask.astype(np.uint8), (resize_width, resize_height), interpolation=cv2.INTER_NEAREST) > 0

        # 2. Center crop to target size
        if resize_height > tgt_height or resize_width > tgt_width:
            start_y = (resize_height - tgt_height) // 2
            start_x = (resize_width - tgt_width) // 2
            left_image = left_image[start_y:start_y+tgt_height, start_x:start_x+tgt_width]
            right_image = right_image[start_y:start_y+tgt_height, start_x:start_x+tgt_width]
            disparity = disparity[start_y:start_y+tgt_height, start_x:start_x+tgt_width]
            disparity_mask = disparity_mask[start_y:start_y+tgt_height, start_x:start_x+tgt_width]
        elif resize_height < tgt_height or resize_width < tgt_width:
            # Pad if needed
            pad_y = max(0, tgt_height - resize_height)
            pad_x = max(0, tgt_width - resize_width)
            pad_top, pad_bottom = pad_y // 2, pad_y - pad_y // 2
            pad_left, pad_right = pad_x // 2, pad_x - pad_x // 2
            
            left_image = np.pad(left_image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
            right_image = np.pad(right_image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
            disparity = np.pad(disparity, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
            disparity_mask = np.pad(disparity_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

        # 3. Stereo-consistent augmentations
        if self.stereo_augmentation:
            # Horizontal flip (flip both images horizontally then swap left and right)
            if rng.choice([True, False]):
                # First flip both images horizontally
                left_image_flipped = np.flip(left_image, axis=1).copy()
                right_image_flipped = np.flip(right_image, axis=1).copy()
                # Then swap the flipped images
                left_image, right_image = right_image_flipped, left_image_flipped
                # Flip disparity horizontally
                disparity = np.flip(disparity, axis=1).copy()
                disparity_mask = np.flip(disparity_mask, axis=1).copy()
            
            # Vertical flip (both images)
            if rng.choice([True, False]):
                left_image = np.flip(left_image, axis=0).copy()
                right_image = np.flip(right_image, axis=0).copy()
                disparity = np.flip(disparity, axis=0).copy()
                disparity_mask = np.flip(disparity_mask, axis=0).copy()

        # 4. Color augmentation (applied to both images consistently)
        image_augmentation = self.datasets[instance['dataset']].get('image_augmentation', self.image_augmentation)
        if 'jittering' in image_augmentation:
            # Apply same color transformation to both images
            brightness_factor = rng.uniform(0.8, 1.2)
            contrast_factor = rng.uniform(0.8, 1.2)
            # Enhanced saturation range: 0 (grayscale) to 1.4
            saturation_factor = rng.uniform(0.0, 1.4)
            hue_factor = rng.uniform(-0.05, 0.05)
            gamma_factor = rng.uniform(0.8, 1.2)
            
            # Apply color augmentation to left image
            left_img_tensor = torch.from_numpy(left_image).permute(2, 0, 1)
            left_img_tensor = TF.adjust_brightness(left_img_tensor, brightness_factor)
            left_img_tensor = TF.adjust_contrast(left_img_tensor, contrast_factor)
            left_img_tensor = TF.adjust_saturation(left_img_tensor, saturation_factor)
            left_img_tensor = TF.adjust_hue(left_img_tensor, hue_factor)
            left_img_tensor = TF.adjust_gamma(left_img_tensor, gamma_factor)
            left_image = left_img_tensor.permute(1, 2, 0).numpy()
            
            # Apply color augmentation to right image (with potential perturbation)
            right_img_tensor = torch.from_numpy(right_image).permute(2, 0, 1)
            right_img_tensor = TF.adjust_brightness(right_img_tensor, brightness_factor)
            right_img_tensor = TF.adjust_contrast(right_img_tensor, contrast_factor)
            right_img_tensor = TF.adjust_saturation(right_img_tensor, saturation_factor)
            right_img_tensor = TF.adjust_hue(right_img_tensor, hue_factor)
            right_img_tensor = TF.adjust_gamma(right_img_tensor, gamma_factor)
            right_image = right_img_tensor.permute(1, 2, 0).numpy()
            
            # Right image perturbation to simulate imperfect rectification
            # (as seen in ETH3D and Middlebury datasets)
            if rng.choice([True, False], p=[0.3, 0.7]):  # 30% chance of perturbation
                # Small random translation and rotation
                tx = rng.uniform(-2.0, 2.0)  # horizontal translation in pixels
                ty = rng.uniform(-1.0, 1.0)  # vertical translation in pixels
                angle = rng.uniform(-0.5, 0.5)  # rotation in degrees
                
                # Create transformation matrix
                h, w = right_image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                M[0, 2] += tx
                M[1, 2] += ty
                
                # Apply transformation
                right_image = cv2.warpAffine(right_image, M, (w, h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REFLECT)

        # 5. Disparity stretching augmentation
        # Apply random stretching factor in range [2.02, 2.04] to simulate different disparity ranges
        if 'disparity_stretching' in image_augmentation and rng.choice([True, False], p=[0.5, 0.5]):
            stretch_factor = rng.uniform(2.02, 2.04)
            
            # Stretch images horizontally
            h, w = left_image.shape[:2]
            new_w = int(w * stretch_factor)
            
            # Resize images
            left_image = cv2.resize(left_image, (new_w, h), interpolation=cv2.INTER_LINEAR)
            right_image = cv2.resize(right_image, (new_w, h), interpolation=cv2.INTER_LINEAR)
            
            # Stretch disparity accordingly
            disparity = cv2.resize(disparity, (new_w, h), interpolation=cv2.INTER_NEAREST) * stretch_factor
            disparity_mask = cv2.resize(disparity_mask.astype(np.uint8), (new_w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # Crop back to original size from center
            start_x = (new_w - w) // 2
            left_image = left_image[:, start_x:start_x + w]
            right_image = right_image[:, start_x:start_x + w]
            disparity = disparity[:, start_x:start_x + w]
            disparity_mask = disparity_mask[:, start_x:start_x + w]

        # 6. Clamp disparity to max range
        disparity = np.clip(disparity, 0, self.max_disparity)
        
        # Ensure mask is not empty
        if disparity_mask.sum() / disparity_mask.size < 0.001:
            disparity_mask = np.ones_like(disparity_mask)
            disparity = np.ones_like(disparity)
            instance['label_type'] = 'invalid'

        instance.update({
            'left_image': torch.from_numpy(left_image.astype(np.float32) / 255.0).permute(2, 0, 1),
            'right_image': torch.from_numpy(right_image.astype(np.float32) / 255.0).permute(2, 0, 1),
            'disparity': torch.from_numpy(disparity).float(),
            'disparity_mask': torch.from_numpy(disparity_mask).bool(),
        })
        
        return instance

    def _collate_batch(self, instances: List[Dict[str, Any]]):
        batch = {k: torch.stack([instance[k] for instance in instances], dim=0) 
                for k in ['left_image', 'right_image', 'disparity', 'disparity_mask']}
        batch.update({
            'label_type': [instance['label_type'] for instance in instances],
            'info': [{'dataset': instance['dataset'], 'filename': instance['filename']} for instance in instances],
        })
        return batch

    def get(self) -> Dict[str, Union[torch.Tensor, str]]:
        # Simple synchronous data loading for now
        batch_data = next(self._sample_batch())
        
        # Load instances
        for instance in batch_data:
            instance = self._load_instance(instance)
            instance = self._process_instance(instance)
        
        # Collate batch
        return self._collate_batch(batch_data)

    def start(self):
        # For compatibility with MoGe pipeline interface
        pass

    def stop(self):
        # For compatibility with MoGe pipeline interface
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False


# Utility functions for stereo data processing
def stereo_consistent_crop(left_img, right_img, disparity, disparity_mask, crop_height, crop_width, random_crop=True):
    """
    Perform stereo-consistent cropping that maintains epipolar geometry
    """
    h, w = left_img.shape[:2]
    
    if random_crop:
        start_y = np.random.randint(0, max(1, h - crop_height + 1))
        start_x = np.random.randint(0, max(1, w - crop_width + 1))
    else:
        start_y = (h - crop_height) // 2
        start_x = (w - crop_width) // 2
    
    end_y = start_y + crop_height
    end_x = start_x + crop_width
    
    left_img = left_img[start_y:end_y, start_x:end_x]
    right_img = right_img[start_y:end_y, start_x:end_x]
    disparity = disparity[start_y:end_y, start_x:end_x]
    disparity_mask = disparity_mask[start_y:end_y, start_x:end_x]
    
    return left_img, right_img, disparity, disparity_mask


def stereo_consistent_resize(left_img, right_img, disparity, disparity_mask, target_height, target_width):
    """
    Perform stereo-consistent resizing that scales disparity appropriately
    """
    h, w = left_img.shape[:2]
    width_ratio = target_width / w
    
    left_img = cv2.resize(left_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    right_img = cv2.resize(right_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    disparity = cv2.resize(disparity, (target_width, target_height), interpolation=cv2.INTER_NEAREST) * width_ratio
    disparity_mask = cv2.resize(disparity_mask.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST) > 0
    
    return left_img, right_img, disparity, disparity_mask