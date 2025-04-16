import os, sys
import shutil
import numpy as np
import datetime
import argparse
import logging
import cv2
import imageio
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F

# Add parent directory to path for imports
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

# Import project modules
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *

# Configure matplotlib backend
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# OmniDataset class for loading ERP stereo pairs
class OmniDataset(Dataset):
    def __init__(self, dataset_path, split='train', scale=1.0, max_disp=416):
        self.dataset_path = dataset_path
        self.scale = scale
        self.max_disp = max_disp
        
        # Get list of all subdirectories (each represents a scene)
        if os.path.isdir(os.path.join(dataset_path, 'cam1')):
            # This is a single dataset directory
            self.datasets = [dataset_path]
        else:
            # This is a parent directory containing multiple datasets
            self.datasets = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) 
                           if os.path.isdir(os.path.join(dataset_path, d))]
        
        logging.info(f"Found {len(self.datasets)} dataset directories")
        
        # Collect all image pairs
        self.samples = []
        for dataset_dir in self.datasets:
            # Check if this is OmniHouse format with cam1, cam2, cam3, cam4 directories
            if os.path.exists(os.path.join(dataset_dir, 'cam1')) and os.path.exists(os.path.join(dataset_dir, 'cam3')):
                logging.info(f"Processing dataset in directory: {dataset_dir}")
                
                # OmniHouse format: front (cam1) and rear (cam3) are used as stereo pair
                img_files = [f for f in os.listdir(os.path.join(dataset_dir, 'cam1')) if f.endswith('.png')]
                logging.info(f"Found {len(img_files)} image files in {os.path.join(dataset_dir, 'cam1')}")
                
                # Split images into train/val
                random.seed(42)  # For reproducibility
                random.shuffle(img_files)
                
                if split == 'train':
                    img_files = img_files[:int(0.8 * len(img_files))]
                else:  # val
                    img_files = img_files[int(0.8 * len(img_files)):]
                
                for img_file in img_files:
                    left_path = os.path.join(dataset_dir, 'cam1', img_file)
                    right_path = os.path.join(dataset_dir, 'cam3', img_file)
                    
                    # Check if depth ground truth exists
                    depth_path = None
                    if os.path.exists(os.path.join(dataset_dir, 'omnidepth_gt_640')):
                        depth_file = img_file.replace('.png', '.npy')
                        depth_path = os.path.join(dataset_dir, 'omnidepth_gt_640', depth_file)
                    
                    # Get intrinsic file
                    intrinsic_path = os.path.join(dataset_dir, 'intrinsic_extrinsic.mat')
                    if not os.path.exists(intrinsic_path):
                        intrinsic_path = os.path.join(dataset_dir, 'K.txt')
                    
                    if os.path.exists(left_path) and os.path.exists(right_path) and (depth_path is None or os.path.exists(depth_path)):
                        self.samples.append({
                            'left': left_path,
                            'right': right_path,
                            'depth': depth_path,
                            'intrinsic': intrinsic_path
                        })
            else:
                # Assume standard format with up_erp.png and down_erp.png
                left_path = os.path.join(dataset_dir, 'up_erp.png')
                right_path = os.path.join(dataset_dir, 'down_erp.png')
                intrinsic_path = os.path.join(dataset_dir, 'K.txt')
                
                if os.path.exists(left_path) and os.path.exists(right_path) and os.path.exists(intrinsic_path):
                    self.samples.append({
                        'left': left_path,
                        'right': right_path,
                        'depth': None,
                        'intrinsic': intrinsic_path
                    })
        
        logging.info(f"Total samples found: {len(self.samples)}")
        if len(self.samples) == 0:
            logging.error(f"No valid samples found in {dataset_path}!")
            logging.error(f"Please check the dataset structure.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        left_img = imageio.imread(sample['left'])
        right_img = imageio.imread(sample['right'])
        
        # Resize if needed
        if self.scale < 1.0:
            left_img = cv2.resize(left_img, None, fx=self.scale, fy=self.scale)
            right_img = cv2.resize(right_img, None, fx=self.scale, fy=self.scale)
        
        # Convert to RGB if grayscale
        if len(left_img.shape) == 2:
            left_img = left_img[:, :, None]
            right_img = right_img[:, :, None]
            left_img = np.concatenate([left_img, left_img, left_img], axis=2)
            right_img = np.concatenate([right_img, right_img, right_img], axis=2)
        
        # Handle RGBA
        if left_img.shape[2] == 4:
            left_img = left_img[:, :, :3]
            right_img = right_img[:, :, :3]
        
        # Load depth if available
        depth_gt = None
        if sample['depth'] is not None and os.path.exists(sample['depth']):
            depth_gt = np.load(sample['depth'])
            if self.scale < 1.0:
                depth_gt = cv2.resize(depth_gt, None, fx=self.scale, fy=self.scale)
        
        # Convert to torch tensors
        left_tensor = torch.from_numpy(left_img).float().permute(2, 0, 1)
        right_tensor = torch.from_numpy(right_img).float().permute(2, 0, 1)
        
        result = {
            'left': left_tensor,
            'right': right_tensor,
            'intrinsic_path': sample['intrinsic']
        }
        
        if depth_gt is not None:
            result['depth_gt'] = torch.from_numpy(depth_gt).float()
        
        return result

# Loss function for disparity estimation
def disparity_loss(pred_disp, gt_disp, mask=None):
    if mask is None:
        mask = (gt_disp > 0) & (gt_disp < 512)
    
    # L1 loss
    l1_loss = torch.abs(pred_disp[mask] - gt_disp[mask]).mean()
    
    return l1_loss

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='FoundationStereo training script for ERP stereo depth estimation')
    
    # Dataset arguments
    parser.add_argument('--dataset_path', default='./OmniDatasets', type=str, 
                      help='Path to the dataset')
    parser.add_argument('--batch_size', default=2, type=int, 
                      help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int, 
                      help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--ckpt_dir', default='./pretrained_models/model_best_bp2.pth', type=str, 
                      help='Path to pretrained model')
    parser.add_argument('--output_dir', default='./trained_models/', type=str, 
                      help='Directory to save results')
    
    # Training arguments
    parser.add_argument('--lr', default=1e-5, type=float, 
                      help='Learning rate')
    parser.add_argument('--epochs', default=10, type=int, 
                      help='Number of epochs')
    parser.add_argument('--scale', default=0.5, type=float, 
                      help='Scale factor for input images')
    parser.add_argument('--valid_iters', default=16, type=int, 
                      help='Number of flow-field updates during forward pass')
    parser.add_argument('--save_interval', default=1, type=int, 
                      help='Save model every N epochs')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir = os.path.join(args.output_dir, f'erp_finetune_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    set_logging_format()
    set_seed(0)
    
    # Save configuration
    cfg_path = os.path.join(os.path.dirname(args.ckpt_dir), 'cfg.yaml')
    if os.path.exists(cfg_path):
        cfg = OmegaConf.load(cfg_path)
    else:
        # Create default config
        cfg = OmegaConf.create({
            'vit_size': 'vitl',
            'hidden_dims': [128, 128, 128],
            'n_gru_layers': 3,
            'n_downsample': 2,
            'max_disp': 416,
            'corr_radius': 4,
            'corr_levels': 2,
            'mixed_precision': True
        })
    
    # Update config with args
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    
    # Save updated config
    OmegaConf.save(cfg, os.path.join(args.output_dir, 'cfg.yaml'))
    
    # Create datasets and dataloaders
    train_dataset = OmniDataset(args.dataset_path, split='train', scale=args.scale)
    val_dataset = OmniDataset(args.dataset_path, split='val', scale=args.scale)
    
    if len(train_dataset) == 0:
        logging.error("Training dataset is empty! Exiting...")
        return
    
    if len(val_dataset) == 0:
        logging.warning("Validation dataset is empty! Using a subset of training data for validation.")
        # Split training data for validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Val dataset size: {len(val_dataset)}")
    
    # Initialize model
    model = FoundationStereo(cfg)
    
    # Load pretrained weights
    if os.path.exists(args.ckpt_dir):
        logging.info(f"Loading pretrained model from {args.ckpt_dir}")
        ckpt = torch.load(args.ckpt_dir)
        model.load_state_dict(ckpt['model'])
    else:
        logging.warning(f"Pretrained model not found at {args.ckpt_dir}, starting from scratch")
    
    model.cuda()
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_losses = []
        
        for i, batch in enumerate(train_loader):
            # Move data to GPU
            left = batch['left'].cuda()
            right = batch['right'].cuda()
            
            # Forward pass
            _, disp_preds = model(left, right, iters=args.valid_iters)
            
            # Calculate loss
            loss = 0
            if 'depth_gt' in batch:
                depth_gt = batch['depth_gt'].cuda()
                # Convert depth to disparity
                # Note: This is a simplified conversion and may need to be adjusted
                # based on your dataset's camera parameters
                disp_gt = 1.0 / (depth_gt + 1e-6)
                disp_gt = disp_gt / disp_gt.max() * cfg.max_disp
                
                # Calculate loss on the final prediction
                loss = disparity_loss(disp_preds[-1].squeeze(1), disp_gt)
            else:
                # If no ground truth, use self-supervised loss
                # This is a placeholder and should be replaced with a proper
                # self-supervised loss function for your specific case
                loss = torch.tensor(0.0).cuda()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if (i + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{args.epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Average Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # Move data to GPU
                left = batch['left'].cuda()
                right = batch['right'].cuda()
                
                # Forward pass
                _, disp_preds = model(left, right, iters=args.valid_iters)
                
                # Calculate loss
                loss = 0
                if 'depth_gt' in batch:
                    depth_gt = batch['depth_gt'].cuda()
                    # Convert depth to disparity
                    disp_gt = 1.0 / (depth_gt + 1e-6)
                    disp_gt = disp_gt / disp_gt.max() * cfg.max_disp
                    
                    # Calculate loss on the final prediction
                    loss = disparity_loss(disp_preds[-1].squeeze(1), disp_gt)
                else:
                    # If no ground truth, use self-supervised loss
                    loss = torch.tensor(0.0).cuda()
                
                val_losses.append(loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Average Val Loss: {avg_val_loss:.4f}")
        
        # Save model
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, save_path)
            logging.info(f"Model saved to {save_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.output_dir, 'model_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, save_path)
            logging.info(f"Best model saved to {save_path}")

if __name__ == "__main__":
    main()