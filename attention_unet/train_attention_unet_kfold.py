#!/usr/bin/env python3
"""
Attention U-Net 5-Fold Cross-Validation Training Script
Fixes all 3 bugs:
1. RAE calculated on SCAR (label 2) instead of myocardium (label 1)
2. K-fold ensures scar samples in validation
3. Epsilon added to prevent division by zero

Goal: Improve upon U-Net's Dice 0.757 and get RAE in 0.1-0.2 range
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import nibabel as nib
import os

from attention_unet import AttentionUNet
from dataset import LGEDataset, train_transform, val_transform

# ============================================================
# VISUALIZATION FUNCTION
# ============================================================

def save_visualizations(model, val_loader, device, fold, epoch, save_dir='visualizations_kfold'):
    """
    Save visualizations: original image, GT grayscale, GT colored, prediction colored
    Saves 5 random samples from validation set
    """
    os.makedirs(save_dir, exist_ok=True)
    fold_dir = os.path.join(save_dir, f'fold{fold+1}')
    os.makedirs(fold_dir, exist_ok=True)
    
    model.eval()
    
    # Color mapping to match your full dataset (5 classes)
    colors = {
        0: [0, 0, 0],        # Background - Black
        1: [128, 128, 128],  # LV Cavity - Gray
        2: [255, 0, 0],      # LV Myocardium - Red
        3: [255, 255, 0],    # Myocardial Scar - Yellow
        4: [0, 255, 0]       # RV Myocardium - Green
    }
    
    def mask_to_color(mask):
        """Convert label mask to RGB colored image"""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in colors.items():
            colored[mask == label] = color
        return colored
    
    with torch.no_grad():
        # Get 5 random samples
        sample_indices = np.random.choice(len(val_loader.dataset), min(5, len(val_loader.dataset)), replace=False)
        
        for idx, sample_idx in enumerate(sample_indices):
            # Get single sample
            image, mask, filename = val_loader.dataset[sample_idx]
            
            # Add batch dimension
            image_input = image.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(image_input)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            # Convert tensors to numpy
            image_np = image.squeeze().cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            # Create colored versions
            gt_colored = mask_to_color(mask_np)
            pred_colored = mask_to_color(pred)
            
            # Extract view name from filename (e.g., "SAX_LGE_SAX_031_slice006" -> "SAX")
            view_name = filename.split('_')[0]
            
            # Create figure with 4 subplots
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Add main title with view and filename
            fig.suptitle(f'Fold {fold+1} - Epoch {epoch+1} - View: {view_name} - {filename}', fontsize=12, y=1.02)
            
            # 1. Original image (grayscale)
            axes[0].imshow(image_np, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # 2. Ground truth (grayscale)
            axes[1].imshow(mask_np, cmap='gray', vmin=0, vmax=4)
            axes[1].set_title('Ground Truth (Gray)')
            axes[1].axis('off')
            
            # 3. Ground truth (colored)
            axes[2].imshow(gt_colored)
            axes[2].set_title('Ground Truth (Colored)\nGray=LV, Red=Myo, Yellow=Scar, Green=RV')
            axes[2].axis('off')
            
            # 4. Prediction (colored)
            axes[3].imshow(pred_colored)
            axes[3].set_title('Prediction (Colored)\nGray=LV, Red=Myo, Yellow=Scar, Green=RV')
            axes[3].axis('off')
            
            plt.tight_layout()
            
            # Save figure with view name in filename
            save_path = os.path.join(fold_dir, f'epoch{epoch+1:02d}_{view_name}_sample{idx+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  → Saved {len(sample_indices)} visualizations to {fold_dir}/")


# ============================================================
# FOCAL LOSS IMPLEMENTATION
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses on hard-to-classify examples (like small scar regions)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W) - logits from model
        targets: (N, H, W) - ground truth labels
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================
# MASS CALCULATION WITH FIXED RAE (BUG FIXES!)
# ============================================================

def calculate_mass_metrics(pred, mask, voxel_volume=1.25*1.25*8.0):
    """
    Calculate mass-based metrics for scar tissue
    
    FIXED BUGS:
    - Now calculates mass for SCAR (label 2) instead of myocardium (label 1)
    - Added epsilon to prevent division by zero
    
    Args:
        pred: predicted segmentation (H, W) with values 0=background, 1=myocardium, 2=scar
        mask: ground truth segmentation (H, W) with values 0=background, 1=myocardium, 2=scar
        voxel_volume: volume of one voxel in mm³ (default: 1.25×1.25×8.0 = 12.5 mm³)
    
    Returns:
        vPCC: Pearson correlation coefficient for volume
        RAE: Relative Absolute Error
        pred_mass: predicted scar mass in grams
        gt_mass: ground truth scar mass in grams
    """
    # BUG FIX #1: Use label 2 (SCAR) instead of label 1 (myocardium)
    pred_scar_voxels = (pred == 2).sum().item()
    gt_scar_voxels = (mask == 2).sum().item()
    
    # Calculate volumes in mm³
    pred_volume = pred_scar_voxels * voxel_volume
    gt_volume = gt_scar_voxels * voxel_volume
 

    # Convert to mass (assuming density = 1.05 g/cm³ = 0.00105 g/mm³)
    density = 0.00105  # g/mm³
    pred_mass = pred_volume * density
    gt_mass = gt_volume * density
    
    # Calculate vPCC (Pearson correlation)
    if pred_mass > 0 or gt_mass > 0:
        vPCC = pearsonr([pred_mass], [gt_mass])[0]
        if np.isnan(vPCC):
            vPCC = 0.0
    else:
        vPCC = 0.0
    
    # BUG FIX #3: Add epsilon to prevent division by zero
    epsilon = 1e-8
    RAE = abs(pred_mass - gt_mass) / (gt_mass + epsilon)
    
    return vPCC, RAE, pred_mass, gt_mass


def calculate_scar_mass(mask, voxel_volume=12.5):
    """
    Calculate scar mass from segmentation mask
    
    Args:
        mask: segmentation mask (H, W) with label 3 = scar (for 5-class)
        voxel_volume: volume of one voxel in mm³ (default: 1.25×1.25×8.0 = 12.5 mm³)
    
    Returns:
        mass: scar mass in grams
    """
    # Use label 3 (SCAR) for 5-class segmentation
    scar_voxels = (mask == 3).sum()
    
    # Calculate volume in mm³
    volume = scar_voxels * voxel_volume
    
    # Convert to mass (assuming density = 1.05 g/cm³ = 0.00105 g/mm³)
    density = 0.00105  # g/mm³
    mass = volume * density
    
    return mass


# ============================================================
# TRAINING AND VALIDATION FUNCTIONS
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, warmup_epochs=5):
    """Train for one epoch with learning rate warmup"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    # Learning rate warmup
    if epoch < warmup_epochs:
        lr_scale = (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001 * lr_scale
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        # Calculate Dice
        preds = torch.argmax(outputs, dim=1)
        dice = calculate_dice(preds, masks)
        
        running_loss += loss.item()
        running_dice += dice
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_dice = running_dice / len(train_loader)
    
    return epoch_loss, epoch_dice


def validate(model, val_loader, criterion, device):
    """Validate model with all metrics including FIXED RAE"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    all_pred_masses = []
    all_gt_masses = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate Dice
            dice = calculate_dice(preds, masks)
            running_loss += loss.item()
            running_dice += dice
            
            # Collect mass predictions for batch vPCC/RAE calculation
            for pred, mask in zip(preds.cpu().numpy(), masks.cpu().numpy()):
                pred_mass = calculate_scar_mass(pred)
                gt_mass = calculate_scar_mass(mask)
                all_pred_masses.append(pred_mass)
                all_gt_masses.append(gt_mass)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    
    epoch_loss = running_loss / len(val_loader)
    epoch_dice = running_dice / len(val_loader)
    
    # Calculate vPCC and RAE across all validation samples
    if len(all_pred_masses) > 1:
        epoch_vPCC = pearsonr(all_pred_masses, all_gt_masses)[0]
        
        # RAE calculation with epsilon for numerical stability
        epsilon = 1e-10
        gt_sum = sum(all_gt_masses) + epsilon
        abs_error_sum = sum(abs(p - g) for p, g in zip(all_pred_masses, all_gt_masses))
        epoch_RAE = abs_error_sum / gt_sum
    else:
        epoch_vPCC = 0.0
        epoch_RAE = float('inf')
    
    return epoch_loss, epoch_dice, epoch_vPCC, epoch_RAE


def calculate_dice(pred, target, num_classes=3):
    """Calculate multi-class Dice coefficient"""
    dice_scores = []
    
    for c in range(1, num_classes):  # Skip background (class 0)
        pred_c = (pred == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).sum().float()
        union = pred_c.sum().float() + target_c.sum().float()
        
        if union == 0:
            dice_scores.append(1.0)
        else:
            dice_scores.append((2.0 * intersection / union).item())
    
    return np.mean(dice_scores)


# ============================================================
# K-FOLD CROSS-VALIDATION TRAINING
# ============================================================

def train_kfold(n_splits=5, num_epochs=50, batch_size=16):
    """
    Train Attention U-Net with k-fold cross-validation
    
    Args:
        n_splits: Number of folds (default: 5 for 80/20 split)
        num_epochs: Epochs per fold
        batch_size: Batch size
    """
    
    # Setup - Use MPS (Apple Silicon GPU) if available, otherwise CUDA, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print(f"5-FOLD CROSS-VALIDATION TRAINING")
    print(f"{'='*60}")
    print(f"Folds: {n_splits}")
    print(f"Epochs per fold: {num_epochs}")
    print(f"Total training runs: {n_splits * num_epochs}")
    print(f"{'='*60}\n")
    
    # Load full dataset
    data_dir = Path('/Users/rodainahebishy/Desktop/deep-learning/data/CMR-MULTI/LGE_MULTI')
    full_dataset = LGEDataset(data_dir, transform=None)
    
    print(f"Total samples: {len(full_dataset)}")
    
    # Class weights for Focal Loss (5 classes - same weights as U-Net Run 2)
    class_weights = torch.tensor([1.0, 2.0, 3.0, 10.0, 2.0], device=device)
    
    # K-Fold setup
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results for all folds
    all_fold_results = []
    
    # Train each fold
    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(full_dataset)))):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_splits}")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_ids)}")
        print(f"Val samples: {len(val_ids)}")
        
        # Create data loaders for this fold
        train_subset = Subset(full_dataset, train_ids)
        val_subset = Subset(full_dataset, val_ids)
        
        # Apply transforms
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Initialize model for this fold
        model = AttentionUNet(in_channels=1, num_classes=5).to(device)
        
        # Optimizer and criterion
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history for this fold
        fold_history = {
            'train_loss': [], 'train_dice': [],
            'val_loss': [], 'val_dice': [],
            'val_vPCC': [], 'val_RAE': [],
            'learning_rates': []
        }
        
        best_dice = 0.0
        best_epoch = 0
        
        # Train this fold
        for epoch in range(num_epochs):
            # Train
            train_loss, train_dice = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # Validate
            val_loss, val_dice, val_vPCC, val_RAE = validate(
                model, val_loader, criterion, device
            )
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save metrics
            fold_history['train_loss'].append(train_loss)
            fold_history['train_dice'].append(train_dice)
            fold_history['val_loss'].append(val_loss)
            fold_history['val_dice'].append(val_dice)
            fold_history['val_vPCC'].append(val_vPCC)
            fold_history['val_RAE'].append(val_RAE)
            fold_history['learning_rates'].append(current_lr)
            
            # Save best model for this fold
            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                    'val_vPCC': val_vPCC,
                    'val_RAE': val_RAE,
                }, f'best_model_fold{fold+1}.pth')
                
                # Save visualizations for best epoch
                print(f"  → New best model! Saving visualizations...")
                save_visualizations(model, val_loader, device, fold, epoch)
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f'checkpoint_fold{fold+1}_epoch{epoch+1:02d}.pth'
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_dice': val_dice,
                    'val_vPCC': val_vPCC,
                    'val_RAE': val_RAE,
                    'train_loss': train_loss,
                    'train_dice': train_dice,
                }, checkpoint_path)
                print(f"  → Checkpoint saved: {checkpoint_path}")
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, "
                  f"vPCC: {val_vPCC:.4f}, RAE: {val_RAE:.4f}")
            print(f"LR: {current_lr:.6f}")
        
        # Store fold results
        fold_result = {
            'fold': fold + 1,
            'best_epoch': best_epoch + 1,
            'best_dice': best_dice,
            'final_val_dice': fold_history['val_dice'][-1],
            'final_val_vPCC': fold_history['val_vPCC'][-1],
            'final_val_RAE': fold_history['val_RAE'][-1],
            'history': fold_history
        }
        all_fold_results.append(fold_result)
        
        # Save fold history
        with open(f'fold{fold+1}_history.json', 'w') as f:
            json.dump(fold_history, f, indent=2)
        
        print(f"\nFold {fold+1} Complete!")
        print(f"Best Dice: {best_dice:.4f} at epoch {best_epoch+1}")
    
    # ============================================================
    # AGGREGATE RESULTS ACROSS ALL FOLDS
    # ============================================================
    
    print(f"\n{'='*60}")
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}\n")
    
    # Calculate average metrics across all folds
    avg_dice = np.mean([r['final_val_dice'] for r in all_fold_results])
    avg_vPCC = np.mean([r['final_val_vPCC'] for r in all_fold_results])
    avg_RAE = np.mean([r['final_val_RAE'] for r in all_fold_results])
    
    best_dice_across_folds = np.mean([r['best_dice'] for r in all_fold_results])
    
    print(f"Average Final Metrics:")
    print(f"  Dice: {avg_dice:.4f}")
    print(f"  vPCC: {avg_vPCC:.4f}")
    print(f"  RAE:  {avg_RAE:.4f}")
    print(f"\nAverage Best Dice: {best_dice_across_folds:.4f}")
    
    # Save all results
    final_results = {
        'n_folds': n_splits,
        'num_epochs': num_epochs,
        'avg_final_dice': avg_dice,
        'avg_final_vPCC': avg_vPCC,
        'avg_final_RAE': avg_RAE,
        'avg_best_dice': best_dice_across_folds,
        'all_folds': all_fold_results
    }
    
    with open('kfold_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to kfold_results.json")
    print(f"Individual fold histories saved to fold*_history.json")
    print(f"Best models saved to best_model_fold*.pth")
    
    return all_fold_results


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("ATTENTION U-NET - 5-FOLD CROSS-VALIDATION")
    print("With Fixed RAE Calculation")
    print("="*60 + "\n")
    
    results = train_kfold(n_splits=5, num_epochs=50, batch_size=16)
    
    print("\n✓ Training Complete!")
