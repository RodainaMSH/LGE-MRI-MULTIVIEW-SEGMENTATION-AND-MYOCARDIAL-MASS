#!/usr/bin/env python3
"""
U-Net Training with Corrected Pixel Spacing - Run 6
Based on Run 2's successful architecture but with proper mass calculation

Run 2 Results (with bug):
- Dice: 0.7568
- vPCC: 0.8849
- RAE: 0.4820 (but used hardcoded 12.5 mm³)

Goal: Achieve similar Dice/vPCC with accurate RAE using correct pixel spacing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

from model import UNet
from dataset import LGEDataset, train_transform, val_transform


# ============================================================
# FOCAL LOSS IMPLEMENTATION (Same as Run 2)
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
# MASS CALCULATION WITH CORRECTED PIXEL SPACING
# ============================================================

def calculate_scar_mass(mask, pixel_area, slice_thickness):
    """
    Calculate scar mass from segmentation mask with CORRECTED pixel spacing
    
    Args:
        mask: segmentation mask (H, W) with label 3 = scar (for 5-class)
        pixel_area: area of one pixel in mm² (from NIfTI header: pixdim[1] × pixdim[2])
        slice_thickness: thickness of slice in mm (from NIfTI header: pixdim[3])
    
    Returns:
        mass: scar mass in grams
    """
    # Use label 3 (SCAR) for 5-class segmentation
    scar_voxels = (mask == 3).sum()
    
    # Calculate volume in mm³ using ACTUAL spacing
    volume = scar_voxels * pixel_area * slice_thickness
    
    # Convert to mass (assuming density = 1.05 g/cm³ = 0.00105 g/mm³)
    density = 0.00105  # g/mm³
    mass = volume * density
    
    return mass


# ============================================================
# DICE COEFFICIENT CALCULATION
# ============================================================

def calculate_dice(pred, target, num_classes=5):
    """Calculate multi-class Dice coefficient (excluding background)"""
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
# VISUALIZATION FUNCTION
# ============================================================

def save_visualizations(model, val_loader, device, epoch, save_dir='visualizations_run6'):
    """
    Save visualizations organized by view: original image, GT, prediction
    Saves 8 random samples from validation set
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create view-specific directories
    for view in ['SAX_TR', '2CH_TR', '4CH_TR', 'RAS_TR']:
        os.makedirs(os.path.join(save_dir, view), exist_ok=True)
    
    model.eval()
    
    # Color mapping (5 classes)
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
        # Get 8 random samples
        sample_indices = np.random.choice(len(val_loader.dataset), min(8, len(val_loader.dataset)), replace=False)
        
        for idx, sample_idx in enumerate(sample_indices):
            # Get single sample with metadata
            image, mask, filename, pixel_area, slice_thickness = val_loader.dataset[sample_idx]
            
            # Add batch dimension
            image_input = image.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(image_input)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            # Convert tensors to numpy
            image_np = image.squeeze().cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            # Calculate masses with CORRECTED spacing
            pred_mass = calculate_scar_mass(pred, pixel_area, slice_thickness)
            gt_mass = calculate_scar_mass(mask_np, pixel_area, slice_thickness)
            
            # Create colored versions
            gt_colored = mask_to_color(mask_np)
            pred_colored = mask_to_color(pred)
            
            # Extract view name from filename
            view_name = filename.split('_')[0] + '_TR'
            
            # Create figure
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Add title with mass information
            fig.suptitle(f'Epoch {epoch+1} - {filename}\nGT Mass: {gt_mass:.3f}g | Pred Mass: {pred_mass:.3f}g | Error: {abs(pred_mass-gt_mass):.3f}g', 
                        fontsize=10, y=1.05)
            
            # 1. Original image
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
            
            # Save to view-specific folder
            save_path = os.path.join(save_dir, view_name, f'epoch{epoch+1:02d}_sample{idx+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  → Saved {len(sample_indices)} visualizations to {save_dir}/")


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, warmup_epochs=5):
    """Train for one epoch with learning rate warmup"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    # Learning rate warmup (same as Run 2)
    if epoch < warmup_epochs:
        lr_scale = (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003 * lr_scale  # Run 2 used 3e-4 base LR
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    for batch in pbar:
        # Unpack batch with metadata
        images, masks, filenames, pixel_areas, slice_thicknesses = batch
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
    """Validate model with all metrics including CORRECTED mass calculation"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    all_pred_masses = []
    all_gt_masses = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            # Unpack batch with metadata
            images, masks, filenames, pixel_areas, slice_thicknesses = batch
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate Dice
            dice = calculate_dice(preds, masks)
            running_loss += loss.item()
            running_dice += dice
            
            # Collect mass predictions with CORRECTED spacing
            for i in range(len(preds)):
                pred = preds[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                pixel_area = pixel_areas[i].item()
                slice_thickness = slice_thicknesses[i].item()
                
                pred_mass = calculate_scar_mass(pred, pixel_area, slice_thickness)
                gt_mass = calculate_scar_mass(mask, pixel_area, slice_thickness)
                
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


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train(num_epochs=50, batch_size=16):
    """
    Train U-Net with corrected pixel spacing
    Same configuration as Run 2 but with proper mass calculation
    """
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print(f"U-NET TRAINING WITH CORRECTED PIXEL SPACING - RUN 6")
    print(f"{'='*60}")
    print(f"Architecture: U-Net (same as Run 2)")
    print(f"Loss: Focal Loss (gamma=2.0)")
    print(f"Class Weights: [1.0, 2.0, 2.0, 10.0, 2.0]")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Base LR: 3e-4 (same as Run 2)")
    print(f"Pixel Spacing: CORRECTED (from NIfTI headers)")
    print(f"{'='*60}\n")
    
    # Load dataset
    data_dir = Path('/Users/rodainahebishy/Desktop/deep-learning/data/CMR-MULTI/LGE_MULTI')
    full_dataset = LGEDataset(data_dir, transform=None)
    
    print(f"Total samples: {len(full_dataset)}")
    
    # Split dataset (80/20, same as Run 2)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}\n")
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = UNet(in_channels=1, num_classes=5).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Class weights for Focal Loss (SAME AS RUN 2)
    class_weights = torch.tensor([1.0, 2.0, 2.0, 10.0, 2.0], device=device)
    
    # Loss and optimizer (SAME AS RUN 2)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_dice': [],
        'val_loss': [], 'val_dice': [],
        'val_vPCC': [], 'val_RAE': [],
        'learning_rates': []
    }
    
    best_dice = 0.0
    best_epoch = 0
    start_time = time.time()
    
    # Training loop
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
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_vPCC'].append(val_vPCC)
        history['val_RAE'].append(val_RAE)
        history['learning_rates'].append(current_lr)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_vPCC': val_vPCC,
                'val_RAE': val_RAE,
            }, 'best_model_run6.pth')
            
            print(f"  → New best model! Saving visualizations...")
            save_visualizations(model, val_loader, device, epoch)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'checkpoint_run6_epoch{epoch+1:02d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_dice': val_dice,
                'val_vPCC': val_vPCC,
                'val_RAE': val_RAE,
            }, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, "
              f"vPCC: {val_vPCC:.4f}, RAE: {val_RAE:.4f}")
        print(f"LR: {current_lr:.6f}")
        print(f"Best Dice: {best_dice:.4f} at epoch {best_epoch+1}")
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Best Dice: {best_dice:.4f} at epoch {best_epoch+1}")
    print(f"Final vPCC: {history['val_vPCC'][best_epoch]:.4f}")
    print(f"Final RAE: {history['val_RAE'][best_epoch]:.4f}")
    
    # Save training history
    with open('training_history_run6.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nResults saved:")
    print(f"  - best_model_run6.pth")
    print(f"  - training_history_run6.json")
    print(f"  - visualizations_run6/")
    
    return history


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("U-NET WITH CORRECTED PIXEL SPACING - RUN 6")
    print("Based on Run 2's Architecture")
    print("="*60 + "\n")
    
    history = train(num_epochs=50, batch_size=16)
    
    print("\n✓ Training Complete!")
