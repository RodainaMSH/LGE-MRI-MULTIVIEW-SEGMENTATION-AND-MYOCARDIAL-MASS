
#!/usr/bin/env python3
"""
Complete Training Script with ALL Metrics + Segmentation Visualization
Includes: DSC (Dice), Mass vPCC, Mass RAE + saves segmentation images
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
import nibabel as nib

from model import UNet
from dataset import LGEDataset, train_transform, val_transform

# ============================================================
# FOCAL LOSS IMPLEMENTATION
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses on hard-to-classify pixels (like scar boundaries)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
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
# METRIC FUNCTIONS
# ============================================================

def dice_coefficient(pred, target, num_classes, smooth=1e-8):
    """
    Calculate Dice Similarity Coefficient (DSC)
    Main segmentation metric - measures overlap between prediction and ground truth
    
    Returns: Mean Dice across classes 1-4 (excluding background)
    """
    dice_scores = []
    pred = torch.argmax(pred, dim=1)  # (B, H, W)
    
    for cls in range(1, num_classes):  # Skip background (class 0)
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        if union == 0:
            dice = 1.0  # Perfect score if both are empty
        else:
            dice = ((2.0 * intersection + smooth) / (union + smooth)).item()
        
        dice_scores.append(dice)
    
    return np.mean(dice_scores)


def calculate_myocardial_mass(segmentation, pixel_spacing=(1.0, 1.0), slice_thickness=10.0):
    """
    Calculate myocardial mass from segmentation
    
    Args:
        segmentation: 3D numpy array of segmentation (H, W, D)
        pixel_spacing: (x, y) spacing in mm
        slice_thickness: z spacing in mm
    
    Returns:
        mass in grams
    """
    # Count myocardial pixels (class 2 = LV Myocardium)
    myocardial_pixels = (segmentation == 2).sum()
    
    # Calculate volume in mm³
    pixel_volume = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
    volume_mm3 = myocardial_pixels * pixel_volume
    
    # Convert to cm³ and then to grams (density = 1.05 g/cm³)
    volume_cm3 = volume_mm3 / 1000.0
    mass_g = volume_cm3 * 1.05
    
    return mass_g


def calculate_scar_mass(segmentation, pixel_spacing=(1.0, 1.0), slice_thickness=10.0):
    """
    Calculate scar mass from segmentation
    
    Args:
        segmentation: 3D numpy array of segmentation (H, W, D)
        pixel_spacing: (x, y) spacing in mm
        slice_thickness: z spacing in mm
    
    Returns:
        mass in grams
    """
    # Count scar pixels (class 3 = Myocardial Scar)
    scar_pixels = (segmentation == 3).sum()
    
    # Calculate volume in mm³
    pixel_volume = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
    volume_mm3 = scar_pixels * pixel_volume
    
    # Convert to cm³ and then to grams (density = 1.05 g/cm³)
    volume_cm3 = volume_mm3 / 1000.0
    mass_g = volume_cm3 * 1.05
    
    return mass_g


def calculate_clinical_metrics(pred_masses, gt_masses):
    """
    Calculate clinical metrics: vPCC and RAE
    
    Args:
        pred_masses: List of predicted scar masses
        gt_masses: List of ground truth scar masses
    
    Returns:
        vPCC: Volumetric Pearson Correlation Coefficient (-1 to 1, higher is better)
        RAE: Relative Absolute Error (lower is better)
    """
    # Filter out patients with zero ground truth mass for RAE calculation
    valid_indices = [i for i, gt in enumerate(gt_masses) if gt > 0]
    
    if len(valid_indices) == 0:
        return 0.0, float('inf')
    
    # vPCC: Pearson correlation between all predicted and ground truth masses
    if len(pred_masses) > 1:
        vPCC, _ = pearsonr(pred_masses, gt_masses)
    else:
        vPCC = 0.0
    
    # RAE: Mean relative absolute error for patients with non-zero ground truth
    rae_values = []
    for i in valid_indices:
        pred = pred_masses[i]
        gt = gt_masses[i]
        rae = abs(pred - gt) / abs(gt)
        rae_values.append(rae)
    
    RAE = np.mean(rae_values) if rae_values else float('inf')
    
    return vPCC, RAE


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def save_segmentation_visualization(images, labels, predictions, filenames, save_dir, epoch):
    """
    Save segmentation visualization comparing ground truth vs predictions
    Uses original dataset filenames for easy identification
    
    Creates images showing: Input | Original GT (grayscale) | Colored GT | Prediction
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Color map for segmentation classes
    colors = {
        0: [0, 0, 0],        # Background - Black
        1: [255, 0, 0],      # LV Cavity - Red
        2: [0, 255, 0],      # LV Myocardium - Green
        3: [255, 255, 0],    # Myocardial Scar - Yellow
        4: [0, 0, 255]       # RV Cavity - Blue
    }
    
    for i in range(len(images)):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Input image
        img = images[i].cpu().numpy()[0]  # Remove channel dimension
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Input LGE MRI', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Original Ground Truth (grayscale - showing raw label values)
        gt = labels[i].cpu().numpy()
        axes[1].imshow(gt, cmap='gray', vmin=0, vmax=4)
        axes[1].set_title('Original GT\n(Labels 0-4)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Colored Ground Truth
        gt_colored = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
        for cls, color in colors.items():
            gt_colored[gt == cls] = color
        axes[2].imshow(gt_colored)
        axes[2].set_title('Colored GT', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Prediction
        pred = predictions[i].cpu().numpy()
        pred_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cls, color in colors.items():
            pred_colored[pred == cls] = color
        axes[3].imshow(pred_colored)
        axes[3].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[3].axis('off')
        
        # Use original filename from dataset
        original_name = filenames[i]
        plt.suptitle(f'Epoch {epoch} - {original_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save with original filename
        save_path = save_dir / f'epoch{epoch:02d}_{original_name}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]")
    for images, labels, filenames in pbar:  # Now includes filenames
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            dice = dice_coefficient(outputs, labels, num_classes=5)
        
        total_loss += loss.item()
        total_dice += dice
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    return avg_loss, avg_dice


@torch.no_grad()
def validate_with_metrics(model, dataloader, criterion, device, epoch, save_dir='visualizations'):
    """
    Validate with ALL metrics: DSC, Mass vPCC, Mass RAE
    Also saves segmentation visualizations with original filenames
    """
    model.eval()
    total_loss = 0
    total_dice = 0
    
    # For clinical metrics
    pred_scar_masses = []
    gt_scar_masses = []
    
    # For visualization - collect ALL samples
    all_images = []
    all_labels = []
    all_predictions = []
    all_filenames = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Val]   ")
    for batch_idx, (images, labels, filenames) in enumerate(pbar):  # Now includes filenames
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        dice = dice_coefficient(outputs, labels, num_classes=5)
        
        total_loss += loss.item()
        total_dice += dice
        
        # Get predictions
        predictions = torch.argmax(outputs, dim=1)
        
        # Calculate scar masses for this batch
        for i in range(len(images)):
            pred_seg = predictions[i].cpu().numpy()
            gt_seg = labels[i].cpu().numpy()
            
            # Calculate scar mass (class 3)
            pred_mass = calculate_scar_mass(pred_seg[..., np.newaxis])
            gt_mass = calculate_scar_mass(gt_seg[..., np.newaxis])
            
            pred_scar_masses.append(pred_mass)
            gt_scar_masses.append(gt_mass)
        
        # Collect ALL samples for visualization
        all_images.extend([img.cpu() for img in images])
        all_labels.extend([lbl.cpu() for lbl in labels])
        all_predictions.extend([pred.cpu() for pred in predictions])
        all_filenames.extend(filenames)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    # Calculate clinical metrics
    vPCC, RAE = calculate_clinical_metrics(pred_scar_masses, gt_scar_masses)
    
    # Save visualization for ALL validation samples
    save_segmentation_visualization(all_images, all_labels, all_predictions, all_filenames, save_dir, epoch)
    
    return avg_loss, avg_dice, vPCC, RAE


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():
    # Check for Apple Silicon GPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠️  MPS not available, using CPU")
    
    # Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 50  # Increased from 30 for better convergence
    WARMUP_EPOCHS = 5  # LR warm-up period
    VIEWS = ['SAX', '2CH', '4CH', 'RAS']  # Train on all 4 views
    NUM_WORKERS = 0
    
    # Class weights: [Background, LV Cavity, LV Myo, Scar, RV Cavity]
    # Scar gets 10x weight because it's tiny and hardest to detect
    CLASS_WEIGHTS = torch.tensor([1.0, 2.0, 2.0, 10.0, 2.0]).to(device)
    USE_FOCAL_LOSS = True  # Use Focal Loss instead of CrossEntropy
    
    print(f"\n{'='*70}")
    print(f"IMPROVED TRAINING - MULTI-VIEW (Run 2)")
    print(f"{'='*70}")
    print(f"Metrics: DSC (Dice) + Mass vPCC + Mass RAE")
    print(f"Views: {', '.join(VIEWS)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Base LR: {LEARNING_RATE} (with {WARMUP_EPOCHS}-epoch warmup)")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Loss: {'Focal Loss' if USE_FOCAL_LOSS else 'CrossEntropy'}")
    print(f"Class weights: {CLASS_WEIGHTS.cpu().tolist()}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Load dataset - ALL VIEWS
    full_dataset = LGEDataset(
        data_dir='data/CMR-MULTI/LGE_MULTI',
        views=VIEWS,  # Load all 4 views together
        transform=train_transform
    )
    
    # Split 80-20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_dataset.dataset.transform = val_transform
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Model
    model = UNet(in_channels=1, num_classes=5).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss, optimizer, scheduler
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=CLASS_WEIGHTS, gamma=2.0)
        print("✅ Using Focal Loss with class weights")
    else:
        criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
        print("✅ Using CrossEntropy with class weights")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training history
    history = {
        'train_loss': [], 'train_dice': [],
        'val_loss': [], 'val_dice': [],
        'val_vPCC': [], 'val_RAE': [],
        'learning_rates': []
    }
    
    # Training loop
    best_dice = 0.0
    start_time = time.time()
    
    print("🚀 Starting training...\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Learning rate warm-up for first WARMUP_EPOCHS
        if epoch <= WARMUP_EPOCHS:
            warmup_lr = LEARNING_RATE * (epoch / WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            current_lr = warmup_lr
            print(f"🔥 Warm-up: LR = {warmup_lr:.2e}")
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate with ALL metrics
        val_loss, val_dice, val_vPCC, val_RAE = validate_with_metrics(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_vPCC'].append(val_vPCC)
        history['val_RAE'].append(val_RAE)
        history['learning_rates'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS} SUMMARY")
        print(f"{'='*70}")
        print(f"Training:    Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")
        print(f"Validation:  Loss: {val_loss:.4f} | Dice: {val_dice:.4f}")
        print(f"Clinical:    Mass vPCC: {val_vPCC:.4f} | Mass RAE: {val_RAE:.4f}")
        print(f"Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_vPCC': val_vPCC,
                'val_RAE': val_RAE,
            }, 'best_model_multiview.pth', _use_new_zipfile_serialization=True)
            print(f"✅ SAVED BEST MODEL (Dice: {val_dice:.4f})")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_vPCC': val_vPCC,
                'val_RAE': val_RAE,
                'best_dice': best_dice,
            }, f'checkpoint_epoch{epoch:02d}.pth', _use_new_zipfile_serialization=True)
            print(f"💾 Saved checkpoint at epoch {epoch}")
        
        print(f"Best Dice: {best_dice:.4f}")
        print(f"{'='*70}\n")
    
    total_time = time.time() - start_time
    
    # Save history
    with open('training_history_multiview.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Final vPCC: {history['val_vPCC'][-1]:.4f}")
    print(f"Final RAE: {history['val_RAE'][-1]:.4f}")
    print(f"\nSaved: best_model_multiview.pth")
    print(f"Saved: training_history_multiview.json")
    print(f"Saved: visualizations/ folder with segmentation images")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
