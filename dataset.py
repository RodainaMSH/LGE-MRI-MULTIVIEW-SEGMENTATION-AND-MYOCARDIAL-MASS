#!/usr/bin/env python3
"""
LGE MRI Dataset Loader
Handles NIfTI format medical images with data augmentation
"""

import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LGEDataset(Dataset):
    """Dataset for LGE MRI segmentation - Multi-View"""
    
    def __init__(self, data_dir, views=['SAX', '2CH', '4CH', 'RAS'], transform=None, split='train'):
        """
        Args:
            data_dir: Path to LGE_MULTI folder
            views: List of views to load (default: all 4 views)
            transform: Albumentations transforms
            split: 'train' or 'val' (for future use)
        """
        self.data_dir = Path(data_dir)
        self.views = views if isinstance(views, list) else [views]
        self.transform = transform
        self.split = split
        
        # Build list of all 2D slices from ALL views
        self.slices = []
        print(f"Loading multi-view dataset...")
        
        for view in self.views:
            # Get all image files for this view
            image_dir = self.data_dir / f"{view}_TR" / "image"
            label_dir = self.data_dir / f"{view}_TR" / "anno"
            
            image_files = sorted(list(image_dir.glob("*.nii.gz")))
            label_files = sorted(list(label_dir.glob("*.nii.gz")))
            
            assert len(image_files) == len(label_files), \
                f"Mismatch in {view}: {len(image_files)} images, {len(label_files)} labels"
            
            print(f"  Loading {view}...")
            for img_file, label_file in zip(image_files, label_files):
                # Load volume to count slices
                img_nii = nib.load(str(img_file))
                num_slices = img_nii.shape[2]
                
                # Store (img_file, label_file, slice_idx, view_name) tuples
                for slice_idx in range(num_slices):
                    self.slices.append((img_file, label_file, slice_idx, view))
            
            print(f"  ✅ {view}: {len(image_files)} volumes")
        
        print(f"\n✅ Total: {len(self.slices)} slices from {len(self.views)} views")
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        img_file, label_file, slice_idx, view_name = self.slices[idx]
        
        # Load NIfTI files
        img_nii = nib.load(str(img_file))
        label_nii = nib.load(str(label_file))
        
        # Get 3D volumes
        img_volume = img_nii.get_fdata()
        label_volume = label_nii.get_fdata()
        
        # Extract 2D slice
        img_2d = img_volume[:, :, slice_idx].astype(np.float32)
        label_2d = label_volume[:, :, slice_idx].astype(np.int64)
        
        # Normalize image to [0, 1]
        if img_2d.max() > img_2d.min():
            img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min())
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=img_2d, mask=label_2d)
            img_2d = augmented['image']
            label_2d = augmented['mask']
        
        # Convert to torch tensors
        img_tensor = torch.from_numpy(img_2d).float().unsqueeze(0)  # (1, H, W)
        label_tensor = torch.from_numpy(label_2d).long()  # (H, W)
        
        # Create filename identifier: view_originalname_slice
        original_name = img_file.stem.replace('.nii', '')  # Remove .nii extension
        filename = f"{view_name}_{original_name}_slice{slice_idx:03d}"
        
        return img_tensor, label_tensor, filename

# Data augmentation for training (aggressive for better generalization)
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=20, p=0.5, border_mode=0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
])

# No augmentation for validation (only resize)
val_transform = A.Compose([
    A.Resize(256, 256),
])

if __name__ == "__main__":
    # Test the dataset
    print("Testing Multi-View LGEDataset...")
    
    dataset = LGEDataset(
        data_dir='data/CMR-MULTI/LGE_MULTI',
        views=['SAX', '2CH', '4CH', 'RAS'],  # Load all 4 views
        transform=train_transform
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Load one sample from each view
    for i in [0, 100, 200, 300]:
        if i < len(dataset):
            img, label, filename = dataset[i]
            print(f"\nSample {i}: {filename}")
            print(f"  Image shape: {img.shape}")
            print(f"  Label shape: {label.shape}")
            print(f"  Unique labels: {torch.unique(label).tolist()}")
    
    print("\n✅ Multi-view dataset test passed!")
