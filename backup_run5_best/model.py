#!/usr/bin/env python3
"""
U-Net Model for Medical Image Segmentation
Optimized for Apple Silicon (M4)
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module): 
    """(Conv2D -> BatchNorm -> ReLU) x 2"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """U-Net Architecture for Medical Image Segmentation"""
    
    def __init__(self, in_channels=1, num_classes=5):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale MRI)
            num_classes: Number of segmentation classes
                0: Background
                1: LV Cavity
                2: LV Myocardium
                3: Myocardial Scar
                4: RV Cavity
        """
        super().__init__()
        
        # Encoder (downsampling path)
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 = 512 (upconv) + 512 (skip)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)  # 512 = 256 (upconv) + 256 (skip)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)  # 256 = 128 (upconv) + 128 (skip)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)  # 128 = 64 (upconv) + 64 (skip)
        
        # Final classifier
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1(x)          # 64 channels
        enc2 = self.enc2(self.pool1(enc1))  # 128 channels
        enc3 = self.enc3(self.pool2(enc2))  # 256 channels
        enc4 = self.enc4(self.pool3(enc3))  # 512 channels
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))  # 1024 channels
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Concatenate skip connection
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output segmentation map
        out = self.out(dec1)
        
        return out

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("Testing U-Net model...\n")
    
    # Create model
    model = UNet(in_channels=1, num_classes=5)
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256)  # Batch of 2 images
    print(f"Input shape: {x.shape}")
    
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Expected: torch.Size([2, 5, 256, 256])")
    
    # Count parameters
    params = count_parameters(model)
    print(f"\nTotal trainable parameters: {params:,}")
    print(f"Model size: ~{params * 4 / 1024**2:.1f} MB (fp32)")
    
    # Test on MPS device
    if torch.backends.mps.is_available():
        print("\n✅ MPS (Apple Silicon GPU) available")
        device = torch.device("mps")
        model = model.to(device)
        x = x.to(device)
        
        with torch.no_grad():
            y = model(x)
        print(f"GPU inference successful! Output on {y.device}")
    else:
        print("\n⚠️  MPS not available, using CPU")
    
    print("\n✅ Model test passed!")
