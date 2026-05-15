#!/usr/bin/env python3
"""
Attention U-Net for Medical Image Segmentation
Adds attention gates to focus on relevant features (like myocardial scars)
"""

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """
    Attention Gate Module
    Filters encoder features based on decoder context
    Helps model focus on important regions (scars, boundaries)
    """
    
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of feature maps in gating signal (from decoder)
            F_l: Number of feature maps in encoder features (from skip connection)
            F_int: Number of intermediate feature maps
        """
        super(AttentionGate, self).__init__()
        
        # Transform gating signal (decoder features)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Transform encoder features
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Combine and produce attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (B, F_g, H, W)
            x: Encoder features from skip connection (B, F_l, H, W)
        
        Returns:
            Attention-weighted encoder features (B, F_l, H, W)
        """
        # Transform both inputs
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Combine with ReLU activation
        psi = self.relu(g1 + x1)
        
        # Generate attention coefficients (values between 0 and 1)
        psi = self.psi(psi)
        
        # Apply attention weights to encoder features
        # Broadcasting: (B, 1, H, W) * (B, F_l, H, W) = (B, F_l, H, W)
        return x * psi


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


class AttentionUNet(nn.Module):
    """
    Attention U-Net Architecture
    
    Same as U-Net but with Attention Gates on skip connections
    to focus on important features (small structures like scars)
    """
    
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
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)  # NEW: Attention Gate
        self.dec4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)  # NEW: Attention Gate
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)   # NEW: Attention Gate
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)     # NEW: Attention Gate
        self.dec1 = DoubleConv(128, 64)
        
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
        
        # Decoder with attention-filtered skip connections
        dec4 = self.upconv4(bottleneck)
        enc4_att = self.att4(g=dec4, x=enc4)  # Apply attention to enc4
        dec4 = torch.cat([dec4, enc4_att], dim=1)  # Concatenate with filtered features
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        enc3_att = self.att3(g=dec3, x=enc3)  # Apply attention to enc3
        dec3 = torch.cat([dec3, enc3_att], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        enc2_att = self.att2(g=dec2, x=enc2)  # Apply attention to enc2
        dec2 = torch.cat([dec2, enc2_att], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        enc1_att = self.att1(g=dec1, x=enc1)  # Apply attention to enc1
        dec1 = torch.cat([dec1, enc1_att], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output segmentation map
        out = self.out(dec1)
        
        return out


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing Attention U-Net model...\n")
    
    # Create model
    model = AttentionUNet(in_channels=1, num_classes=5)
    
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
    
    # Compare with U-Net
    print(f"\nComparison:")
    print(f"  U-Net (Run 2):        31,036,741 parameters")
    print(f"  Attention U-Net:      {params:,} parameters")
    print(f"  Difference:           +{params - 31036741:,} parameters")
    print(f"  Increase:             +{((params - 31036741) / 31036741 * 100):.2f}%")
    
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
    
    print("\n✅ Attention U-Net test passed!")
    print("\nKey features:")
    print("  ✓ 4 Attention Gates (one per skip connection)")
    print("  ✓ Filters encoder features based on decoder context")
    print("  ✓ Better focus on small structures (scars)")
    print("  ✓ Same architecture as U-Net + attention mechanism")
