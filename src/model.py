"""
Voxel-based Deep Regression Model for Body Composition Estimation
Matches the architecture described in the SN Computer Science paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard residual block with two 3x3 convolutions"""
    
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        return self.relu(out)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: global average pooling
        y = self.pool(x).view(b, c)
        # Excitation: FC layers
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: element-wise multiplication
        return x * y.expand_as(x)


class VoxelDemographicNet(nn.Module):
    """
    End-to-end voxel-based network for body composition prediction
    
    Args:
        num_outputs: Number of body composition targets (default: 66)
        use_demographics: Whether to fuse demographic features (ablation)
        use_residual: Whether to use residual blocks (ablation)
        use_se: Whether to use SE attention (ablation)
    """
    
    def __init__(
        self, 
        num_outputs=66,
        use_demographics=True,
        use_residual=True,
        use_se=True
    ):
        super().__init__()
        self.use_demographics = use_demographics
        self.use_residual = use_residual
        self.use_se = use_se
        
        # Stem: Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Encoder Stage 1: 64 channels
        stage1_layers = []
        if use_residual:
            stage1_layers.append(ResidualBlock(64))
        if use_se:
            stage1_layers.append(SEBlock(64, reduction=16))
        stage1_layers.extend([
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        ])
        self.stage1 = nn.Sequential(*stage1_layers)
        
        # Encoder Stage 2: 128 channels
        stage2_layers = []
        if use_residual:
            stage2_layers.append(ResidualBlock(128))
        if use_se:
            stage2_layers.append(SEBlock(128, reduction=16))
        stage2_layers.extend([
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        ])
        self.stage2 = nn.Sequential(*stage2_layers)
        
        # Encoder Stage 3: 256 channels
        stage3_layers = []
        if use_residual:
            stage3_layers.append(ResidualBlock(256))
        if use_se:
            stage3_layers.append(SEBlock(256, reduction=16))
        stage3_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.stage3 = nn.Sequential(*stage3_layers)
        
        # Demographic embedding
        if use_demographics:
            self.demo_embedding = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True)
            )
            fusion_dim = 256 * 4 * 4 + 64  # 4096 + 64 = 4160
        else:
            fusion_dim = 256 * 4 * 4  # 4096
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)
        )
    
    def forward(self, voxel_map, demographics=None):
        """
        Args:
            voxel_map: Tensor of shape [B, 3, 512, 512]
            demographics: Tensor of shape [B, 4] (height, weight, sex, ethnicity)
        
        Returns:
            predictions: Tensor of shape [B, 66]
        """
        # Voxel feature extraction
        x = self.stem(voxel_map)        # [B, 64, 256, 256]
        x = self.stage1(x)              # [B, 128, 128, 128]
        x = self.stage2(x)              # [B, 256, 64, 64]
        x = self.stage3(x)              # [B, 256, 4, 4]
        x = x.flatten(1)                # [B, 4096]
        
        # Demographic fusion
        if self.use_demographics and demographics is not None:
            demo_feat = self.demo_embedding(demographics)  # [B, 64]
            x = torch.cat([x, demo_feat], dim=1)           # [B, 4160]
        
        # Prediction
        return self.classifier(x)  # [B, 66]


if __name__ == "__main__":
    # Test model
    model = VoxelDemographicNet(
        num_outputs=66,
        use_demographics=True,
        use_residual=True,
        use_se=True
    )
    
    # Dummy input
    voxel = torch.randn(2, 3, 512, 512)
    demo = torch.randn(2, 4)
    
    # Forward pass
    output = model(voxel, demo)
    print(f"Input shape: {voxel.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
