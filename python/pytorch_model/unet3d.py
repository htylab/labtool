import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Applies (Conv3D -> GroupNorm -> ReLU) twice."""
    
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """A downsampling module that applies MaxPool3d followed by DoubleConv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)

class Up(nn.Module):
    """An upsampling module that optionally applies trilinear upsampling or transposed convolution, followed by DoubleConv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Out(nn.Module):
    """Final convolution that maps the feature maps to the desired number of classes."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3d(nn.Module):
    """Standard U-Net architecture for volumetric (3D) segmentation."""
    
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        # Define the architecture
        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)
        
        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        # if x has shape [1, 1, 256, 256, 64]
        x1 = self.conv(x)  # Output: [1, n_channels, 256, 256, 64]
        x2 = self.enc1(x1) # Output: [1, 2*n_channels, 128, 128, 32]
        x3 = self.enc2(x2) # Output: [1, 4*n_channels, 64, 64, 16]
        x4 = self.enc3(x3) # Output: [1, 8*n_channels, 32, 32, 8]
        x5 = self.enc4(x4) # Output: [1, 8*n_channels, 16, 16, 4]
        
        # Now we begin the upsampling process, concatenating the outputs with the corresponding downsampled outputs
        mask = self.dec1(x5, x4)   # concat with x4, output: [1, 4*n_channels, 32, 32, 8]
        mask = self.dec2(mask, x3) # concat with x3, output: [1, 2*n_channels, 64, 64, 16]
        mask = self.dec3(mask, x2) # concat with x2, output: [1, n_channels, 128, 128, 32]
        mask = self.dec4(mask, x1) # concat with x1, output: [1, n_channels, 256, 256, 64]
        
        mask = self.out(mask) # Final convolution, output shape: [1, n_classes, 256, 256, 64]
        return mask
