import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
    
class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x
    
class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):

        x = self.up(x)  

        if x.shape[-2] != skip.shape[-2] or x.shape[-1] != skip.shape[-1]:
            diff_y = skip.shape[-2] - x.shape[-2]
            diff_x = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1) 
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    """
    UNet for 2D medical segmentation. 256x256
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 64):
        super().__init__()

        # Encoder
        self.inc   = DoubleConv(in_channels, base_channels)         
        self.down1 = Down(base_channels, base_channels * 2)          
        self.down2 = Down(base_channels * 2, base_channels * 4)      
        self.down3 = Down(base_channels * 4, base_channels * 8)  

        # Bottleneck
        self.down4 = Down(base_channels * 8, base_channels * 16)  

        # Decoder
        self.up1   = Up(base_channels * 16, base_channels * 8)  
        self.up2   = Up(base_channels * 8,  base_channels * 4)  
        self.up3   = Up(base_channels * 4,  base_channels * 2)    
        self.up4   = Up(base_channels * 2,  base_channels)     

        self.outc  = OutConv(base_channels, num_classes)   

    def forward(self, x):
        # Encoder path with skips
        x1 = self.inc(x)      
        x2 = self.down1(x1)   
        x3 = self.down2(x2)   
        x4 = self.down3(x3)   
        x5 = self.down4(x4)   

        # Decoder path 
        x  = self.up1(x5, x4) 
        x  = self.up2(x,  x3) 
        x  = self.up3(x,  x2) 
        x  = self.up4(x,  x1) 

        logits = self.outc(x)
        return logits

import torch
import torch.nn.functional as F

def dice_per_class(pred_logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6):
    B, C, H, W = pred_logits.shape

    # convert logits -> probabilities
    probs = F.softmax(pred_logits, dim=1) 

    target_1hot = F.one_hot(target, num_classes=C)        
    target_1hot = target_1hot.permute(0, 3, 1, 2).float() 

    probs_flat  = probs.view(B, C, -1)        
    target_flat = target_1hot.view(B, C, -1)

    intersection = (probs_flat * target_flat).sum(dim=2)         
    pred_sum     = probs_flat.sum(dim=2)                       
    target_sum   = target_flat.sum(dim=2)                       

    dice = (2 * intersection + epsilon) / (pred_sum + target_sum + epsilon)  

    dice_per_class = dice.mean(dim=0)

    return dice_per_class
