import torch
import torch.nn as nn
import torch.nn.functional as F


class Action(nn.Module):
    def __init__(self, in_channels):
        super(Action, self).__init__()

        self.in_channels = in_channels
        self.reduced_channels = self.in_channels//4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fold = self.in_channels // 4

        # motion excitation
        self.pad = (0,0,0,0,0,0,0,1)
        self.action_p3_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p3_bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.action_p3_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 3), 
                                    stride=(1 ,1), bias=False, padding=(1, 1), groups=self.reduced_channels)
        self.action_p3_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))


    def forward(self, x):
        b, c, t, h, w = x.shape # [B, C, T, H, W]
        nt = b * t

        # Squeeze
        x_reshaped = x.transpose(1, 2).reshape(nt, c, h, w) 
        x3 = self.action_p3_squeeze(x_reshaped)
        x3 = self.action_p3_bn1(x3)

        # [B, T, C', H, W]
        _, c_new, h_new, w_new = x3.size()
        x3_shifted = x3.view(b, t, c_new, h_new, w_new)

        # Motion Excitation
        x3_plus0, _ = x3_shifted.split([t - 1, 1], dim=1)
        
        x3_conv = self.action_p3_conv1(x3).view(b, t, c_new, h_new, w_new)
        _, x3_plus1 = x3_conv.split([1, t - 1], dim=1) 

        # Motion feature
        x_p3 = x3_plus1 - x3_plus0

        # Padding
        zero_padding = torch.zeros(b, 1, c_new, h_new, w_new, device=x.device, dtype=x.dtype)
        x_p3 = torch.cat([x_p3, zero_padding], dim=1)  

        # Channel Expansion
        x_p3 = x_p3.view(nt, c_new, h_new, w_new)
        x_p3 = self.avg_pool(x_p3)
        x_p3 = self.action_p3_expand(x_p3)
        x_p3 = self.sigmoid(x_p3) 

        # Residual connection
        x_p3 = x_reshaped * x_p3 + x_reshaped

        # [B, C, T, H, W]
        x_p3 = x_p3.view(b, t, c, h, w).transpose(1, 2).contiguous()

        return x_p3
