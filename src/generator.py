import torch.nn as nn
import torch


class ResBlock(nn.Module):
    def __init__(self, n_ch, kernel_size):
        super().__init__()
        padding = (kernel_size - 1)//2 
        self.conv1 = nn.Conv2d(in_channels=n_ch, out_channels=n_ch, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm2d(n_ch)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=n_ch, out_channels=n_ch, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(n_ch)
    
    def forward(self, x):
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + x

class UpSampleBlock(nn.Module):
    def __init__(self, scale_factor, in_ch, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch*(scale_factor**2), kernel_size=kernel_size, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out

class Generator(nn.Module):
    def __init__(self, n_res_blocks=16):
        super().__init__()
        self.first_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4) # padding = (kernel_size - 1)//2 
        self.prelu1 = nn.PReLU()

        self.resnet_blocks1 = nn.Sequential(*[
            ResBlock(n_ch=64, kernel_size=3) for _ in range(n_res_blocks)
        ])

        self.dilate_conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.separ_conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # padding = (kernel_size - 1)//2 
        self.prelu2 = nn.PReLU()
        self.bn = nn.BatchNorm2d(128)

        self.resnet_blocks2 = nn.Sequential(*[
            ResBlock(n_ch=128, kernel_size=3) for _ in range(n_res_blocks)
        ])

        self.mid_conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.mid_bn = nn.BatchNorm2d(128)
        self.upsample1 = UpSampleBlock(scale_factor=2, in_ch=128, kernel_size=3)
        self.upsample2 = UpSampleBlock(scale_factor=2, in_ch=128, kernel_size=3)

        self.last_conv = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=9, stride=1, padding=4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.prelu1(self.first_conv(x))
        out = self.resnet_blocks1(x)

        out2 = self.prelu2(self.bn(self.separ_conv(out)))
        out = self.dilate_conv(out)
        
        out2 = out + out2
        out_final = self.resnet_blocks2(out2)
        out = self.mid_bn(self.mid_conv(out_final)) + out2
        
        out = self.upsample1(out)
        out = self.upsample2(out)
        out = self.last_conv(out)
        out = self.tanh(out) # to force it to be in range [-1, 1] like high_img as they will be both fed to vgg for content loss
        return out
