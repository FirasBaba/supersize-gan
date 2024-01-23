import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyrelu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.n_channels_list = [64,64,128,128,256,256,512,512,1024,1024,2048,2048]
        self.conv_list = [ConvBlock(in_ch=self.n_channels_list[i],
                                    out_ch=self.n_channels_list[i+1],
                                    kernel_size=3,
                                    stride=2 - i%2
                                    ) 
                                    for i in range(len(self.n_channels_list) - 1)]
        self.vgg_block = nn.Sequential(*self.conv_list)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flat = nn.Flatten()
        self.linear = nn.Sequential(
            *[nn.Linear(in_features=self.n_channels_list[-1], out_features=1024),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(in_features=1024, out_features=1),]
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.vgg_block(x)
        x = self.avg_pool(x)
        x = self.flat(x)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    model = Discriminator()
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print(y.shape)