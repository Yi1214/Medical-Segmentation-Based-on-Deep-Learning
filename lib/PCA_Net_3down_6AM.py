import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from help_functions import *
from Adaptive_NL2_Block import Adaptive_NL2_Block2D


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool"""
    def __init__(self):
        super(Down, self).__init__()

        self.maxpool= nn.Sequential(
            nn.MaxPool2d((2, 2))
        )

    def forward(self, x):
        return self.maxpool(x)

class Up(nn.Module):
    """Upscaling with maxpool"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.up(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PCA_Net(nn.Module):
    # initializers
    def __init__(self, scales=[6,3,1,1,2,3]):
        super(PCA_Net, self).__init__()
        self.scales = scales       # s大小，数组形式
        self.PCA1 = Adaptive_NL2_Block2D(32, 16, self.scales[0], sub_sample=False, bn_layer=True, in_layer=False)
        self.PCA2 = Adaptive_NL2_Block2D(64, 32, self.scales[1], sub_sample=False, bn_layer=True, in_layer=False)
        self.PCA3 = Adaptive_NL2_Block2D(128, 64, self.scales[2], sub_sample=False, bn_layer=True, in_layer=False)
        self.PCA4 = Adaptive_NL2_Block2D(256, 128, self.scales[3], sub_sample=False, bn_layer=True, in_layer=False)
        self.PCA5 = Adaptive_NL2_Block2D(256, 128, self.scales[4], sub_sample=False, bn_layer=True, in_layer=False)
        self.PCA6 = Adaptive_NL2_Block2D(256, 128, self.scales[5], sub_sample=False, bn_layer=True, in_layer=False)

        self.doubleConv1 = DoubleConv(1, 32)
        self.down1 = Down()
        self.doubleConv2= DoubleConv(32, 64)
        self.down2 = Down()
        self.doubleConv3 = DoubleConv(64, 128)
        self.down3 = Down()
        self.doubleConv4 = DoubleConv(128, 256)

        self.up1 = Up(256*4, 128)
        self.doubleConv5 = DoubleConv(256, 128)
        self.up2 = Up(128, 64)
        self.doubleConv6 = DoubleConv(128, 64)
        self.up3 = Up(64, 32)
        self.doubleConv7 = DoubleConv(64, 32)
        self.outc = OutConv(32, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):

        CB1 = self.doubleConv1(input)
        CB1= self.PCA1(CB1)
        down1 = self.down1(CB1)

        CB2 = self.doubleConv2(down1)
        CB2 =  self.PCA2(CB2)
        down2 = self.down2(CB2)

        CB3 = self.doubleConv3(down2)
        CB3 = self.PCA3(CB3)
        down3 = self.down3(CB3)

        CB4 = self.doubleConv4(down3)
        PCA4 = self.PCA4(CB4)
        PCA5 = self.PCA5(CB4)
        PCA6 = self.PCA6(CB4)
        multi_head = torch.cat((PCA4, PCA5, PCA6, CB4), dim=1)   # (N,C,H,W) 按channel拼接

        up1 = self.up1(multi_head)
        up1 = torch.cat((CB3, up1), dim=1)
        CB5 = self.doubleConv5(up1)

        up2 = self.up2(CB5)
        up2 = torch.cat((CB2, up2), dim=1)
        CB6 = self.doubleConv6(up2)

        up3 = self.up3(CB6)
        up3 = torch.cat((CB1, up3), dim=1)
        CB7 = self.doubleConv7(up3)

        output = self.outc(CB7)
        output = torch.sigmoid(output)

        return output

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()






# 一个python文件通常有两种使用方法，第一是作为脚本直接执行，
# 第二是 import 到其他的 python 脚本中被调用（模块重用）执行。
# 因此 if __name__ == 'main': 的作用就是控制这两种情况执行代码的过程，
# 在 if __name__ == 'main': 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行，
# 而 import 到其他脚本中是不会被执行的。
if __name__ == '__main__':
    import torch

    # for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:
    '''
    img = torch.zeros(2, 3, 20)
    net = Adaptive_NL2_Block1D(in_channels=3 , inter_channels=2, scale=3, sub_sample=False, bn_layer=True, in_layer=False)
    out = net(img)
    print(out.size())
    '''
    img = torch.zeros(10, 1, 48, 48)
    net = PCA_Net()
    out = net(img)
    print(out.size())

    '''
    img = torch.randn(2, 3, 8, 20, 20)
    net = Adaptive_NL2_Block3D(in_channels=3 , inter_channels=2, scale=3, sub_sample=False, bn_layer=True, in_layer=False)
    #out = net(img)
    out = net.forward(img)
    print(out.size())
    '''