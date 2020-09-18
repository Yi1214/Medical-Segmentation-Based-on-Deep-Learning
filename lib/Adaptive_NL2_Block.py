import torch
from torch import nn
from torch.nn import functional as F


class Adaptive_NL2_BlockND(nn.Module):
    def __init__(self, in_channels, inter_channels, scale, dimension=3,
                 sub_sample=True, bn_layer=True, in_layer=False):
        super(Adaptive_NL2_BlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.scale = scale      # multi-scale,可选scale

        if dimension==3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.adaptive_pool_layer = nn.AdaptiveAvgPool3d((1, scale, scale))    # 用于第二个分支的Adaptive Pooling
            bn = nn.BatchNorm3d
            instanceNorm = nn.InstanceNorm3d        # 实例归一化
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.adaptive_pool_layer = nn.AdaptiveAvgPool2d((scale, scale))        # 用于第二个分支的Adaptive Pooling
            bn = nn.BatchNorm2d
            instanceNorm = nn.InstanceNorm2d        # 实例归一化
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            self.adaptive_pool_layer = nn.AdaptiveAvgPool1d(scale)         # 用于第二个分支的Adaptive Pooling
            bn = nn.BatchNorm1d
            instanceNorm = nn.InstanceNorm1d        # 实例归一化

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)     # 只对bn参数初始化，因为nn.conv2d会自动初始化参数
            nn.init.constant_(self.W[1].bias, 0)
        elif in_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,  # channel
                        # 减半，减少计算量: 1024 ->512
                        kernel_size=1, stride=1, padding=0),
                instanceNorm(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)  # 只对bn参数初始化，因为nn.conv2d会自动初始化参数
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=1,
                           kernel_size=1, stride=1, padding=0)
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.alpha = conv_nd(in_channels=self.inter_channels, out_channels=self.scale*self.scale,
                         kernel_size=1, stride=1, padding=0)
        self.beta = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)




        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)

        # phi支路：
        phi_x = self.phi(x).view(batch_size, 1, -1)   # (N, 1, HW)
        # 输出在dim=-1维(最后一维)上的概率分布（即每张图都得到一张只与各自有关的概率分布）
        attention_weight = F.softmax(phi_x, dim=-1).permute(0,2,1).contiguous()   # (N, HW, 1)
        # print("attention_weight:")
        # print(attention_weight.size())
        # theta支路 (N, C1, H, W) --> (N, C1, HW)：
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)   # (N, C1, HW)
        # print("theta_x:")
        # print(theta_x.size())
        # 获得全局上下文信息Context_Information  (N, C1, HW) * (N, HW, 1) = (N, C1, 1):
        # torch.matmul()：第0维若数量相等则不参与计算
        Context_Info = torch.matmul(theta_x, attention_weight)   # (N, C1, 1)
        Context_Local_Info = Context_Info + theta_x        # (N, C1, HW)
        # 恢复原图(N, C1, H, W)
        Context_Local_Info = Context_Local_Info.view(batch_size, self.inter_channels, *x.size()[2:])  # 恢复原图(N, C1, H, W)
        # print("Context_Local_Info:")
        # print(Context_Local_Info.size())
        # alpha路：
        AM = self.alpha(Context_Local_Info)     # (N, s^2, H, W)
        AM = torch.sigmoid(AM).view(batch_size, -1, self.scale*self.scale)   # (N, HW, s^2)

        # g / AveragePooling支路:
        AP = self.adaptive_pool_layer(x)       # (N, C, S, S)
        AP = self.g(AP).view(batch_size, self.inter_channels, -1)        # (N, C1, s^2)
        AP = AP.permute(0, 2, 1).contiguous()          # (N, s^2, C1)

        # AM * AP
        y = torch.matmul(AM, AP)          # (N, HW, C1)
        y = y.permute(0, 2, 1).contiguous()           # (N, C1, HW)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])   # (N, C1, H, W)
        W_y = self.W(y)
        z = W_y + x

        return z


class Adaptive_NL2_Block1D(Adaptive_NL2_BlockND):
    # 构造函数
    def __init__(self, in_channels, inter_channels, scale, sub_sample=False, bn_layer=True, in_layer=False):
        #调用父类构造方法
        super(Adaptive_NL2_Block1D, self).__init__(in_channels, inter_channels, scale, dimension=1,
                 sub_sample=sub_sample, bn_layer=bn_layer, in_layer=in_layer)


class Adaptive_NL2_Block2D(Adaptive_NL2_BlockND):
    def __init__(self, in_channels, inter_channels, scale, sub_sample=False, bn_layer=True, in_layer=False):
        super(Adaptive_NL2_Block2D, self).__init__(in_channels, inter_channels, scale, dimension=2,
                 sub_sample=sub_sample, bn_layer=bn_layer, in_layer=in_layer)


class Adaptive_NL2_Block3D(Adaptive_NL2_BlockND):
    def __init__(self, in_channels, inter_channels, scale, sub_sample=False, bn_layer=True, in_layer=False):
        super(Adaptive_NL2_Block3D, self).__init__(in_channels, inter_channels, scale, dimension=1,
                 sub_sample=sub_sample, bn_layer=bn_layer, in_layer=in_layer)


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
    img = torch.zeros(2, 3, 20, 20)
    net = Adaptive_NL2_Block2D(in_channels=3 , inter_channels=2, scale=3, sub_sample=False, bn_layer=True, in_layer=False)
    out = net.forward(img)
    print(out.size())

    '''
    img = torch.randn(2, 3, 8, 20, 20)
    net = Adaptive_NL2_Block3D(in_channels=3 , inter_channels=2, scale=3, sub_sample=False, bn_layer=True, in_layer=False)
    #out = net(img)
    out = net.forward(img)
    print(out.size())
    '''