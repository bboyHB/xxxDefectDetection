import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


# https://blog.csdn.net/justsolow/article/details/105971437

class DeformConvUnit(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DeformConvUnit, self).__init__()
        self.offsets = nn.Conv2d(in_channel, 18, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2D(in_channel, out_channel, kernel_size=3, padding=1)
        # self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        # deformable convolution
        offsets = self.offsets(x)
        x = self.deform_conv(x, offsets)
        # x = F.relu(self.deform_conv(x, offsets))
        # x = self.bn(x)
        return x


class DeformNet(nn.Module):
    def __init__(self):
        super(DeformNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.offsets = nn.Conv2d(128, 18, kernel_size=3, padding=1)
        self.conv4 = DeformConv2D(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        # convs
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        # deformable convolution
        offsets = self.offsets(x)
        x = F.relu(self.conv4(x, offsets))
        x = self.bn4(x)
        print(x)

        x = F.avg_pool2d(x, kernel_size=28, stride=1).view(x.size(0), -1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    # 注意，offset的Tensor尺寸是[b, 18, h, w]，offset传入的其实就是每个像素点的坐标偏移，也就是一个坐标量，最终每个点的像素还需要这个坐标偏移和原图进行对应求出。
    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size
        # N=9=3x3
        N = offset.size(1) // 2

        # 这里其实没必要，我们反正这个顺序是我们自己定义的，那我们直接按照[x1, x2, .... y1, y2, ...]定义不就好了。
        # 将offset的顺序从[x1, y1, x2, y2, ...] 改成[x1, x2, .... y1, y2, ...]
        offsets_index = Variable(torch.cat([torch.arange(0, 2 * N, 2), torch.arange(1, 2 * N + 1, 2)]),
                                 requires_grad=False).type_as(x).long()
        # torch.unsqueeze()是为了增加维度,使offsets_index维度等于offset
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        # 根据维度dim按照索引列表index将offset重新排序，得到[x1, x2, .... y1, y2, ...]这样顺序的offset
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        # 对输入x进行padding
        if self.padding:
            x = self.zero_padding(x)

        # 将offset放到网格上，也就是标定出每一个坐标位置
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # 维度变换
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # floor是向下取整
        q_lt = Variable(p.data, requires_grad=False).floor()
        # +1相当于向上取整，这里为什么不用向上取整函数呢？是因为如果正好是整数的话，向上取整跟向下取整就重合了，这是我们不想看到的。
        q_rb = q_lt + 1

        # 将lt限制在图像范围内，其中[..., :N]代表x坐标，[..., N:]代表y坐标
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        # 将rb限制在图像范围内
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        # 获得lb
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        # 获得rt
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # 限制在一定的区域内,其实这部分可以写的很简单。有点花里胡哨的感觉。。在numpy中这样写：
        # p = np.where(p >= 1, p, 0)
        # p = np.where(p <x.shape[2]-1, p, x.shape[2]-1)

        # 插值的时候需要考虑一下padding对原始索引的影响
        # (b, h, w, N)
        # torch.lt() 逐元素比较input和other，即是否input < other
        # torch.rt() 逐元素比较input和other，即是否input > other
        mask = torch.cat([p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
                          p[..., N:].lt(self.padding) + p[..., N:].gt(x.size(3) - 1 - self.padding)], dim=-1).type_as(p)
        # 禁止反向传播
        mask = mask.detach()
        # p - (p - torch.floor(p))不就是torch.floor(p)呢。。。
        floor_p = p - (p - torch.floor(p))
        # 总的来说就是把超出图像的偏移量向下取整
        p = p * (1 - mask) + floor_p * mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        # 插值的4个系数
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        # 插值的最终操作在这里
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # 偏置点含有九个方向的偏置，_reshape_x_offset() 把每个点9个方向的偏置转化成 3×3 的形式，
        # 于是就可以用 3×3 stride=3 的卷积核进行 Deformable Convolution，
        # 它等价于使用 1×1 的正常卷积核（包含了这个点9个方向的 context）对原特征直接进行卷积。
        x_offset = self._reshape_x_offset(x_offset, ks)

        out = self.conv_kernel(x_offset)

        return out

    # 求每个点的偏置方向
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                   range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2 * N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    # 求每个点的坐标
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h + 1), range(1, w + 1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    # 求最后的偏置后的点=每个点的坐标+偏置方向+偏置
    def _get_p(self, offset, dtype):
        # N = 9, h, w
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    # 求出p点周围四个点的像素
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)将图片压缩到1维，方便后面的按照index索引提取
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)这个目的就是将index索引均匀扩增到图片一样的h*w大小
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        # 双线性插值法就是4个点再乘以对应与 p 点的距离。获得偏置点 p 的值，这个 p 点是 9 个方向的偏置所以最后的 x_offset 是 b×c×h×w×9。
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    # _reshape_x_offset() 把每个点9个方向的偏置转化成 3×3 的形式
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


if __name__ == '__main__':
    a = torch.rand((1, 3, 280, 280), requires_grad=True)
    d = DeformConvUnit(3, 3)
    c = d(a)
    pass
