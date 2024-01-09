# 编码器用自己设计的，超先验网络用Block_unet替代，整个网络框架用Neural——Syn
import os
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

import argparse
import glob
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import torch
import torchvision as tv
from torch import nn, optim
import torch.nn.functional as F

import pickle
from PIL import Image
from torch.autograd import Function
import time
import os
import math
# from .gdn import GDN, IGDN
#
# from .han import HAN_Head as HAN
# from .han import MeanShift
from .gdn import GDN, IGDN

from .han import HAN_Head as HAN
from .han import MeanShift
from tqdm import tqdm
from layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from .Haar import define_G

# coding:utf-8
"""
"""
from .Block_unet import *
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .entroformer_helper import Config, Block, clones
from .ops import UpPixelShuffle
from .util import *
# from .cit_content.TransDecoder2 import *
from .DepthwiseSeparableConv import *

import torch
from torch import nn
from torch.distributions import uniform
import numpy as np
import inspect
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
import torch.nn.functional as F

def Analyze_data(data):
    print("data.max():", data.max())
    print("data.min():", data.min())
    # for lice in range(0, 10):
    #     x = 0.1 ** lice
    #     count = torch.sum(torch.lt(data, x))
    #     print("<1e-{}: {},torch.log(): {}".format(lice, count, math.log(x)))


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)



import torch
from torch.cuda.amp import autocast
torch_device = "cuda"
pretrained_model_name_or_path = "/root/data/2021/buxiaobu/DDPM/stable-diffusion-main/stable_diffusion"



class ResidualBottleneck(nn.Module):
    def __init__(self, N=192, act=nn.GELU) -> None:
        super().__init__()

        self.branch = nn.Sequential(
            conv1x1(N, N // 2),
            act(),
            nn.Conv2d(N // 2, N // 2, kernel_size=3, stride=1, padding=1),
            act(),
            conv1x1(N // 2, N)
        )

    def forward(self, x):
        out = x + self.branch(x)
        return out


class Block_1(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block_1, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path) -> None:
        super().__init__()
        self.block_1 = Block_1(input_dim, output_dim, head_dim, window_size, drop_path, type='W')
        self.block_2 = Block_1(input_dim, output_dim, head_dim, window_size, drop_path, type='SW')
        self.window_size = window_size

    def forward(self, x):
        resize = False
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_col + 1, padding_row, padding_row + 1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        trans_x = self.block_1(trans_x)
        trans_x = self.block_2(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        if resize:
            x = F.pad(x, (-padding_col, -padding_col - 1, -padding_row, -padding_row - 1))
        return trans_x


class SWAtten(AttentionBlock):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192) -> None:
        if inter_dim is not None:
            super().__init__(N=inter_dim)
            self.non_local_block = SwinBlock(inter_dim, inter_dim, head_dim, window_size, drop_path)
        else:
            super().__init__(N=input_dim)
            self.non_local_block = SwinBlock(input_dim, input_dim, head_dim, window_size, drop_path)
        if inter_dim is not None:
            self.in_conv = conv1x1(input_dim, inter_dim)
            self.out_conv = conv1x1(inter_dim, output_dim)

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)
        out += identity
        out = self.out_conv(out)
        return out


def visualization_FeatureMap(inputs, model, channel, image_path, visual_iamge=0, start_layer=0, end_layer=0, ):
    '''
    channel: 可视化多少通道
    visual_image: 可视化第几张图
    start_layer: 如果是Sequential容器，从哪一层开始可视化
    end_layer: 到哪一层结束
    '''
    if isinstance(model, nn.Sequential):
        print("model is nn.Sequential type")
        conv_module = []
        for module in (model.children()):
            conv_module.append(module)
        conv_layer = conv_module[:end_layer]

        # 将取到的网络层重新放入Sequential中
        model = nn.Sequential(*list(conv_layer))
    else:
        model = model
    # print(conv_layer)
    image_mid = model(inputs)
    # print(inputs)
    # print("从第三层网络出来后image_mid：{}".format(image_mid.shape))
    image_mid = image_mid[visual_iamge]
    # 将数据恢复
    image_mid = (image_mid + 1) / 2
    for num in range(channel):
        image = image_mid[num, :, :]
        # print("image.shape:{}".format(image))
        import torchvision
        image_save = torchvision.transforms.functional.to_pil_image(image)
        # 将灰度图像转换为RGB格式
        image_save.save(
            image_path + '/orign_{}.png'.format(num, image_save))

        image = image.cpu().detach()
        image_numpy = np.array(image)  # 转为numpy类型
        # print(type(image_numpy))
        import cv2
        '''
        在 PyTorch 中，图像数据通常以张量（Tensor）的形式表示，张量的取值范围一般为 [0, 1] 或 [-1, 1]，
        这是由于神经网络训练时常使用 Batch Normalization 等技术对图像进行归一化处理。
        而在 OpenCV 中，图像数据通常以 Mat 类型表示，取值范围为 [0, 255]，即每个像素点的取值在 0 到 255 之间。
        因此，在将 PyTorch 中的张量转换为 OpenCV 中的 Mat 类型时，需要将张量的取值范围从 [0, 1] 或 [-1, 1] 转换为 [0, 255]

        astype(np.uint8) 表示将数组的数据类型转换为 uint8，以适配 Mat 类型的数据格式。

        其中，cv2.cvtColor 函数用于将 numpy 数组的颜色空间从 RGB 转换为 BGR，以适配 Mat 类型的数据格式
        '''
        image_numpy = cv2.cvtColor((image_numpy * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # image_numpy = image_numpy.astype(np.uint8)

        # print("image_mid:{}".format(image_mid.shape))
        import cv2
        for i in range(0, 13):
            im_color = cv2.applyColorMap(image_numpy, i)
            cv2.imwrite(image_path + "/orign_{}_{}.png".format(num, i),
                        im_color)


class NoiseQuant(nn.Module):
    def __init__(self, table_range=128, bin_size=1.0):
        super(NoiseQuant, self).__init__()
        self.table_range = table_range
        half_bin = torch.tensor(bin_size / 2).to(torch.device("cuda"))
        print("half_bin.get_device()", half_bin.get_device())
        self.noise = uniform.Uniform(-half_bin, half_bin)

    def forward(self, x):
        if self.training:
            x_quant = x + self.noise.sample(x.shape)
        else:
            x_quant = torch.floor(x + 0.5)  # modified
        return torch.clamp(x_quant, -self.table_range, self.table_range - 1)


# 编码器
# class analysisTransformModel(nn.Module):
#     '''
#                 ResidualBottleneck是一种卷积神经网络中的模块，
#                 主要用于加深网络深度，提高网络性能和效果
#     '''
#
#     # analysisTransformModel(3, [N, N, N, N])   N = 192
#     def __init__(self, in_dim, num_filters, conv_trainable=True):
#         super(analysisTransformModel, self).__init__()
#         self.transform = nn.Sequential(
#             # nn.Conv2d(in_dim, in_dim, 3, stride=1, padding=1),
#             # nn.ReLU(inplace=True),
#             # nn.ZeroPad2d((1, 2, 1, 2)),
#             # nn.Conv2d(in_dim, num_filters[0], 5, 2, 0),
#
#             ResidualBottleneck(in_dim),
#             ResidualBottleneck(in_dim),
#             ResidualBottleneck(in_dim),
#             ResidualBlockWithStride(in_dim, num_filters[0], stride=2),
#             GDN(num_filters[0]),
#             # nn.LeakyReLU(inplace=True),
#
#             # nn.Conv2d(num_filters[0], num_filters[0], 3, stride=1, padding=1),
#             # nn.ReLU(inplace=True),
#             nn.ZeroPad2d((1, 2, 1, 2)),
#             # ResidualBottleneck(num_filters[0]),
#             # ResidualBottleneck(num_filters[0]),
#             # ResidualBottleneck(num_filters[0]),
#             # ResidualBlockWithStride(num_filters[0], num_filters[1],2),
#             nn.Conv2d(num_filters[0], num_filters[1], 5, 2, 0),
#
#             GDN(num_filters[1]),
#
#             # nn.LeakyReLU(inplace=True),
#             Win_noShift_Attention(dim=num_filters[1], num_heads=8, window_size=8, shift_size=4),
#
#             # nn.Conv2d(num_filters[1], num_filters[1], 3, stride=1, padding=1),
#             # nn.ReLU(inplace=True),
#             # nn.ZeroPad2d((1, 2, 1, 2)),
#             ResidualBottleneck(num_filters[1]),
#             ResidualBottleneck(num_filters[1]),
#             ResidualBottleneck(num_filters[1]),
#             ResidualBlockWithStride(num_filters[1], num_filters[2], 2),
#             # nn.Conv2d(num_filters[1], num_filters[2], 5, 2, 0),
#             GDN(num_filters[2]),
#             # nn.LeakyReLU(inplace=True),
#
#             # nn.Conv2d(num_filters[2], num_filters[2], 3, stride=1, padding=1),
#             # nn.ReLU(inplace=True),
#             nn.ZeroPad2d((1, 2, 1, 2)),
#             nn.Conv2d(num_filters[2], num_filters[3], 5, 2, 0),
#             Win_noShift_Attention(dim=num_filters[3], num_heads=8, window_size=4, shift_size=2),
#         )
#
#     def forward(self, inputs):
#         x = self.transform(inputs)
#         return x
#
#
# class synthesisTransformModel_RBS(nn.Module):
#     # self.s_model = synthesisTransformModel(179, [N, N, N, M])  N 192  M  16
#     def __init__(self, in_dim, num_filters, conv_trainable=True):
#         super(synthesisTransformModel, self).__init__()
#         self.transform = nn.Sequential(
#
#             # nn.ConvTranspose2d(in_dim, in_dim, 3, stride=1, padding=1),
#             # nn.ReLU(inplace=True),
#             Win_noShift_Attention(dim=in_dim, num_heads=8, window_size=4, shift_size=2),
#             # nn.ZeroPad2d((1, 0, 1, 0)),
#             ResidualBlockUpsample(in_dim, num_filters[0], 2),
#             ResidualBottleneck(num_filters[0]),
#             ResidualBottleneck(num_filters[0]),
#             ResidualBottleneck(num_filters[0]),
#
#             IGDN(num_filters[0], inverse=True),
#             # nn.LeakyReLU(inplace=True),
#
#             nn.ConvTranspose2d(num_filters[0], num_filters[0], 3, stride=1, padding=1),
#             # nn.ReLU(inplace=True),
#             # nn.ZeroPad2d((1, 0, 1, 0)),
#             # ResidualBlockUpsample(num_filters[0], num_filters[1], 2),
#             IGDN(num_filters[1], inverse=True),
#             # nn.LeakyReLU(inplace=True),
#             Win_noShift_Attention(dim=num_filters[1], num_heads=8, window_size=8, shift_size=2),
#             # nn.ConvTranspose2d(num_filters[1], num_filters[1], 3, stride=1, padding=1),
#             # nn.ReLU(inplace=True),
#             # nn.ZeroPad2d((1, 0, 1, 0)),
#             ResidualBottleneck(num_filters[1]),
#             ResidualBottleneck(num_filters[1]),
#             ResidualBottleneck(num_filters[1]),
#
#             ResidualBlockUpsample(num_filters[1], num_filters[2], 2),
#             IGDN(num_filters[2], inverse=True),
#             # nn.LeakyReLU(inplace=True),
#             ResidualBottleneck(num_filters[2]),
#             ResidualBottleneck(num_filters[2]),
#             ResidualBottleneck(num_filters[2]),
#             # nn.ConvTranspose2d(num_filters[2], num_filters[2], 3, stride=1, padding=1),
#             # nn.ReLU(inplace=True),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.ConvTranspose2d(num_filters[2], num_filters[3], 5, 2, 3, output_padding=1),
#             IGDN(num_filters[3], inverse=True)
#             # nn.LeakyReLU(inplace=True)
#         )
#
#     def forward(self, inputs):
#         x = self.transform(inputs)
#         return x
#
#
# #  基于数据相关变换的解码器
# class synthesisTransformModel(nn.Module):
#     # self.s_model = synthesisTransformModel(179, [N, N, N, M])  N 192  M  16
#     def __init__(self, in_dim, num_filters, conv_trainable=True):
#         super(synthesisTransformModel, self).__init__()
#         self.transform = nn.Sequential(
#
#             # nn.ConvTranspose2d(in_dim, in_dim, 3, stride=1, padding=1),
#             # nn.ReLU(inplace=True),
#             Win_noShift_Attention(dim=in_dim, num_heads=8, window_size=4, shift_size=2),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.ConvTranspose2d(in_dim, num_filters[0], 5, 2, 3, output_padding=1),
#             IGDN(num_filters[0], inverse=True),
#             # nn.LeakyReLU(inplace=True),
#
#             # nn.ConvTranspose2d(num_filters[0], num_filters[0], 3, stride=1, padding=1),
#             # nn.ReLU(inplace=True),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.ConvTranspose2d(num_filters[0], num_filters[1], 5, 2, 3, output_padding=1),
#             IGDN(num_filters[1], inverse=True),
#             # nn.LeakyReLU(inplace=True),
#
#             Win_noShift_Attention(dim=num_filters[1], num_heads=8, window_size=8, shift_size=2),
#             # nn.ConvTranspose2d(num_filters[1], num_filters[1], 3, stride=1, padding=1),
#             # nn.ReLU(inplace=True),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.ConvTranspose2d(num_filters[1], num_filters[2], 5, 2, 3, output_padding=1),
#             IGDN(num_filters[2], inverse=True),
#             # nn.LeakyReLU(inplace=True),
#
#             # nn.ConvTranspose2d(num_filters[2], num_filters[2], 3, stride=1, padding=1),
#             # nn.ReLU(inplace=True),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.ConvTranspose2d(num_filters[2], num_filters[3], 5, 2, 3, output_padding=1),
#             IGDN(num_filters[3], inverse=True)
#             # nn.LeakyReLU(inplace=True)
#         )
#
#     def forward(self, inputs):
#         x = self.transform(inputs)
#         return x

class analysisTransformModel(nn.Module):

    # analysisTransformModel(3, [N, N, N, N])   N = 192
    def __init__(self, in_dim, num_filters, conv_trainable=True):
        super(analysisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_dim, num_filters[0], 5, 2, 0),
            GDN(num_filters[0]),

            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(num_filters[0], num_filters[1], 5, 2, 0),
            GDN(num_filters[1]),
            # Win_noShift_Attention(dim=num_filters[1], num_heads=8, window_size=8, shift_size=4),

            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(num_filters[1], num_filters[2], 5, 2, 0),
            GDN(num_filters[2]),

            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(num_filters[2], num_filters[3], 5, 2, 0),
            # Win_noShift_Attention(dim=num_filters[3], num_heads=8, window_size=4, shift_size=2),
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x


#  基于数据相关变换的解码器
class synthesisTransformModel(nn.Module):
    # self.s_model = synthesisTransformModel(179, [N, N, N, M])  N 192  M  16
    def __init__(self, in_dim, num_filters, conv_trainable=True):
        super(synthesisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            # Win_noShift_Attention(dim=in_dim, num_heads=8, window_size=4, shift_size=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(in_dim, num_filters[0], 5, 2, 3, output_padding=1),
            IGDN(num_filters[0], inverse=True),

            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters[0], num_filters[1], 5, 2, 3, output_padding=1),
            IGDN(num_filters[1], inverse=True),
            # Win_noShift_Attention(dim=num_filters[1], num_heads=8, window_size=8, shift_size=4),

            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters[1], num_filters[2], 5, 2, 3, output_padding=1),
            IGDN(num_filters[2], inverse=True),

            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(num_filters[2], num_filters[3], 5, 2, 3, output_padding=1),
            IGDN(num_filters[3], inverse=True)
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x


class Space2Depth(nn.Module):
    def __init__(self, r):
        super(Space2Depth, self).__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c * (r ** 2)
        out_h = h // 2
        out_w = w // 2
        x_view = x.view(b, c, out_h, r, out_w, r)
        x_prime = x_view.permute(0, 3, 5, 1, 2, 4).contiguous().view(b, out_c, out_h, out_w)
        return x_prime


class Depth2Space(nn.Module):
    def __init__(self, r):
        super(Depth2Space, self).__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c // (r ** 2)
        out_h = h * 2
        out_w = w * 2
        x_view = x.view(b, r, r, out_c, h, w)
        x_prime = x_view.permute(0, 3, 4, 1, 5, 2).contiguous().view(b, out_c, out_h, out_w)
        return x_prime


# 超先验模型
# h_analysisTransformModel   和  h_synthesisTransformModel构成超先验网络
class h_analysisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, strides_list, conv_trainable=True):
        super(h_analysisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_dim, num_filters[0], 3, strides_list[0], 1),
            nn.ReLU(),
            nn.Conv2d(num_filters[0], num_filters[1], 5, strides_list[1], 2),
            nn.ReLU(),
            nn.Conv2d(num_filters[1], num_filters[2], 5, strides_list[2], 2)
        )

    def forward(self, inputs):
        x = torch.abs(inputs)
        x = self.transform(x)
        return x


# 上采样
class h_synthesisTransformModel(nn.Module):
    def __init__(self, in_dim, num_filters, strides_list, conv_trainable=True):
        super(h_synthesisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.ConvTranspose2d(in_dim, num_filters[0], 5, strides_list[0], 2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters[0], num_filters[1], 5, strides_list[1], 2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters[1], num_filters[2], 3, strides_list[2], 1),
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x


# self.y_sampler = BlockSample((b, N - M, h // 8, w // 8))
class BlockSample(nn.Module):
    def __init__(self, in_shape, masked=True):
        super(BlockSample, self).__init__()
        self.masked = masked
        dim = in_shape[1]
        flt = np.zeros((dim * 16, dim, 7, 7), dtype=np.float32)
        for i in range(0, 4):
            for j in range(0, 4):
                if self.masked:
                    if i == 3:
                        if j == 2 or j == 3:
                            break
                for k in range(0, dim):
                    s = k * 16 + i * 4 + j
                    flt[s, k, i, j + 1] = 1
        flt_tensor = torch.Tensor(flt).float().cuda()
        self.register_buffer('sample_filter', flt_tensor)

    def forward(self, inputs):  # [8,384,16,16]

        t = F.conv2d(inputs, self.sample_filter, padding=3)
        b, c, h, w = inputs.size()
        t = t.contiguous().view(b, c, 4, 4, h, w).permute(0, 4, 5, 1, 2, 3)
        t = t.contiguous().view(b * h * w, c, 4, 4)
        return t


class NeighborSample(nn.Module):
    def __init__(self, in_shape):
        super(NeighborSample, self).__init__()
        dim = in_shape[1]
        flt = np.zeros((dim * 25, dim, 5, 5), dtype=np.float32)
        for i in range(0, 5):
            for j in range(0, 5):
                for k in range(0, dim):
                    s = k * 25 + i * 5 + j
                    flt[s, k, i, j] = 1
        flt_tensor = torch.Tensor(flt).float().cuda()
        self.register_buffer('sample_filter', flt_tensor)

    def forward(self, inputs):
        t = F.conv2d(inputs, self.sample_filter, padding=2)
        b, c, h, w = inputs.size()
        t = t.contiguous().view(b, c, 5, 5, h, w).permute(0, 4, 5, 1, 2, 3)
        t = t.contiguous().view(b * h * w, c, 5, 5)
        return t


class GaussianModel(nn.Module):
    def __init__(self):
        super(GaussianModel, self).__init__()
        #  创建均值为1  方差为0的正态分布    也成为高斯分布
        self.m_normal_dist = torch.distributions.normal.Normal(0., 1.)

    def _cumulative(self, inputs, stds, mu):
        half = 0.5
        eps = 1e-6
        upper = (inputs - mu + half) / (stds)
        lower = (inputs - mu - half) / (stds)
        cdf_upper = self.m_normal_dist.cdf(upper)  # 求小于upper值的概率
        cdf_lower = self.m_normal_dist.cdf(lower)  # 求小于lower值的概率
        res = cdf_upper - cdf_lower  # 相减得到区间内的概率
        return res

    # inputs: 8 * 192 * 4 * 4   hyper_sigma: 1 * 192 * 1 * 1  hyper_mu：1 * 192 * 1 * 1
    def forward(self, inputs, hyper_sigma, hyper_mu):
        likelihood = self._cumulative(inputs, hyper_sigma, hyper_mu)
        # likelihood_bound = 1e-8
        likelihood_bound = 1e-12
        likelihood = torch.clamp(likelihood, min=likelihood_bound)
        return likelihood


class PredictionModel_Context(nn.Module):
    def __init__(self, in_dim, dim=192, trainable=True, outdim=None):
        super(PredictionModel_Context, self).__init__()
        if outdim is None:
            outdim = dim
        self.transform = nn.Sequential(
            nn.Conv2d(in_dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            # Win_noShift_Attention(dim=dim, num_heads=8, window_size=4, shift_size=2),

            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(dim * 2 * 2, outdim)
        self.flatten = nn.Flatten()

    def forward(self, y_rounded, h_tilde, y_sampler, h_sampler):
        b, c, h, w = y_rounded.size()
        y_sampled = y_sampler(y_rounded)
        h_sampled = h_sampler(h_tilde)
        merged = torch.cat([y_sampled, h_sampled], 1)
        y_context = self.transform(merged)
        y_context = self.flatten(y_context)
        y_context = self.fc(y_context)
        hyper_mu = y_context[:, :c]
        hyper_mu = hyper_mu.view(b, h, w, c).permute(0, 3, 1, 2)
        hyper_sigma = y_context[:, c:]
        hyper_sigma = torch.exp(hyper_sigma)
        hyper_sigma = hyper_sigma.contiguous().view(b, h, w, c).permute(0, 3, 1, 2)

        return hyper_mu, hyper_sigma


class conv_generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(conv_generator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.transform = nn.Sequential(
            nn.Linear(in_dim, 128),
            # 激活函数
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, out_dim * 3)
        )

    def forward(self, x):
        b, _, _, _ = x.shape
        x = x.view(b, -1)

        weights = self.transform(x)
        weights = weights.view(b, 3, self.out_dim, 1, 1)

        return weights

    #  语法生成模型


#  语法模型生成器
class Syntax_Model(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Syntax_Model, self).__init__()
        self.Depth_down0 = DepthwiseSeparableConv(in_channels=16, out_channels=16)
        self.down0 = nn.Conv2d(in_dim, 32, 3, 2, 1)

        self.Depth_down1 = DepthwiseSeparableConv(in_channels=32, out_channels=32)
        self.down1 = nn.Conv2d(32, 64, 3, 2, 1)

        self.Depth_down2 = DepthwiseSeparableConv(in_channels=64, out_channels=64)
        self.down2 = nn.Conv2d(64, 128, 3, 2, 1)

        self.WAM = Win_noShift_Attention(dim=64, num_heads=8, window_size=4, shift_size=2)
        self.conv = nn.Conv2d(in_dim + 32 + 64 + 128, out_dim, 1, 1, 0)
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, syntax):
        out1 = self.pooling(syntax)
        depth1 = self.Depth_down0(syntax)
        ds1 = self.down0(depth1)
        ds1 = F.relu(ds1)

        out2 = self.pooling(ds1)
        depth2 = self.Depth_down1(ds1)
        ds2 = self.down1(depth2)
        ds2 = F.relu(ds2)
        ds2 = self.WAM(ds2)

        out3 = self.pooling(ds2)
        depth3 = self.Depth_down2(ds2)
        ds3 = self.down2(depth3)
        ds3 = F.relu(ds3)
        out4 = self.pooling(ds3)

        out = torch.cat((out1, out2, out3, out4), 1)

        out = self.conv(out)
        return out


class PredictionModel_Syntax(nn.Module):
    def __init__(self, in_dim, dim=192, trainable=True, outdim=None):
        super(PredictionModel_Syntax, self).__init__()
        if outdim is None:
            outdim = dim

        self.down0 = nn.Conv2d(in_dim, dim, 3, 2, 1)
        self.down1 = nn.Conv2d(dim, dim, 3, 2, 1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.WAM = Win_noShift_Attention(dim=dim, num_heads=8, window_size=4, shift_size=2)

        self.fc = nn.Linear(dim * 2 + in_dim, outdim)
        self.flatten = nn.Flatten()

    def forward(self, y_rounded, h_tilde, h_sampler=None):
        b, c, h, w = y_rounded.size()

        ds0 = self.down0(h_tilde)
        ds0 = F.relu(ds0)

        ds1 = self.down1(ds0)
        ds1 = F.relu(ds1)
        ds1 = self.WAM(ds1)

        ds0 = self.pooling(ds0)
        ds1 = self.pooling(ds1)
        ori = self.pooling(h_tilde)
        y_context = torch.cat((ori, ds0, ds1), 1)

        y_context = self.flatten(y_context)
        y_context = self.fc(y_context)
        hyper_mu = y_context[:, :c]
        hyper_mu = hyper_mu.view(b, h, w, c).permute(0, 3, 1, 2)
        hyper_sigma = y_context[:, c:]
        hyper_sigma = torch.exp(hyper_sigma)
        hyper_sigma = hyper_sigma.contiguous().view(b, h, w, c).permute(0, 3, 1, 2)

        return hyper_mu, hyper_sigma


class BypassRound(Function):
    @staticmethod
    def forward(ctx, inputs):
        return torch.round(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


bypass_round = BypassRound.apply


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def ste_round(x):
    # 对x 进行去噪的操作
    '''
    torch.round(x) : 对x进行四舍五入的操作
    x.detach（）: 返回与张量x相同的数据，但不会被梯度跟踪，也就是不会对原始张量x的梯度产生影响
    '''
    return torch.round(x) - x.detach() + x


# 初始化权重
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


num = 1


class Net(nn.Module):
    def __init__(self, train_size, test_size, is_high, post_processing):
        super(Net, self).__init__()
        print("初始化完成--------------------------------")
        parser = get_parser()
        opt = parser.parse_args()
        # print("opt:",opt)

        self.mse = nn.MSELoss()
        self.num_slices = 4
        self.max_support_slices = 4
        self.gaussian_conditional = GaussianConditional(None)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_size = train_size
        self.test_size = test_size

        self.post_processing = post_processing
        # self.post_processing = True

        self.is_high = is_high
        # is_high 这个参数 不知道是做什么的  M  和  N是干嘛的
        '''
        在n=192和m=16的情况下，构造了较低比特率范围的模型。 对于较高的比特率范围，
        n=384和m=32，以提供足够的信息容量。 我们的后处理网络基于Han[26]。
        我们的后处理网络基于Han[26]。 具体来说，我们对低比特率模型使用4个残差组，
        对高比特率模型使用6个残差组。 我们去掉上采样器，
        '''
        if self.is_high:
            N = 384
            M = 32
        else:
            N = 192
            M = 16
        self.M = M  # 16
        self.N = N  # 192
        self.conv_1 = conv1x1(192,4)
        self.conv_2 = conv1x1(4, 192)
        # 加入netG网络，加入图像缩放
        # self.netG = define_G()
        self.a_model = analysisTransformModel(3, [N, N, N, N])
        self.a_model.apply(weight_init)
        # self.S = 128
        # a_model 为编码器
        # self.a_model = analysisTransformModel(3, [N, N, N, N])
        # s_model 为解码器
        self.s_model = synthesisTransformModel(N, [N, N, N, M])
        self.s_model.apply(weight_init)
        # 语法生成模型
        # 初始化模型
        # Syntax_Model = create_model(num_classes=16,in_c=self.S, has_logits=False).to(device)
        # self.syntax_model = Syntax_Model
        self.syntax_model = Syntax_Model(M, M)
        self.syntax_model.apply(weight_init)

        # 双向上下文模型
        # print("出错了")
        # self.cit_ar = TransDecoder2(cin=opt.last_channels, opt=opt)
        # print("self.cit_ar = TransDecoder(cin=opt.last_channels, opt=opt)执行完毕")
        # self.content_model = Block_train(768,256,16)

        self.conv_weights_gen = conv_generator(in_dim=M, out_dim=M)
        self.conv_weights_gen.apply(weight_init)
        self.quant_noise = NoiseQuant(table_range=128)

        #  构建超先验网络
        # self.ha_model = h_analysisTransformModel(N, [N, N, N], [1, 2, 2])
        # self.hs_model = h_synthesisTransformModel(N, [N, N, N], [2, 2, 1])
        # self.cit_he = TransHyperScale(cin=N, cout=N, scale=2, down=True, opt=opt)
        # self.cit_hd = TransHyperScale(cin=N, cout=N, scale=2, down=False, opt=opt)
        # self.CTM_ha = ConvTransBlock_ha(N // 2, N // 2, 4, 4, 0, )
        # self.CTM_hs = ConvTransBlock_hs(N // 2, N // 2, 4, 4, 0, )
        # self.entropy_bottleneck = EntropyBottleneck(192)

        self.h_a = Unet_ha_new(192, 8, 3)
        self.h_s = Unet_hs_new(192, 8, 3)
        # 初始化超先验网络的参数
        #
        # self.cit_he.apply(vit2_init)
        # self.cit_hd.apply(vit2_init)

        # self.gaussian_conditional = GaussianConditional(None)
        self.entropy_bottleneck_z2 = GaussianModel()
        self.entropy_bottleneck_z2.apply(weight_init)

        self.entropy_bottleneck_z3 = GaussianModel()
        self.entropy_bottleneck_z3.apply(weight_init)
        self.entropy_bottleneck = EntropyBottleneck(512)
        self.entropy_bottleneck_z3_syntax = GaussianModel()
        self.entropy_bottleneck_z3_syntax.apply(weight_init)
        self.window_size = 8
        self.atten_mean = nn.ModuleList(
            nn.Sequential(
                SWAtten((192 + (192 // self.num_slices) * min(i, 4)), (192 + (192 // self.num_slices) * min(i, 4)), 16,
                        self.window_size, 0, inter_dim=128)
            ) for i in range(self.num_slices)
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(192 + (192 // self.num_slices) * min(i, 4), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (192 // self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_mean_transforms.apply(weight_init)

        self.atten_scale = nn.ModuleList(
            nn.Sequential(
                SWAtten((192 + (192 // self.num_slices) * min(i, 4)), (192 + (192 // self.num_slices) * min(i, 4)), 16,
                        self.window_size, 0, inter_dim=128)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(192 + (192 // self.num_slices) * min(i, 4), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (192 // self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms.apply(weight_init)
        b, h, w, c = train_size
        tb, th, tw, tc = test_size

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(192 + (192 // self.num_slices) * min(i + 1, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (192 // self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.v_z2_sigma = nn.Parameter(torch.ones((1, N, 1, 1), dtype=torch.float32, requires_grad=True))
        #  这行代码不理解  向模型里面添加参数
        '''
        jupyter 中要用这行代码  你得告诉程序   你要往哪个网络里面添加参数，也就是这个net
        net.register_parameter('z2_sigma',v_z2_sigma)
        '''
        self.register_parameter('z2_sigma', self.v_z2_sigma)

        self.prediction_model = PredictionModel_Context(in_dim=2 * N - M, dim=N, outdim=(N - M) * 2)
        self.prediction_model.apply(weight_init)

        self.prediction_model_syntax = PredictionModel_Syntax(in_dim=N, dim=M, outdim=M * 2)
        self.prediction_model_syntax.apply(weight_init)

        self.y_sampler = BlockSample((b, N - M, h // 8, w // 8))
        # self.y_sampler = BlockSample((b, N, h // 8, w // 8))
        self.y_sampler.apply(weight_init)

        self.h_sampler = BlockSample((b, N, h // 8, w // 8), False)
        self.h_sampler.apply(weight_init)

        self.test_y_sampler = BlockSample((b, N - M, th // 8, tw // 8))
        self.test_y_sampler.apply(weight_init)

        self.test_h_sampler = BlockSample((b, N, th // 8, tw // 8), False)
        self.test_h_sampler.apply(weight_init)

        self.HAN = HAN(is_high=self.is_high)
        self.conv_weights_gen_HAN = conv_generator(in_dim=M, out_dim=64)
        self.conv_weights_gen_HAN.apply(weight_init)

        self.add_mean = MeanShift(1.0, (0.4488, 0.4371, 0.4040), (1.0, 1.0, 1.0), 1)
        self.add_mean.apply(weight_init)
        # print("初始化完毕----------------------------")

    def post_processing_params(self):
        params = []
        params += self.conv_weights_gen_HAN.parameters()
        params += self.HAN.parameters()

        return params

    def base_params(self):
        params = []
        params += self.a_model.parameters()
        params += self.s_model.parameters()

        params += self.h_a.parameters()
        params += self.h_s.parameters()

        params += self.syntax_model.parameters()
        params += self.conv_weights_gen.parameters()

        params += self.prediction_model.parameters()
        params += self.prediction_model_syntax.parameters()

        params.append(self.v_z2_sigma)

        return params

    def batch_conv(self, weights, inputs):
        # print("----------------1----------------------")
        b, ch, _, _ = inputs.shape
        _, ch_out, _, k, _ = weights.shape

        weights = weights.reshape(b * ch_out, ch, k, k)
        inputs = torch.cat(torch.split(inputs, 1, dim=0), dim=1)
        out = F.conv2d(inputs, weights, stride=1, padding=0, groups=b)
        out = torch.cat(torch.split(out, ch_out, dim=1), dim=0)

        return out

    def forward(self, inputs, mode='train', num=1):

        b, h, w, c = self.train_size
        # print("b, h, w, c  ----------{}".format((b, h, w, c)))
        tb, th, tw, tc = self.test_size
        # 8 * 192 *  16 * 16  经过四层下采样
        # 从输入图片汇总抽取信息
        z3 = self.a_model(inputs)
        y_shape = z3.shape[2:]

        z, middle_x, down_x1, input = self.h_a(z3)

        _, z_likelihoods = self.entropy_bottleneck(z)  # 8 * 192 * 4 * 4
        # print("z_likelihoods:",z_likelihoods.shape)
        # _get_medians() 用于计算符号张量中每个通道的中位数
        z_offset = self.entropy_bottleneck._get_medians()  # 8 * 192 * 1 * 1
        z_tmp = z - z_offset

        # ste_round(z_tmp) 对z_tmp 进行简单的去噪
        z_hat = ste_round(z_tmp) + z_offset  # 8 * 192 * 4 * 4

        # 经过超先验解码器  分成两流，用于输入到后面每个切片网络
        latent_scales = self.h_s(z_hat,middle_x,down_x1 ,input)  # 8 * 320 * 16 * 16
        latent_means = self.h_s(z_hat,middle_x,down_x1 ,input)  # 8 * 320 * 16 * 16

        z3_syntax = z3[:, :self.M, :, :]
        # 生成语法  syntax_model： 语法生成模型
        '''
        输入的大小 z3_syntax torch.Size([8, 16, 16, 16])
        '''
        z3_syntax = self.syntax_model(z3_syntax)
        z3_syntax_rounded = bypass_round(z3_syntax)

        '''
        基于均匀通道的自适应熵编码
        '''
        y_slices = z3.chunk(self.num_slices, 1)
        # print("y_slices:{}",y_slices)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []
        # print("zhixingdao 这里")
        for slice_index, y_slice in enumerate(y_slices):
            # max_support_slices = 4
            # 48
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            # print("type(support_slices):{},latent_means:{}".format(type(support_slices), type(latent_means)))

            mean_support = torch.cat([latent_means] + support_slices, dim=1)  # 8 * 240 * 16 * 16

            # 注意力机制
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)  # 8 * 64 * 16 * 16
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)  # 8 * 320 * 16 * 16
            scale = self.cc_scale_transforms[slice_index](scale_support)  # 8 * 64 * 16 * 16
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_list.append(scale)

            # 8 * 64 * 16 * 16
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)

            # ste_round 进行去噪处理
            y_hat_slice = ste_round(y_slice - mu) + mu  # 8 *64 * 16 * 16
            # if self.training:
            #     lrp_support = torch.cat([mean_support + torch.randn(mean_support.size()).cuda().mul(scale_support), y_hat_slice], dim=1)
            # else:
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)  # 8 * 384 * 16 * 16
            lrp = self.lrp_transforms[slice_index](lrp_support)  # 8 * 64 * 16 * 16
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)  # 8 * 320 * 16 * 16
        means = torch.cat(mu_list, dim=1)  # 8 * 320 * 16 * 16
        scales = torch.cat(scale_list, dim=1)  # 8 * 320 * 16 * 16
        y_likelihoods = torch.cat(y_likelihood, dim=1)  # 8 * 320 * 16 * 16

        # 解码
        '''
        输入大小 
        torch.Size([8, 176, 16, 16])
        输出大小：
        torch.Size([8, 16, 256, 256])
        '''
        # 解码
        # x_tilde = self.s_model(z3_content_rounded)
        x_tilde = self.s_model(y_hat)
        # print("x_tilde")
        # Analyze_data(x_tilde)
        # 这些参数在解码时间从权重生成器动态生成
        # 利用多层全连通网络将神经语法表示映射到解码器网络中最终层的核参数
        conv_weights = self.conv_weights_gen(z3_syntax_rounded)
        # '''
        # conv_weights: 8 * 3 *16 * 1 * 1
        # x_tilde :  8 * 16  * 256 * 256
        # 输出 x_tilde_bf : 8 * 3 * 256 * 256
        # '''
        x_tilde_bf = self.batch_conv(conv_weights, x_tilde)
        # print("x_tilde_bf")
        # Analyze_data(x_tilde_bf)
        x_tilde_bf = torch.tanh(x_tilde_bf)
        # print("x_tilde_bf_tanhs")
        # Analyze_data(x_tilde_bf)
        # x_tilde_bf = torch.clamp(x_tilde_bf, -1, 1)
        if self.post_processing:
            x_tilde = self.HAN(x_tilde_bf)
            conv_weights = self.conv_weights_gen_HAN(z3_syntax_rounded)
            x_tilde = self.batch_conv(conv_weights, x_tilde)
            x_tilde = self.add_mean(x_tilde)
        else:
            x_tilde = x_tilde_bf

        num_pixels = inputs.size()[0] * h * w

        if mode == 'train':
            bpp_list = torch.sum(torch.log(y_likelihoods), [0, 1, 2, 3]) / (-np.log(2) * num_pixels)
            # train_bpp = bpp_list[0] + bpp_list[1] + bpp_list[2]
            # print("--------inputs[:,:,:h,:w]:{},x_tilde[:,:,:h,:w]:{}".format(inputs[:,:,:h,:w].shape,x_tilde[:,:,:h,:w].shape))
            # train_mse = torch.mean((inputs[:, :, :h, :w] - x_tilde[:, :, :h, :w]) ** 2, [0, 1, 2, 3])
            #
            # train_mse *= 255 ** 2
            # 更改loss计算方式
            train_mse = self.mse(x_tilde, inputs)
            return bpp_list, train_mse

        elif mode == 'test':
            x_tilde = torch.clamp(x_tilde, -1, 1)

            data = x_tilde
            image = data[0]
            image = (image + 1) / 2
            print("image :{}".format(image.shape))
            print("image.min:{}".format(image.min()))
            print("image.max:{}".format(image.max()))
            import torchvision
            image = torchvision.transforms.functional.to_pil_image(image)
            image.save('/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/Net_WAM_CTM/Net_WAM_CTM_Spilt/2.png')
            # print("已保存")
            # image.save('/root/data/2021/buxiaobu/Neural_Syntax/Pretrain_model/image/2.png')

            test_num_pixels = inputs.size()[0] * th * tw

            bpp_list = torch.sum(torch.log(y_likelihoods), [0, 1, 2, 3]) / (-np.log(2) * num_pixels)

            # Bring both images back to 0..255 range.
            gt = torch.round((inputs + 1) * 127.5)
            x_hat = torch.clamp((x_tilde + 1) * 127.5, 0, 255)
            x_hat = torch.round(x_hat).float()

            v_mse = torch.mean((x_hat - gt) ** 2, [1, 2, 3])
            v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0)

            return bpp_list, v_mse, v_psnr


if __name__ == "__main__":
    pass
