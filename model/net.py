import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import glob
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import torch
import torchvision as tv
from torch import nn, optim
import torch.nn.functional as F
from .vit_model import vit_base_patch16_224_in21k as create_model
import pickle
from PIL import Image
from torch.autograd import Function
import time
import os
import math
from .gdn import GDN, IGDN

from .han import HAN_Head as HAN
from .han import MeanShift
from tqdm import tqdm
from layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from .Haar import define_G


# 可视化Sequential里面某一层的特征图
'''
model：Sequential的名字
inputs: 输入图片信息
start_layer: 开始的层数
end_layer: 结束的层数
visual_iamge: 默认可视化第一张图片
channel : 通道数
'''
def visualization_FeatureMap(inputs,model,channel,image_path,visual_iamge=0,start_layer=0,end_layer=0,):

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
            image_path+'/orign_{}.png'.format(num, image_save))

        image = image.cpu()
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
            cv2.imwrite(image_path+"/orign_{}_{}.png".format(num, i),
                        im_color)
# 编码器
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
        x = torch.abs(inputs)  # 求绝对值
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

    def forward(self, inputs):
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
        cdf_upper = self.m_normal_dist.cdf(upper)
        cdf_lower = self.m_normal_dist.cdf(lower)
        res = cdf_upper - cdf_lower
        return res

    def forward(self, inputs, hyper_sigma, hyper_mu):
        likelihood = self._cumulative(inputs, hyper_sigma, hyper_mu)
        likelihood_bound = 1e-8
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
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(dim * 2 * 2, outdim)
        self.flatten = nn.Flatten()

    def forward(self, y_rounded, h_tilde, y_sampler, h_sampler):
        b, c, h, w = y_rounded.size()   # 8 * 176 * 16 * 16
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
        # 输入大小 8 * 16 * 16 * 16
        super(Syntax_Model, self).__init__()
        self.down0 = nn.Conv2d(in_dim, 32, 3, 2, 1)
        self.down1 = nn.Conv2d(32, 64, 3, 2, 1)

        self.conv = nn.Conv2d(in_dim + 32 + 64, out_dim, 1, 1, 0)
        self.pooling = nn.AdaptiveAvgPool2d(1)  # 自适应池化层

    def forward(self, syntax):
        out1 = self.pooling(syntax)  # 8 * 16 * 1 * 1
        ds1 = self.down0(syntax)  # 8 * 32 * 1 * 1
        ds1 = F.relu(ds1)
        out2 = self.pooling(ds1)  # 8 * 32 * 1 * 1

        ds2 = self.down1(ds1)  # 8 * 64 * 1 * 1
        ds2 = F.relu(ds2)
        out3 = self.pooling(ds2)  # 8 * 64 * 1 * 1

        # ds3 = self.down1(ds2)
        # ds3 = F.relu(ds3)
        # out4 = self.pooling(ds3)

        out = torch.cat((out1, out2, out3), 1)
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

        self.fc = nn.Linear(dim * 2 + in_dim, outdim)
        self.flatten = nn.Flatten()

    def forward(self, y_rounded, h_tilde, h_sampler=None):
        b, c, h, w = y_rounded.size()

        ds0 = self.down0(h_tilde)
        ds0 = F.relu(ds0)

        ds1 = self.down1(ds0)
        ds1 = F.relu(ds1)

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


class Net(nn.Module):
    def __init__(self, train_size, test_size, is_high, post_processing):
        super(Net, self).__init__()
        print("初始化完成--------------------------------")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_size = train_size
        self.test_size = test_size
        self.post_processing = post_processing
        self.is_high = is_high
        self.mse = nn.MSELoss()
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
        # 加入netG网络，加入图像缩放
        # self.netG = define_G()
        self.a_model = analysisTransformModel(3, [N, N, N, N])

        # self.S = 128
        # a_model 为编码器
        # self.a_model = analysisTransformModel(3, [N, N, N, N])
        # s_model 为解码器
        self.s_model = synthesisTransformModel(N - M, [N, N, N, M])

        # 语法生成模型
        # 初始化模型
        # Syntax_Model = create_model(num_classes=16,in_c=self.S, has_logits=False).to(device)
        # self.syntax_model = Syntax_Model
        self.syntax_model = Syntax_Model(M, M)
        self.conv_weights_gen = conv_generator(in_dim=M, out_dim=M)

        #  构建超先验网络
        self.ha_model = h_analysisTransformModel(N, [N, N, N], [1, 2, 2])
        self.hs_model = h_synthesisTransformModel(N, [N, N, N], [2, 2, 1])

        self.entropy_bottleneck_z2 = GaussianModel()
        self.entropy_bottleneck_z3 = GaussianModel()
        self.entropy_bottleneck_z3_syntax = GaussianModel()

        b, h, w, c = train_size
        tb, th, tw, tc = test_size

        self.v_z2_sigma = nn.Parameter(torch.ones((1, N, 1, 1), dtype=torch.float32, requires_grad=True))
        #  这行代码不理解  向模型里面添加参数
        '''
        jupyter 中要用这行代码  你得告诉程序   你要往哪个网络里面添加参数，也就是这个net
        net.register_parameter('z2_sigma',v_z2_sigma)
        '''
        self.register_parameter('z2_sigma', self.v_z2_sigma)

        self.prediction_model = PredictionModel_Context(in_dim=2 * N - M, dim=N, outdim=(N - M) * 2)
        self.prediction_model_syntax = PredictionModel_Syntax(in_dim=N, dim=M, outdim=M * 2)

        self.y_sampler = BlockSample((b, N - M, h // 8, w // 8))
        self.h_sampler = BlockSample((b, N, h // 8, w // 8), False)
        self.test_y_sampler = BlockSample((b, N - M, th // 8, tw // 8))
        self.test_h_sampler = BlockSample((b, N, th // 8, tw // 8), False)

        self.HAN = HAN(is_high=self.is_high)
        self.conv_weights_gen_HAN = conv_generator(in_dim=M, out_dim=64)
        self.add_mean = MeanShift(1.0, (0.4488, 0.4371, 0.4040), (1.0, 1.0, 1.0), 1)

    def post_processing_params(self):
        params = []
        params += self.conv_weights_gen_HAN.parameters()
        params += self.HAN.parameters()

        return params

    def base_params(self):
        params = []
        params += self.a_model.parameters()
        params += self.s_model.parameters()

        params += self.ha_model.parameters()
        params += self.hs_model.parameters()

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

    def forward(self, inputs, mode='train',num = 1):
        # print("----------------2----------------------")
        # net_g = self.netG(inputs)
        # print("-----------------inputs:{}".format(inputs.shape))
        b, h, w, c = self.train_size
        # print("b, h, w, c  ----------{}".format((b, h, w, c)))
        tb, th, tw, tc = self.test_size

        # 8 * 192 *  16 * 16  经过四层下采样
        # 从输入图片汇总抽取信息
        # print("inputs 的 大小 {}----------------".format(inputs.shape))
        # conv_layers = list(self.a_model.children())
        # print("con_layer:",type(conv_layers),conv_layers)
        # # print("conv_layers:".format(conv_layers))
        # conv_module = []
        # for module in (self.a_model.transform.children()):
        #     print(module)
        #     conv_module.append(module)
        # conv_layer = conv_module[:3]
        # print(len(conv_layer))
        # conv_layer = nn.Sequential(*list(conv_layer))
        # # print(conv_layer)
        # image_mid = conv_layer(inputs)
        # # print(inputs)
        # # print("从第三层网络出来后image_mid：{}".format(image_mid.shape))
        # image_mid = image_mid[0]
        # image_mid = (image_mid + 1) / 2
        # for num in range(192):
        #     image = image_mid[num,:,:]
        #     # print("image.shape:{}".format(image))
        #     import torchvision
        #     image_save = torchvision.transforms.functional.to_pil_image(image)
        #     # 将灰度图像转换为RGB格式
        #     image_save.save('/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/test_kokal/CNN/CNN_3/orign_{}.png'.format(num,image_save))
        #
        #     image = image.cpu()
        #
        #     image_numpy = np.array(image)   # 转为numpy类型
        #     # print(type(image_numpy))
        #
        #     import cv2
        #     '''
        #     在 PyTorch 中，图像数据通常以张量（Tensor）的形式表示，张量的取值范围一般为 [0, 1] 或 [-1, 1]，
        #     这是由于神经网络训练时常使用 Batch Normalization 等技术对图像进行归一化处理。
        #     而在 OpenCV 中，图像数据通常以 Mat 类型表示，取值范围为 [0, 255]，即每个像素点的取值在 0 到 255 之间。
        #     因此，在将 PyTorch 中的张量转换为 OpenCV 中的 Mat 类型时，需要将张量的取值范围从 [0, 1] 或 [-1, 1] 转换为 [0, 255]
        #
        #     astype(np.uint8) 表示将数组的数据类型转换为 uint8，以适配 Mat 类型的数据格式。
        #
        #     其中，cv2.cvtColor 函数用于将 numpy 数组的颜色空间从 RGB 转换为 BGR，以适配 Mat 类型的数据格式
        #     '''
        #     image_numpy = cv2.cvtColor((image_numpy * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        #     # image_numpy = image_numpy.astype(np.uint8)
        #
        #     # print("image_mid:{}".format(image_mid.shape))
        #     import cv2
        #     for i in range(0, 13):
        #         im_color = cv2.applyColorMap(image_numpy, i)
        #         cv2.imwrite("/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/test_kokal/CNN/CNN_3/{}_{}.png".format(num,i),
        #                             im_color)


        # for t in range(0,192):
        #     image = image_mid[:,:,t]
        #     # print("image_shape:",image.shape)
        #     cv2.imwrite("/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/test_kokal/CNN/3_orign_{}.png".format(t),
        #                 image)
            # for i in range(0, 13):
            #     im_color = cv2.applyColorMap(image, i)
            #     cv2.imwrite("/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/test_kokal/CNN/3_{}_{}.png".format(t,i),
            #                         im_color)
        # image_path = "/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/test_kokal/CNN/CNN_6/"
        # visualization_FeatureMap(inputs,self.a_model.transform,192,image_path,end_layer=6)
        #
        # image_mid = image_mid.cpu()
        # image_mid = np.array(image_mid)
        # image_mid = image_mid.transpose(1, 2, 0).astype(np.uint8)
        # # print("image_mid:{}".format(image_mid.shape))

        #
        # for t in range(192):
        #     image = image_mid[:, :, t]
        #     # print("image_shape:", image.shape)
        #     for i in range(0, 13):
        #         im_color = cv2.applyColorMap(image, i)
        #         cv2.imwrite("/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/test_kokal/CNN/6_{}_{}.png".format(t, i),
        #                     im_color)

        z3 = self.a_model(inputs)
        # print("开始保存")
        image_path_ga = "/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/Copy/test_image_kokal/ga"
        # visualization_FeatureMap(inputs,self.a_model,192,image_path_ga)

        # print("z3.shape:{}".format(z3.shape))
        # image = z3[0]
        # print("image_z3.shape:{}".format(image.shape))
        # image -= image.mean()
        # image /= image.std()
        # image *= 64
        # image += 128
        # image = torch.clamp(image, 0, 255)
        # image = (image + 1) / 2
        # image = image.cpu()
        # image = np.array(image)
        # print("image_numpy:",image.shape)
        # feature_map = 1
        #
        # # k = image.transpose(2,1,0).astype(np.uint8)
        # k = image.transpose(1,2,0)
        # print("k:", k.shape)
        # import torchvision
        # image = torchvision.transforms.functional.to_pil_image(image)
        # 将灰度图像转换为RGB格式
        # image.save('/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/test_kokal/ga/{}.png'.format(feature_map))
        # feature_map = feature_map + 1

        # import cv2
        # for i in range(0, 13):
        #     im_color = cv2.applyColorMap(k, i)
        #     cv2.imwrite("/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/test_kokal/ga/{}_{}.png".format(a,i),
        #                         im_color)
        # feature_map = feature_map + 1

        # print("-----------------z3:{}".format(z3.shape))
        # print("从a_model中出来Z3 的 大小 {}----------------".format(z3.shape))

        # 进入超先验网络 对z3 进进行三次卷积下采样操作  8 * 192 *  16 * 16--> torch.Size([8, 192, 4, 4])
        z2 = self.ha_model(z3)
        # image_path_ha = "/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/Copy/test_image_kokal/ha"
        # visualization_FeatureMap(z3, self.ha_model, 192, image_path_ha)
        # print("从ha_model中出来Z2 的 大小 {}----------------".format(z2.shape))

        #  返回一个和输入大小相同的张量,其由均值为0、方差为1的标准正态分布填充
        noise = torch.rand_like(z2) - 0.5
        # torch.Size([8, 192, 4, 4])
        z2_noisy = z2 + noise
        # 对z2进行四舍五入  也就是取整函数
        z2_rounded = bypass_round(z2)

        # 对z2_rounded进行反卷积操作
        # torch.Size([8, 192, 4, 4])--> torch.Size([8, 192, 16, 16])
        #  从超先验网络中出来
        h2 = self.hs_model(z2_rounded)
        image_path_ga = "/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/Copy/test_image_kokal/hs"
        # visualization_FeatureMap(z2_rounded, self.hs_model, 192, image_path_ga)

        # image = h2[0]
        # print("image_z3.shape:{}".format(image.shape))
        #
        # image = (image + 1) / 2
        # feature_map = 1
        # for k in image[:, :, :]:
        #     # print(k.shape)
        #     import torchvision
        #     image = torchvision.transforms.functional.to_pil_image(k)
        #     # 将灰度图像转换为RGB格式
        #     image.save('/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/test_kokal/hs/{}.png'.format(feature_map))
        #     feature_map = feature_map + 1
        '''
        z2_sigma  的这个参数是往模型里面加的某一层的层名
        要获取这一层权重的值  就得通过字典的方式来进行访问
        z2_sigma  = net.state_dict()['z2_sigma']
        1 * 192 * 1 * 1
        '''

        # z2_sigma  为全为1   z2_mu 全为0  将z2里面的数据转为均值为1  方差为0的数据
        # torch.Size([1, 192, 1, 1])
        z2_sigma = self.z2_sigma.cuda()
        # torch.Size([1, 192, 1, 1])
        z2_mu = torch.zeros_like(z2_sigma)

        #  前M通道作为语法信息
        # 8 * 16 * 16 *16
        z3_syntax = z3[:, :self.M, :, :]
        # 生成语法  syntax_model： 语法生成模型
        '''
        输入的大小 z3_syntax torch.Size([8, 16, 16, 16])
        '''
        # print("进入模型z3_syntax的大小为：{}------------------".format(z3_syntax.shape))

        z3_syntax = self.syntax_model(z3_syntax)
        # print("从模型出来z3_syntax的大小为：{}------------------".format(z3_syntax.shape))

        # z3_syntax = z3_syntax.view(z3_syntax.shape[0], -1, 1, 1)
        # print("z3_syntax的大小为：{}------------------".format(z3_syntax.shape))
        # N-M个通道作为内容流
        # 8 * 176 * 16 *16
        z3_content = z3[:, self.M:, :, :]

        # Content
        noise = torch.rand_like(z3_content) - 0.5
        z3_content_noisy = z3_content + noise
        '''
         z3_content_rounded    torch.Size([8, 176, 16, 16])
        tensor([[[[0., -0., 0.,  ..., -0., 0., -0.],
              [0., -0., 0.,  ..., -0., -0., -0.],
              [0., -0., -0.,  ..., -0., -0., -0.],
              ...,
              [-0., -0., -0.,  ..., -0., -0., -0.],
              [0., -0., -0.,  ..., -0., -0., -0.],
              [0., -0., 0.,  ..., -0., -0., -0.]],
        '''
        z3_content_rounded = bypass_round(z3_content)

        # Syntax
        noise = torch.rand_like(z3_syntax) - 0.5
        z3_syntax_noisy = z3_syntax + noise

        '''
        torch.Size([8, 16, 16, 16])
        tensor([[[[-0., -0., -0.,  ..., -0., -0., 0.],
              [-0., 0., -0.,  ..., -0., -0., 0.],
              [-0., 0., 0.,  ..., -0., -0., 0.],
        '''
        z3_syntax_rounded = bypass_round(z3_syntax)

        if mode == 'train':
            # z2_noisy是经过超先验网络后的值
            # z2_likelihoods： torch.Size([8, 192, 4, 4])

            z2_likelihoods = self.entropy_bottleneck_z2(z2_noisy, z2_sigma, z2_mu)

            ''' Content
            # h2 是超先验的结果
            # 内容流 将上下文模型和超先验相结合的估计概率进行量化和熵编码
            '''
            # 基于超先验的概率模型生成高斯分布的均值和尺度
            z3_content_mu, z3_content_sigma = self.prediction_model(z3_content_rounded, h2, self.y_sampler,
                                                                    self.h_sampler)
            # 计算累积密度函数和似然
            z3_content_likelihoods = self.entropy_bottleneck_z3(z3_content_noisy, z3_content_sigma, z3_content_mu)

            '''Syntax
            # 神经句法采用基于超先验的概率模型进行熵编码。 由于神经语法不包含空间信息，
            # 因此不应用上下文模型
            '''
            # 基于超先验的概率模型生成高斯分布的均值和尺度
            z3_syntax_sigma, z3_syntax_mu = self.prediction_model_syntax(z3_syntax_rounded, h2)
            # 计算累积密度函数和似然
            z3_syntax_likelihoods = self.entropy_bottleneck_z3_syntax(z3_syntax_noisy, z3_syntax_sigma, z3_syntax_mu)

        else:
            z2_likelihoods = self.entropy_bottleneck_z2(z2_rounded, z2_sigma, z2_mu)

            # Content
            z3_content_mu, z3_content_sigma = self.prediction_model(z3_content_rounded, h2, self.test_y_sampler,
                                                                    self.test_h_sampler)
            z3_content_likelihoods = self.entropy_bottleneck_z3(z3_content_rounded, z3_content_sigma, z3_content_mu)

            # Syntax
            z3_syntax_sigma, z3_syntax_mu = self.prediction_model_syntax(z3_syntax_rounded, h2)
            z3_syntax_likelihoods = self.entropy_bottleneck_z3_syntax(z3_syntax_rounded, z3_syntax_sigma, z3_syntax_mu)

        # 解码
        '''
        输入大小 
        torch.Size([8, 176, 16, 16])
        输出大小：
        torch.Size([8, 16, 256, 256])
        '''
        # 解码
        x_tilde = self.s_model(z3_content_rounded)
        # x_tilde = self.s_model(z3_content)

        # 这些参数在解码时间从权重生成器动态生成
        # 利用多层全连通网络将神经语法表示映射到解码器网络中最终层的核参数
        conv_weights = self.conv_weights_gen(z3_syntax_rounded)
        '''
        conv_weights: 8 * 3 *16 * 1 * 1
        x_tilde :  8 * 16  * 256 * 256
        输出 x_tilde_bf : 8 * 3 * 256 * 256
        '''
        x_tilde_bf = self.batch_conv(conv_weights, x_tilde)

        if self.post_processing:
            x_tilde = self.HAN(x_tilde_bf)
            conv_weights = self.conv_weights_gen_HAN(z3_syntax_rounded)
            x_tilde = self.batch_conv(conv_weights, x_tilde)
            x_tilde = self.add_mean(x_tilde)
        else:
            x_tilde = x_tilde_bf

        num_pixels = inputs.size()[0] * h * w

        if mode == 'train':

            # print(tensor)
            # a = torch.log(z2_likelihoods).sum() / (-math.log(2) * 272556)
            # b = torch.sum(torch.log(z2_likelihoods), [0, 1, 2, 3]) / (-math.log(2) * 272556)
            # print("a", a)
            # print("b", b)

            bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * num_pixels) for l in
                        [z2_likelihoods, z3_content_likelihoods, z3_syntax_likelihoods]]

            train_bpp = bpp_list[0] + bpp_list[1] + bpp_list[2]
            # print("--------inputs[:,:,:h,:w]:{},x_tilde[:,:,:h,:w]:{}".format(inputs[:,:,:h,:w].shape,x_tilde[:,:,:h,:w].shape))
            # train_mse = torch.mean((inputs[:, :, :h, :w] - x_tilde[:, :, :h, :w]) ** 2, [0, 1, 2, 3])
            # train_mse *= 255 ** 2
            train_mse = self.mse(x_tilde, inputs)

            return train_bpp, train_mse


        elif mode == 'test':
            print("x_tilde:{}",x_tilde.shape)
            image = x_tilde[0] # 1 *  3 * 256 * 256

            image =  (image + 1)  / 2
            # image = image.permute(1, 2, 0)
            print("image :{}".format(image.shape))
            import torchvision
            image = torchvision.transforms.functional.to_pil_image(image)

            # image = torchvision.transforms.functional.to_pil_image(image).convert("BGR")
            image.save('/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/Copy/test_image_kokal/2.png')

            test_num_pixels = inputs.size()[0] * th * tw

            bpp_list = [torch.sum(torch.log(l), [0, 1, 2, 3]) / (-np.log(2) * test_num_pixels) for l in
                        [z2_likelihoods, z3_content_likelihoods, z3_syntax_likelihoods]]

            eval_bpp = bpp_list[0] + bpp_list[1] + bpp_list[2]

            # Bring both images back to 0..255 range.
            gt = torch.round((inputs + 1) * 127.5)
            x_hat = torch.clamp((x_tilde + 1) * 127.5, 0, 255)
            x_hat = torch.round(x_hat).float()

            v_mse = torch.mean((x_hat - gt) ** 2, [1, 2, 3])
            v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0)

            return eval_bpp, v_mse, v_psnr
        else:
            print("zzzz")
            return x_tilde