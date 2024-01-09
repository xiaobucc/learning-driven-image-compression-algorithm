import torch
import torch.nn as nn
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from .attention import *
# from SWAM_mask import *
# from .SWAM_mask import *
# from .util import *
from compressai.layers import (
    AttentionBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
import numpy as np
from layers.win_attention import *


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# Block_train(768,256,16)

#   model = Block_train(768,256,16)
class Block_train(nn.Module):
    def __init__(self, out_channel, embed_dim, image_size, patch_size):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, 0., 12)]  # stochastic depth decay rule
        print("dpr:{}".format(dpr))
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(12)])
        self.out_channel = out_channel
        img_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.conv_0 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.fusion0 = nn.Linear(embed_dim, embed_dim // 4)
        # self.conv_block = ResidualBlock(out_channel, out_channel)

        self.fusion1 = nn.Linear(embed_dim, embed_dim // 4)
        self.fusion2 = nn.Linear(embed_dim, embed_dim // 4)
        self.fusion3 = nn.Linear(embed_dim, embed_dim // 4)
        self.fusion = nn.Linear(embed_dim, out_channel)
        self.norm = nn.LayerNorm(out_channel)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 1 * 1 * 768
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))  # 1 * 2 * 768
        self.chans_embed = nn.Linear(out_channel, embed_dim)
        self.pos_drop = nn.Dropout(p=0.)

    def forward(self, input):
        b, c, w, h = input.shape  # 8 * 192 * 16 * 16
        y_hat = input.flatten(2).transpose(1, 2)  # 8 * 256 * 192
        y_hat = self.chans_embed(y_hat)  # 8 * 256  768

        cls_tokens = self.cls_token.expand(b, -1, -1)  # 8 * 1 * 768 stole cls_tokens impl from Phil Wang, thanks
        # print("y_hat:{}   cls_tokens:{}".format(y_hat.shape,cls_tokens.shape))
        y0 = torch.cat((cls_tokens, y_hat), dim=1)  # 8 * 257 * 768
        y0 = y0 + self.pos_embed  # 8 * 257 * 768
        y1 = self.blocks[0](y0)  # # 8 * 257 * 768
        y2 = self.blocks[1](y1)
        y3 = self.blocks[2](y2)
        y4 = self.blocks[3](y3)
        y5 = self.blocks[4](y4)
        y6 = self.blocks[5](y5)
        y7 = self.blocks[6](y6)
        y8 = self.blocks[7](y7)
        y9 = self.blocks[8](y8)
        y10 = self.blocks[9](y9)
        y11 = self.blocks[10](y10)
        y12 = self.blocks[11](y11)

        y0 = self.fusion0(y_hat)

        # print("y_0:{}".format(y0.shape))

        # print("y0:{}".format(y0.shape))  # 8 * 256 * 192
        y1 = self.fusion1(y4[:, 1:])
        # print("y1:{}".format(y1.shape))  # 8 * 256 * 192
        y2 = self.fusion2(y8[:, 1:])
        # print("y2:{}".format(y2.shape))  # 8 * 256 * 192
        y3 = self.fusion3(y12[:, 1:])
        # print("y3:{}".format(y3.shape))  # 8 * 256 * 192

        y_rec = torch.cat((y0, y1, y2, y3), dim=2)
        y_rec = self.fusion(y_rec)
        y_rec = y_rec.transpose(1, 2)
        y_rec = torch.reshape(y_rec, [b, self.out_channel, 16, 16])

        # y_rec = self.conv_0(y_rec)
        # y_rec = self.conv_block(y_rec)
        # print("y_rec:{}".format(y_rec.shape))
        return y_rec


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim  # 96
        self.output_dim = output_dim  # 96
        self.head_dim = head_dim  # 4
        self.scale = self.head_dim ** -0.5  # 0.5
        self.n_heads = input_dim // head_dim  # 24
        self.window_size = window_size  # 4
        self.type = type  # W
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)  # 限制变量的取值范围

        # 24 * 7 * 7
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads).transpose(1,
                                                                                                                 2).transpose(
                0, 1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type != 'W': x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size,
                      p2=self.window_size)  # 8 * 16 * 16 * 96
        h_windows = x.size(1)
        w_windows = x.size(2)
        # 8 * 16 * 16 * 96
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type != 'W': output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2),
                                                 dims=(1, 2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]


class Block_Hyper(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block_Hyper, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # 减少过拟合
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


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def conv5x5(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=stride, padding=2)


class ResidualBlock3_5(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv5x5(out_ch, out_ch)

        self.conv3 = conv3x3(out_ch, out_ch)

        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # print("identity:{}".format(identity.shape))
        out = self.conv1(x)
        out = self.leaky_relu(out)
        # print("out1:{}".format(out.shape))
        out = self.conv2(out)
        out = self.leaky_relu(out)
        # print("out2:{}".format(out.shape))
        out = self.conv3(out)
        out = self.leaky_relu(out)
        # print("out3:{}".format(out.shape))
        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class ResidualBlock5x5(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv5x5(out_ch, out_ch)
        self.conv3 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # print("identity:{}".format(identity.shape))
        out = self.conv2(x)
        out = self.leaky_relu(out)
        # print("out1:{}".format(out.shape))
        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class ResidualBlock3x3(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv3 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # print("identity:{}".format(identity.shape))
        out = self.conv1(x)
        out = self.leaky_relu(out)
        # print("out1:{}".format(out.shape))
        out = self.conv3(out)
        out = self.leaky_relu(out)
        # print("out3:{}".format(out.shape))
        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


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


class Unet(nn.Module):
    def __init__(self, inchannels, outchannels, num_heads, depth):
        super().__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels

        self.num_heads = num_heads

        self.depth = depth

        self.SpatialTransformer1 = SpatialTransformer(96, self.num_heads, d_head=(96 // self.num_heads),
                                                      depth=self.depth)

        self.ResBlock1 = ResidualBottleneck(96)

        self.SpatialTransformer2 = SpatialTransformer(128, self.num_heads, d_head=(128 // self.num_heads),
                                                      depth=self.depth)

        self.ResBlock2 = ResidualBottleneck(128)

        self.SpatialTransformer3 = SpatialTransformer(256, self.num_heads, d_head=(256 // self.num_heads),
                                                      depth=self.depth)

        self.ResBlock3 = ResidualBottleneck(256)

        self.down1 = nn.Conv2d(self.inchannels, 256, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.up1 = nn.ConvTranspose2d(512, 256, 5, 2, 2,
                                      output_padding=1, bias=True)
        self.up2 = nn.ConvTranspose2d(256, 192, 5, 2, 2,
                                      output_padding=1, bias=True)
        self.up3 = nn.ConvTranspose2d(512, 256, 1, 1, 0,
                                      bias=True)
        self.up4 = nn.ConvTranspose2d(384, 192, 1, 1, 0,
                                      bias=True)

        self.middle = nn.Sequential(
            ResidualBottleneck(512),
            SpatialTransformer(512, self.num_heads, d_head=512 // self.num_heads, depth=self.depth),
            ResidualBottleneck(512)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # print("yunxing daozheli ")
        # 下采样
        conv_down_x, trans_down_x = torch.split(x, (x.shape[1] // 2, x.shape[1] // 2), dim=1)
        conv_down_x1 = self.ResBlock1(conv_down_x)
        trans_down_x1 = self.SpatialTransformer1(trans_down_x)
        down_x1 = self.down1(torch.cat((conv_down_x1, trans_down_x1), dim=1))  # 8 * 256 * 8 * 8
        down_x1 = self.relu(down_x1)
        conv_down_y, trans_down_y = torch.split(down_x1, (down_x1.shape[1] // 2, down_x1.shape[1] // 2), dim=1)
        conv_down_y1 = self.ResBlock2(conv_down_y)
        trans_down_y1 = self.SpatialTransformer2(trans_down_y)
        down_x2 = self.down2(torch.cat((conv_down_y1, trans_down_y1), dim=1))  # 8 * 512 * 4 * 4
        down_x2 = self.relu(down_x2)
        middle_x = self.middle(down_x2)  # 8 * 512 * 4 * 4

        # 上采样
        conv_up_x, trans_up_x = torch.split(middle_x, (middle_x.shape[1] // 2, middle_x.shape[1] // 2), dim=1)
        conv_up_x1 = self.ResBlock3(conv_up_x)  # 8 * 256 * 4 * 4
        trans_up_x1 = self.SpatialTransformer3(trans_up_x)
        up_x1 = self.up1(torch.cat((conv_up_x1, trans_up_x1), dim=1))  # 8 * 512 * 8 * 8
        up_x1 = self.relu(up_x1)

        up_x1 = torch.cat((up_x1, down_x1), dim=1)  # 8 * 512 * 8 * 8
        up_x1 = self.up3(up_x1)  # 8 * 256 * 8 * 8
        up_x1 = self.relu(up_x1)
        conv_up_y, trans_up_y = torch.split(up_x1, (up_x1.shape[1] // 2, up_x1.shape[1] // 2), dim=1)
        conv_up_y1 = self.ResBlock2(conv_up_y)  # 8 * 128 * 8 * 8
        trans_up_y1 = self.SpatialTransformer2(trans_up_y)
        up_x2 = self.up2(torch.cat((conv_up_y1, trans_up_y1), dim=1))  # 8 * 192 * 16 * 16
        up_x2 = self.relu(up_x2)
        up_x2 = torch.cat((up_x2, x), dim=1)  # 8 * 384 * 16 * 16
        out = self.up4(up_x2)  # 8 * 192 * 8 * 8

        # print("out.shape:",out.shape)
        return out


class Unet_new(nn.Module):
    def __init__(self, inchannels, outchannels, num_heads, depth):
        super().__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels

        self.num_heads = num_heads

        self.depth = depth

        self.SpatialTransformer1 = SpatialTransformer(96, self.num_heads, d_head=(96 // self.num_heads),
                                                      depth=self.depth)

        self.ResBlock1 = ResidualBottleneck(96)

        self.SpatialTransformer2 = SpatialTransformer(128, self.num_heads, d_head=(128 // self.num_heads),
                                                      depth=self.depth)

        self.ResBlock2 = ResidualBottleneck(128)

        self.SpatialTransformer3 = SpatialTransformer(256, self.num_heads, d_head=(256 // self.num_heads),
                                                      depth=self.depth)

        self.ResBlock3 = ResidualBottleneck(256)
        self.conv1 = nn.Conv2d(96, 96, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(128, 128, 1, 1, 0, bias=True)
        self.down1 = nn.Conv2d(self.inchannels, 256, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.up1 = nn.ConvTranspose2d(512, 256, 5, 2, 2,
                                      output_padding=1, bias=True)
        self.up2 = nn.ConvTranspose2d(256, 192, 5, 2, 2,
                                      output_padding=1, bias=True)
        self.up3 = nn.ConvTranspose2d(512, 256, 1, 1, 0,
                                      bias=True)
        self.up4 = nn.ConvTranspose2d(384, 192, 1, 1, 0,
                                      bias=True)
        self.conv3 = nn.ConvTranspose2d(256, 256, 1, 1, 0,
                                        bias=True)
        self.conv4 = nn.ConvTranspose2d(128, 128, 1, 1, 0,
                                        bias=True)

        self.middle = nn.Sequential(
            ResidualBottleneck(512),
            SpatialTransformer(512, self.num_heads, d_head=512 // self.num_heads, depth=self.depth),
            ResidualBottleneck(512)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # print("yunxing daozheli ")
        # 下采样
        trans_down_x, conv_down_x = torch.split(x, (x.shape[1] // 2, x.shape[1] // 2), dim=1)
        conv_down_x1 = self.conv1(conv_down_x)
        trans_down_x1 = self.SpatialTransformer1(trans_down_x)
        down_x1 = self.down1(torch.cat((conv_down_x1, trans_down_x1), dim=1))  # 8 * 256 * 8 * 8
        down_x1 = self.relu(down_x1)
        conv_down_y, trans_down_y = torch.split(down_x1, (down_x1.shape[1] // 2, down_x1.shape[1] // 2), dim=1)
        conv_down_y1 = self.conv2(conv_down_y)
        trans_down_y1 = self.SpatialTransformer2(trans_down_y)
        down_x2 = self.down2(torch.cat((conv_down_y1, trans_down_y1), dim=1))  # 8 * 512 * 4 * 4
        down_x2 = self.relu(down_x2)
        middle_x = self.middle(down_x2)  # 8 * 512 * 4 * 4

        # 上采样
        trans_up_x, conv_up_x = torch.split(middle_x, (middle_x.shape[1] // 2, middle_x.shape[1] // 2), dim=1)
        conv_up_x1 = self.conv3(conv_up_x)  # 8 * 256 * 4 * 4
        trans_up_x1 = self.SpatialTransformer3(trans_up_x)
        up_x1 = self.up1(torch.cat((conv_up_x1, trans_up_x1), dim=1))  # 8 * 512 * 8 * 8
        up_x1 = self.relu(up_x1)

        up_x1 = torch.cat((up_x1, down_x1), dim=1)  # 8 * 512 * 8 * 8
        up_x1 = self.up3(up_x1)  # 8 * 256 * 8 * 8
        up_x1 = self.relu(up_x1)
        conv_up_y, trans_up_y = torch.split(up_x1, (up_x1.shape[1] // 2, up_x1.shape[1] // 2), dim=1)
        conv_up_y1 = self.conv4(conv_up_y)  # 8 * 128 * 8 * 8
        trans_up_y1 = self.SpatialTransformer2(trans_up_y)
        up_x2 = self.up2(torch.cat((conv_up_y1, trans_up_y1), dim=1))  # 8 * 192 * 16 * 16
        up_x2 = self.relu(up_x2)
        up_x2 = torch.cat((up_x2, x), dim=1)  # 8 * 384 * 16 * 16
        out = self.up4(up_x2)  # 8 * 192 * 8 * 8

        # print("out.shape:",out.shape)
        return out


class Unet_ha_hs(nn.Module):
    def __init__(self, inchannels, outchannels, num_heads, depth):
        super().__init__()
        self.inchannels = inchannels
        self.out_channels = outchannels
        self.num_heads = num_heads

        self.depth = depth

        self.SpatialTransformer1 = SpatialTransformer(inchannels // 2, self.num_heads, d_head=(96 // self.num_heads),
                                                      depth=self.depth)

        self.ResBlock1 = ResidualBottleneck(96)

        self.SpatialTransformer2 = SpatialTransformer(128, self.num_heads, d_head=(128 // self.num_heads),
                                                      depth=self.depth)

        self.ResBlock2 = ResidualBottleneck(128)

        self.SpatialTransformer3 = SpatialTransformer(256, self.num_heads, d_head=(256 // self.num_heads),
                                                      depth=self.depth)

        self.ResBlock3 = ResidualBottleneck(256)
        self.conv1 = nn.Conv2d(inchannels // 2, inchannels // 2, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(128, 128, 1, 1, 0, bias=True)
        self.down1 = nn.Conv2d(self.inchannels, 256, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.up1 = nn.ConvTranspose2d(512, 256, 5, 2, 2,
                                      output_padding=1, bias=True)
        self.up2 = nn.ConvTranspose2d(256, 320, 5, 2, 2,
                                      output_padding=1, bias=True)
        self.up3 = nn.ConvTranspose2d(512, 256, 1, 1, 0,
                                      bias=True)
        self.up4 = nn.ConvTranspose2d(640, self.out_channels, 1, 1, 0,
                                      bias=True)
        self.conv3 = nn.ConvTranspose2d(256, 256, 1, 1, 0,
                                        bias=True)
        self.conv4 = nn.ConvTranspose2d(128, 128, 1, 1, 0,
                                        bias=True)

        self.middle = nn.Sequential(
            ResidualBottleneck(512),
            SpatialTransformer(512, self.num_heads, d_head=512 // self.num_heads, depth=self.depth),
            ResidualBottleneck(512)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # print("yunxing daozheli ")
        # 下采样
        trans_down_x, conv_down_x = torch.split(x, (x.shape[1] // 2, x.shape[1] // 2), dim=1)
        conv_down_x1 = self.conv1(conv_down_x)
        trans_down_x1 = self.SpatialTransformer1(trans_down_x)
        down_x1 = self.down1(torch.cat((conv_down_x1, trans_down_x1), dim=1))  # 8 * 256 * 8 * 8
        down_x1 = self.relu(down_x1)
        conv_down_y, trans_down_y = torch.split(down_x1, (down_x1.shape[1] // 2, down_x1.shape[1] // 2), dim=1)
        conv_down_y1 = self.conv2(conv_down_y)
        trans_down_y1 = self.SpatialTransformer2(trans_down_y)
        down_x2 = self.down2(torch.cat((conv_down_y1, trans_down_y1), dim=1))  # 8 * 512 * 4 * 4
        down_x2 = self.relu(down_x2)
        middle_x = self.middle(down_x2)  # 8 * 512 * 4 * 4
        # 上采样
        trans_up_x, conv_up_x = torch.split(middle_x, (middle_x.shape[1] // 2, middle_x.shape[1] // 2), dim=1)
        conv_up_x1 = self.conv3(conv_up_x)  # 8 * 256 * 4 * 4
        trans_up_x1 = self.SpatialTransformer3(trans_up_x)
        up_x1 = self.up1(torch.cat((conv_up_x1, trans_up_x1), dim=1))  # 8 * 256 * 8 * 8
        up_x1 = self.relu(up_x1)

        up_x1 = torch.cat((up_x1, down_x1), dim=1)  # 8 * 512 * 8 * 8
        up_x1 = self.up3(up_x1)  # 8 * 256 * 8 * 8
        up_x1 = self.relu(up_x1)
        conv_up_y, trans_up_y = torch.split(up_x1, (up_x1.shape[1] // 2, up_x1.shape[1] // 2), dim=1)
        conv_up_y1 = self.conv4(conv_up_y)  # 8 * 128 * 8 * 8
        trans_up_y1 = self.SpatialTransformer2(trans_up_y)
        up_x2 = self.up2(torch.cat((conv_up_y1, trans_up_y1), dim=1))  # 8 * 320 * 16 * 16
        up_x2 = self.relu(up_x2)
        up_x2 = torch.cat((up_x2, x), dim=1)  # 8 * 640 * 16 * 16
        out = self.up4(up_x2)  # 8 * 320 * 8 * 8

        # print("out.shape:",out.shape)
        return out


class Unet_ha(nn.Module):
    def __init__(self, inchannels, num_heads, depth):
        super().__init__()
        self.inchannels = inchannels

        self.num_heads = num_heads

        self.depth = depth

        self.SpatialTransformer1 = SpatialTransformer(inchannels // 2, self.num_heads, d_head=(96 // self.num_heads),
                                                      depth=self.depth)

        self.ResBlock1 = ResidualBottleneck(96)

        self.SpatialTransformer2 = SpatialTransformer(128, self.num_heads, d_head=(128 // self.num_heads),
                                                      depth=self.depth)

        self.ResBlock2 = ResidualBottleneck(128)

        self.SpatialTransformer3 = SpatialTransformer(256, self.num_heads, d_head=(256 // self.num_heads),
                                                      depth=self.depth)

        self.ResBlock3 = ResidualBottleneck(256)
        self.conv1 = nn.Conv2d(inchannels // 2, inchannels // 2, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(128, 128, 1, 1, 0, bias=True)
        self.down1 = nn.Conv2d(self.inchannels, 256, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.middle = nn.Sequential(
            ResidualBottleneck(512),
            SpatialTransformer(512, self.num_heads, d_head=512 // self.num_heads, depth=self.depth),
            ResidualBottleneck(512)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # print("yunxing daozheli ")
        # 下采样
        trans_down_x, conv_down_x = torch.split(x, (x.shape[1] // 2, x.shape[1] // 2), dim=1)
        conv_down_x1 = self.conv1(conv_down_x)
        trans_down_x1 = self.SpatialTransformer1(trans_down_x)
        down_x1 = self.down1(torch.cat((conv_down_x1, trans_down_x1), dim=1))  # 8 * 160 * 16 * 8
        down_x1 = self.relu(down_x1)
        conv_down_y, trans_down_y = torch.split(down_x1, (down_x1.shape[1] // 2, down_x1.shape[1] // 2), dim=1)
        conv_down_y1 = self.conv2(conv_down_y)
        trans_down_y1 = self.SpatialTransformer2(trans_down_y)
        down_x2 = self.down2(torch.cat((conv_down_y1, trans_down_y1), dim=1))  # 8 * 512 * 4 * 4
        down_x2 = self.relu(down_x2)
        middle_x = self.middle(down_x2)  # 8 * 512 * 4 * 4

        out = middle_x

        return out, middle_x, down_x1, x


class Unet_hs(nn.Module):
    def __init__(self, out_channels, num_heads, depth):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.depth = depth

        self.SpatialTransformer2 = SpatialTransformer(128, self.num_heads, d_head=(128 // self.num_heads),
                                                      depth=self.depth)

        self.SpatialTransformer3 = SpatialTransformer(256, self.num_heads, d_head=(256 // self.num_heads),
                                                      depth=self.depth)
        self.up1 = nn.ConvTranspose2d(512, 256, 5, 2, 2,
                                      output_padding=1, bias=True)
        self.up2 = nn.ConvTranspose2d(256, 320, 5, 2, 2,
                                      output_padding=1, bias=True)
        self.up3 = nn.ConvTranspose2d(512, 256, 1, 1, 0,
                                      bias=True)
        self.up4 = nn.ConvTranspose2d(640, self.out_channels, 1, 1, 0,
                                      bias=True)
        self.conv3 = nn.ConvTranspose2d(256, 256, 1, 1, 0,
                                        bias=True)
        self.conv4 = nn.ConvTranspose2d(128, 128, 1, 1, 0,
                                        bias=True)

        self.relu = nn.ReLU()

    def forward(self, x, middle_x, down_x1, input):
        # 上采样
        trans_up_x, conv_up_x = torch.split(middle_x, (middle_x.shape[1] // 2, middle_x.shape[1] // 2), dim=1)
        conv_up_x1 = self.conv3(conv_up_x)  # 8 * 256 * 4 * 4
        trans_up_x1 = self.SpatialTransformer3(trans_up_x)
        up_x1 = self.up1(torch.cat((conv_up_x1, trans_up_x1), dim=1))  # 8 * 256 * 8 * 8
        up_x1 = self.relu(up_x1)

        up_x1 = torch.cat((up_x1, down_x1), dim=1)  # 8 * 512 * 8 * 8
        up_x1 = self.up3(up_x1)  # 8 * 256 * 8 * 8
        up_x1 = self.relu(up_x1)
        conv_up_y, trans_up_y = torch.split(up_x1, (up_x1.shape[1] // 2, up_x1.shape[1] // 2), dim=1)
        conv_up_y1 = self.conv4(conv_up_y)  # 8 * 128 * 8 * 8
        trans_up_y1 = self.SpatialTransformer2(trans_up_y)
        up_x2 = self.up2(torch.cat((conv_up_y1, trans_up_y1), dim=1))  # 8 * 320 * 16 * 16
        up_x2 = self.relu(up_x2)
        up_x2 = torch.cat((up_x2, input), dim=1)  # 8 * 640 * 16 * 16
        out = self.up4(up_x2)  # 8 * 320 * 8 * 8

        # print("out.shape:",out.shape)
        return out


class Unet_ha_new(nn.Module):
    def __init__(self, inchannels, num_heads, depth):
        super().__init__()
        self.inchannels = inchannels

        self.num_heads = num_heads

        self.depth = depth

        self.SpatialTransformer1 = WinBasedAttention(inchannels // 2, self.num_heads, window_size=4, shift_size=2)

        self.ResBlock1 = ResidualBottleneck(96)

        self.SpatialTransformer2 = WinBasedAttention(128, self.num_heads, window_size=4, shift_size=2)

        self.ResBlock2 = ResidualBottleneck(128)

        # self.SpatialTransformer3 = Win_noShift_Attention(256, self.num_heads, d_head=(256 // self.num_heads),shift_size=4,
        #                                               depth=self.depth)

        self.ResBlock3 = ResidualBottleneck(256)
        # self.conv1 = nn.Conv2d(inchannels //2,inchannels //2, 1, 1, 0, bias=True)
        self.conv1 = ResidualBlock3_5(inchannels // 2, inchannels // 2)

        self.conv2 = ResidualBlock5x5(128, 128)

        self.down0 = nn.Conv2d(self.inchannels, self.inchannels, 1, 1, 0, bias=True)

        self.down1 = nn.Conv2d(self.inchannels, 256, kernel_size=3, stride=2, padding=1)

        self.down2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.down3 = nn.Conv2d(256, 256, 1, 1, 0, bias=True)

        self.middle = nn.Sequential(
            ResidualBottleneck(512),
            WinBasedAttention(512, self.num_heads, window_size=2, shift_size=1),
            ResidualBottleneck(512)
        )
        self.relu = nn.GELU()

    def forward(self, x):
        # print("yunxing daozheli ")
        # 下采样
        trans_down_x, conv_down_x = torch.split(x, (x.shape[1] // 2, x.shape[1] // 2), dim=1)
        conv_down_x1 = self.conv1(conv_down_x)
        trans_down_x1 = self.SpatialTransformer1(trans_down_x)
        down_x1 = self.down0(torch.cat((conv_down_x1, trans_down_x1), dim=1))  # 8 * 256 * 8 * 8
        down_x1 = down_x1 + x
        down_x1 = self.down1(down_x1)  # 8 * 256 * 8 * 8
        down_x1 = self.relu(down_x1)

        # 第二次下采样
        conv_down_y, trans_down_y = torch.split(down_x1, (down_x1.shape[1] // 2, down_x1.shape[1] // 2), dim=1)
        conv_down_y1 = self.conv2(conv_down_y)
        trans_down_y1 = self.SpatialTransformer2(trans_down_y)
        down_x2 = self.down3(torch.cat((conv_down_y1, trans_down_y1), dim=1))  # 8 * 256 * 8 * 8
        down_x2 = down_x2 + down_x1
        down_x2 = self.down2(down_x2)  # 8 * 512 * 4 * 4
        down_x2 = self.relu(down_x2)
        middle_x = self.middle(down_x2)  # 8 * 512 * 4 * 4

        out = middle_x
        # print("middle_x:{}".format(middle_x.shape))
        return out, middle_x, down_x1, x


class Unet_hs_new(nn.Module):
    def __init__(self, out_channels, num_heads, depth):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.depth = depth

        self.SpatialTransformer2 = WinBasedAttention(128, self.num_heads, window_size=2, shift_size=1)

        self.SpatialTransformer3 = WinBasedAttention(256, self.num_heads, window_size=2, shift_size=1)
        self.up0 = nn.Conv2d(512, 512, 1, 1, 0, bias=True)
        self.up1 = nn.ConvTranspose2d(512, 256, 5, 2, 2,
                                      output_padding=1, bias=True)
        self.up2 = nn.ConvTranspose2d(256, 192, 5, 2, 2,
                                      output_padding=1, bias=True)
        self.up3 = nn.ConvTranspose2d(512, 256, 1, 1, 0,
                                      bias=True)
        self.up4 = nn.ConvTranspose2d(384, self.out_channels, 1, 1, 0,
                                      bias=True)

        self.up5 = nn.Conv2d(256, 256, 1, 1, 0, bias=True)

        self.conv3 = ResidualBlock3x3(256, 256)
        self.conv4 = ResidualBlock3x3(128, 128)

        self.relu = nn.GELU()

    def forward(self, x, middle_x, down_x1, input):
        # 上采样
        trans_up_x, conv_up_x = torch.split(middle_x, (middle_x.shape[1] // 2, middle_x.shape[1] // 2), dim=1)
        conv_up_x1 = self.conv3(conv_up_x)  # 8 * 256 * 4 * 4
        trans_up_x1 = self.SpatialTransformer3(trans_up_x)
        up_x1 = self.up0(torch.cat((conv_up_x1, trans_up_x1), dim=1))  # 8 * 256 * 8 * 8
        up_x1 = up_x1 + middle_x
        up_x1 = self.up1(up_x1)  # 8 * 256 * 8 * 8
        up_x1 = self.relu(up_x1)

        up_x1 = torch.cat((up_x1, down_x1), dim=1)  # 8 * 512 * 8 * 8
        up_x1 = self.up3(up_x1)  # 8 * 256 * 8 * 8
        up_x1 = self.relu(up_x1)
        conv_up_y, trans_up_y = torch.split(up_x1, (up_x1.shape[1] // 2, up_x1.shape[1] // 2), dim=1)
        conv_up_y1 = self.conv4(conv_up_y)  # 8 * 128 * 8 * 8
        trans_up_y1 = self.SpatialTransformer2(trans_up_y)
        up_x2 = self.up5(torch.cat((conv_up_y1, trans_up_y1), dim=1))
        up_x2 = up_x2 + up_x1
        up_x2 = self.up2(up_x2)  # 8 * 320 * 16 * 16
        up_x2 = self.relu(up_x2)
        up_x2 = torch.cat((up_x2, input), dim=1)  # 8 * 640 * 16 * 16
        out = self.up4(up_x2)
        # print("out:{}".format(out.shape))
        return out