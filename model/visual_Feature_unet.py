
from torch import nn
import numpy as np

import math
import torchvision
import argparse
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import seaborn as sns
  # Set figure parameters
large = 24; med = 24; small = 24
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (48, 32),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)   # 全局配置字典  设置默认的绘图参数
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
# plt.rc('font', **{'family': 'Times New Roman'})
plt.rcParams['axes.unicode_minus'] = False


def heatmap(data, camp='bwr', figsize=(48, 32), ax=None, save_path=None):
    plt.figure(figsize=figsize, dpi=100)

    ax = sns.heatmap(data,
                xticklabels=False,
                yticklabels=False, cmap=camp,
                # vmax=0.2,
                # vmin=0,
                center = 0,
                 annot=False, ax=ax, cbar=False, annot_kws={"size": 24}, fmt='.2f')
    #   =========================== Add a **nicer** colorbar on top of the figure. Works for matplotlib 3.3. For later versions, use matplotlib.colorbar
    #   =========================== or you may simply ignore these and set cbar=True in the heatmap function above.
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    # from mpl_toolkits.axes_grid1.colorbar import colorbar
    from matplotlib.pyplot import colorbar
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('top', size='5%', pad='2%')
    colorbar(ax.get_children()[0], cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    #   ================================================================
    #   ================================================================
    plt.savefig(save_path)


def visual_FeatureMap_unet(inputs, model, channel, image_path,middle_x,down_x1 ,input, visual_iamge=0, start_layer=0, end_layer=-1, ):
    '''
    channel: 可视化多少通道
    visual_image: 可视化第几张图
    start_layer: 如果是Sequential容器，从哪一层开始可视化
    end_layer: 到哪一层结束
    '''
    # print(model)
    if isinstance(model, nn.Sequential):
        print("model is nn.Sequential type")
        conv_module = []
        for module in (model.children()):
            conv_module.append(module)
        conv_layer = conv_module[:]

        # 将取到的网络层重新放入Sequential中
        model = nn.Sequential(*list(conv_layer))
    else:
        model = model
    # print(conv_layer)
    image_mid = model(inputs,middle_x,down_x1 ,input)
    # print(model)
    # print(inputs)
    # print("从第三层网络出来后image_mid：{}".format(image_mid.shape))
    # print("image_mid", image_mid.size)
    # image_mid = image_mid[visual_iamge]
    image_mid = image_mid[visual_iamge]
    # 将数据恢复
    # image_mid = (image_mid + 1) / 2
    # print("image_mid",image_mid.shape)
    # print("channel",channel)
    for num in range(channel):
        # print("num",num)
        image = image_mid[num, :, :]
        # print("image.shape:{}".format(image.shape))
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
        # for i in range(0, 13):
        #     im_color = cv2.applyColorMap(image_numpy, i)
        #     cv2.imwrite(image_path + "/orign_{}_{}.png".format(num, i),
        #                 im_color)

        im_color = cv2.applyColorMap(image_numpy, cv2.COLORMAP_JET)
        im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path + "/orign_{}.png".format(num),
                    im_color)


def visual_FeatureMap_unet_heat(inputs, model, channel, image_path,middle_x,down_x1 ,input, visual_iamge=0, start_layer=0, end_layer=-1, ):
    '''
    channel: 可视化多少通道
    visual_image: 可视化第几张图
    start_layer: 如果是Sequential容器，从哪一层开始可视化
    end_layer: 到哪一层结束
    '''
    # print(model)
    if isinstance(model, nn.Sequential):
        print("model is nn.Sequential type")
        conv_module = []
        for module in (model.children()):
            conv_module.append(module)
        conv_layer = conv_module[:]

        # 将取到的网络层重新放入Sequential中
        model = nn.Sequential(*list(conv_layer))
    else:
        model = model
    # print(conv_layer)
    image_mid = model(inputs,middle_x,down_x1 ,input)
    # print(model)
    # print(inputs)
    # print("从第三层网络出来后image_mid：{}".format(image_mid.shape))
    # print("image_mid", image_mid.size)
    # image_mid = image_mid[visual_iamge]
    image_mid = image_mid[visual_iamge]
    # 将数据恢复
    # image_mid = (image_mid + 1) / 2
    # print("image_mid",image_mid.shape)
    # print("channel",channel)

    # image_mid = image_mid.mean(dim=0)
    # print("image_mid",image_mid.shape)
    # image = image_mid
    # image = image.detach().cpu().numpy()
    # heatmap(image,save_path=image_path + "/111111.png")

    for num in range(channel):
        # print("num",num)
        image = image_mid[num, :, :]
        # print("image.shape:{}".format(image.shape))
        image = image.detach().cpu().numpy()
        heatmap(image,save_path=image_path + "/{}.png".format(num))


#
# class BilinearInterpolation(object):
#     def __init__(self, w_rate: float, h_rate: float, *, align='center'):
#         if align not in ['center', 'left']:
#             logging.exception(f'{align} is not a valid align parameter')
#             align = 'center'
#         self.align = align
#         self.w_rate = w_rate
#         self.h_rate = h_rate
#
#     def set_rate(self,w_rate: float, h_rate: float):
#         self.w_rate = w_rate    # w 的缩放率
#         self.h_rate = h_rate    # h 的缩放率
#
#     # 由变换后的像素坐标得到原图像的坐标    针对高
#     def get_src_h(self, dst_i,source_h,goal_h) -> float:
#         if self.align == 'left':
#             # 左上角对齐
#             src_i = float(dst_i * (source_h/goal_h))
#         elif self.align == 'center':
#             # 将两个图像的几何中心重合。
#             src_i = float((dst_i + 0.5) * (source_h/goal_h) - 0.5)
#         src_i += 0.001
#         src_i = max(0.0, src_i)
#         src_i = min(float(source_h - 1), src_i)
#         return src_i
#     # 由变换后的像素坐标得到原图像的坐标    针对宽
#     def get_src_w(self, dst_j,source_w,goal_w) -> float:
#         if self.align == 'left':
#             # 左上角对齐
#             src_j = float(dst_j * (source_w/goal_w))
#         elif self.align == 'center':
#             # 将两个图像的几何中心重合。
#             src_j = float((dst_j + 0.5) * (source_w/goal_w) - 0.5)
#         src_j += 0.001
#         src_j = max(0.0, src_j)
#         src_j = min((source_w - 1), src_j)
#         return src_j
#
#     def transform(self, img):
#         source_h, source_w, source_c = img.shape  # (235, 234, 3)
#         goal_h, goal_w = round(
#             source_h * self.h_rate), round(source_w * self.w_rate)
#         new_img = np.zeros((goal_h, goal_w, source_c), dtype=np.uint8)
#
#         for i in range(new_img.shape[0]):       # h
#             src_i = self.get_src_h(i,source_h,goal_h)
#             for j in range(new_img.shape[1]):
#                 src_j = self.get_src_w(j,source_w,goal_w)
#                 i2 = ceil(src_i)
#                 i1 = int(src_i)
#                 j2 = ceil(src_j)
#                 j1 = int(src_j)
#                 x2_x = j2 - src_j
#                 x_x1 = src_j - j1
#                 y2_y = i2 - src_i
#                 y_y1 = src_i - i1
#                 new_img[i, j] = img[i1, j1]*x2_x*y2_y + img[i1, j2] * \
#                     x_x1*y2_y + img[i2, j1]*x2_x*y_y1 + img[i2, j2]*x_x1*y_y1
#         return new_img
