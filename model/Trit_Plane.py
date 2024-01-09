# import cv2
# import numpy as np
# import torch
import matplotlib.pyplot as plt
import numpy as np
image_path = "/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/DIV2K_train_HR/DIV2K_train_HR/0001.png"
# # 读取彩色图像
# img = cv2.imread('/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/DIV2K_train_HR/DIV2K_train_HR/0001.png')
# # 分离三个颜色通道
# print(type(img))
# img = torch.from_numpy(img)
# print(img.shape)
# b, g, r = torch.split(img)
import numpy as np
from collections import defaultdict


import cv2
# 读取彩色图像
img = cv2.imread(image_path)
# 分离三个颜色通道
b, g, r = cv2.split(img)

# 定义差分编码函数
def diff_encode(channel):
    diff = [channel[0]]
    # print(len(channel))
    for i in range(1, len(channel)):
        diff.append(channel[i] - channel[i-1])
    return np.array(diff, dtype=np.int16)


# 对三个颜色通道进行差分编码和Huffman编码
b_diff = diff_encode(b)
g_diff = diff_encode(g)
r_diff = diff_encode(r)
# 定义Huffman编码函数

huff_dict = {}
def huffman_encode(data):
    freq_dict = defaultdict(int)
    for val in data:
        freq_dict[val] += 1
    freq_list = [(key, freq_dict[key]) for key in freq_dict]
    freq_list.sort(key=lambda x: x[1], reverse=True)

    for i, val in enumerate(freq_list):
        huff_dict[val[0]] = i
    huff_data = [huff_dict[val] for val in data]
    return huff_data

b_huff = huffman_encode(b_diff)
g_huff = huffman_encode(g_diff)
r_huff = huffman_encode(r_diff)

# 将三个颜色通道的编码结果合并
result = np.concatenate((b_huff, g_huff, r_huff))

cv2.imwrite("/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/DIV2K_train_HR/0.png",result)


# 开始解码

# 解压缩时，将编码流还原为三个颜色通道的编码结果
b_huff_restore = result[:b_huff.size]
g_huff_restore = result[b_huff.size:b_huff.size+g_huff.size]
r_huff_restore = result[b_huff.size+g_huff.size:]


# 定义Huffman解码函数
def huffman_decode(data, huff_dict):
    return [huff_dict[val] for val in data]


# 对三个颜色通道进行Huffman解码和差分解码
b_diff_restore = np.array(huffman_decode(b_huff_restore, huff_dict), dtype=np.int16)
g_diff_restore = np.array(huffman_decode(g_huff_restore, huff_dict), dtype=np.int16)
r_diff_restore = np.array(huffman_decode(r_huff_restore, huff_dict), dtype=np.int16)


def diff_decode(diff):
    channel = np.zeros_like(diff)
    channel[0] = diff[0]
    for i in range(1, diff.size):
        channel[i] = channel[i-1] + diff[i]
    return channel


b_restore = diff_decode(b_diff_restore)
g_restore = diff_decode(g_diff_restore)
r_restore = diff_decode(r_diff_restore)
# 将三个颜色通道合并成原始图像
img_restore = cv2.merge((b_restore, g_restore, r_restore))

cv2.imwrite("/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/DIV2K_train_HR/1.png",img_restore)
