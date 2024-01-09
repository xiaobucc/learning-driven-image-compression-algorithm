import argparse
import glob

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

from model.net_ga import Net

from model.visual_Feature_unet import *
def val(data_path, weight_path, lmbda, is_high, post_processing, pre_processing, tune_iter):
    images = list(sorted(glob.glob(data_path)))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("images[20]:",images[20])
    if pre_processing is False:

        list_eval_bpp = 0.
        list_v_psnr = 0.
        list_v_mse = 0.
        cnt = 0
        sum_time = 0.
        num = 1
        for img_name in images[22:23]:
            import torchvision.transforms as transforms
            from PIL import Image
            print("img_name:",img_name)
            data = Image.open(img_name)
            # print("data:{}", data)
            data = tv.transforms.ToTensor()(data)
            #
            # data = data.to(device)
            # print("data.get_device():",data.get_device())
            c, h, w = data.shape
            # print("h:{}  w:{}".format(h, w))
            # if (h < 256 or w < 256):
            #     print("-------------------pass-------------")
            #     continue
            # transform = tv.transforms.Compose([
            #     tv.transforms.CenterCrop(256),
            # ])
            # data = transform(data)
            image = data
            print("data:{}", data.shape)
            print("image :{}".format(image.shape))
            import torchvision
            image = torchvision.transforms.functional.to_pil_image(image)
            image.save('/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/Net_WAM_CTM/Net_WAM_CTM_Spilt/1.png')
            # image.save('/root/data/2021/buxiaobu/Neural_Syntax/Pretrain_model/image/1.png'.format(num))

            # image.save('/root/data/2021/buxiaobu/Neural_Syntax/shifit/image/{}_1.png'.format(num))

            # import cv2
            #
            # img = cv2.imread(img_name)
            # print("img:{}",img.size)
            # np.transpose(img,(2,1,0))[::-1,:,:]
            # # 将灰度图像转换为RGB格式
            # cv2.imwrite('/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/Test_WAM/orign/{}.png'.format(num), img)

            _, h, w = data.shape
            h_padded = h
            w_padded = w
            if h % 64 != 0:
                h_padded = (h // 64) * 64 + 64
            if w % 64 != 0:
                w_padded = (w // 64) * 64 + 64
            padding_h = h_padded - h
            padding_w = w_padded - w
            # data 3 * 512 * 769
            h_pad_zeros = torch.ones(3, padding_h, w)
            w_pad_zeros = torch.ones(3, h_padded, padding_w)
            data = torch.cat((data, h_pad_zeros), 1)
            data = torch.cat((data, w_pad_zeros), 2)

            data = data.unsqueeze(0)
            data = data * 2.0 - 1.0   # 归一化的操作

            # print("data:",data)
            data_bpp = 0.0
            data_mse = 0.0

            net = Net((1, h, w, 3), (1, h, w, 3), is_high, post_processing).cuda()
            net.load_state_dict(torch.load(weight_path), strict=True)

            begin_time = time.time()

            with torch.no_grad():
                eval_bpp, v_mse, v_psnr = net(data.cuda(), 'test', num)
            num = num + 1
            end_time = time.time()

            sum_time += end_time - begin_time

            list_eval_bpp += eval_bpp.mean().item()
            list_v_psnr += v_psnr.mean().item()
            list_v_mse += v_mse.mean().item()
            print(end_time - begin_time, img_name, eval_bpp.mean().item(), v_psnr.mean().item(),
                  (eval_bpp + lmbda * v_mse).cpu().item())

            cnt += 1

        print('[WITHOUT PRE-PROCESSING] ave_time:%.4f bpp: %.4f psnr: %.4f  v_mse: %.4f' % (
            sum_time / cnt,
            list_eval_bpp / cnt,
            list_v_psnr / cnt,
            list_v_mse / cnt
        )
              )

    else:  # Pre Processing is True
        num = 1
        list_eval_bpp = 0.
        list_v_psnr = 0.
        cnt = 0
        for img_name in images:
            import torchvision.transforms as transforms
            from PIL import Image
            begin_time = time.time()

            data = Image.open(img_name)
            data = tv.transforms.ToTensor()(data)

            print("data:{}", data.shape)

            image = data
            print("image :{}".format(image.shape))
            import torchvision
            image = torchvision.transforms.functional.to_pil_image(image)

            # 将灰度图像转换为RGB格式

            # image.save('/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/RBS/1.png'.format(num))
            image.save('/root/data/2021/buxiaobu/Neural_Syntax/Pretrain_model/image/1.png'.format(num))

            # image.save('/root/data/2021/buxiaobu/Neural_Syntax/shifit/image/{}_1.png'.format(num))

            _, h, w = data.shape
            h_padded = h
            w_padded = w
            if h % 64 != 0:
                h_padded = (h // 64) * 64 + 64
            if w % 64 != 0:
                w_padded = (w // 64) * 64 + 64
            padding_h = h_padded - h
            padding_w = w_padded - w

            h_pad_zeros = torch.ones(3, padding_h, w)
            w_pad_zeros = torch.ones(3, h_padded, padding_w)
            data = torch.cat((data, h_pad_zeros), 1)
            data = torch.cat((data, w_pad_zeros), 2)

            data = data.unsqueeze(0)
            data = data * 2 - 1

            data_bpp = 0.0
            data_mse = 0.0

            net = Net((1, h, w, 3), (1, h, w, 3), is_high, post_processing).cuda()
            net.load_state_dict(torch.load(weight_path), strict=True)
            opt_enc = optim.Adam(net.a_model.parameters(), lr=1e-5)
            sch = optim.lr_scheduler.MultiStepLR(opt_enc, [50], 0.5)

            net.post_processing = False  # Update encoder without post-processing to save GPU memory
            # If the GPU memory is sufficient, you can delete this sentence
            for iters in range(tune_iter):
                train_bpp, train_mse = net(data.cuda(), 'train')

                train_loss = lmbda * train_mse + train_bpp
                train_loss = train_loss.mean()

                opt_enc.zero_grad()
                train_loss.backward()
                opt_enc.step()

                sch.step()

            net.post_processing = post_processing
            with torch.no_grad():
                eval_bpp, _, v_psnr = net(data.cuda(), 'test', num)

            list_eval_bpp += eval_bpp.mean().item()
            list_v_psnr += v_psnr.mean().item()
            cnt += 1

            print([time.time() - begin_time], img_name, eval_bpp.mean().item(), v_psnr.mean().item())

        print('[WITH PRE-PROCESSING] bpp: %.4f psnr: %.4f' % (
            list_eval_bpp / cnt,
            list_v_psnr / cnt
        )
              )


if __name__ == "__main__":
    print(torch.cuda.is_available())
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(

    parser.add_argument(
        # "--train_data_path", default="/media/yang/Pytorch/buxiaobu/code/STF-main/openimages/train/data/*",
        # help="Directory of Testset Images")
        # "--data_path", default="/media/yang/Pytorch/buxiaobu/code/My_dataset/DIV2K_train_HR/train/*",
        # help="Directory of Testset Images")
        "--data_path", default="/media/yang/Pytorch/buxiaobu/code/My_dataset/kodak/*",
            help="Directory of Testset Images")
    # parser.add_argument(
    #     "--data_path", default="/root/data/2021/buxiaobu/Mydataset/kodak/*",
    #     help="Directory of Testset Images")
    parser.add_argument(
        "--weight_path", default="/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/DDPM/net_ga/checkpoint/2199.ckpt",
        help="Path of Checkpoint")
    # parser.add_argument(
    #     "--weight_path",
    #     default="/root/data/2021/buxiaobu/Neural_Syntax/Pretrain_model/0015.ckpt"
    #             ,
    #     help="Path of Checkpoint")
    parser.add_argument(
        "--high", action="store_true",
        help="Using High Bitrate Model")
    parser.add_argument(
        "--post_processing", action="store_true",
        help="Using Post Processing")
    parser.add_argument(
        "--pre_processing", action="store_true",
        help="Using Pre Processing (Online Finetuning)")
    parser.add_argument(
        "--lambda", type=float, default=0.0067, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    parser.add_argument(
        "--tune_iter", type=int, default=100,
        help="Finetune Iteration")

    args = parser.parse_args()
    # print(torch.cuda.get_device_name(0))
    # print(torch.cuda.current_device())
    # device = torch.device("cuda:0 " if torch.cuda.is_available() else "cpu")
    # a = torch.zeros(3).to(device)
    # print(a)


    # print("a",a.get_device())
    print(args.weight_path)
    val(args.data_path, args.weight_path, args.lmbda, args.high, args.post_processing, args.pre_processing,
        args.tune_iter)

