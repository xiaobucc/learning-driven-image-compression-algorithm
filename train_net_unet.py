import os
import argparse
import glob
import numpy as np
import torch
import torchvision as tv
from torch import nn, optim
import torch.nn.functional as F
from model.vit_model import vit_base_patch16_224_in21k as create_model
import pickle
from PIL import Image
from torch.autograd import Function
import time
from pytorch_msssim import ms_ssim
# from model.net import Net
from model.Net_unet import Net
from model.util import *


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)


class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, train_glob, transform):
        super(DIV2KDataset, self).__init__()
        self.transform = transform
        self.images = list(sorted(glob.glob(train_glob)))
        # print(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        # h , w = img.size
        img = self.transform(img)
        # print("img 的大小wei ",img.shape)
        return img

    def __len__(self):
        return len(self.images)


class Preprocess(object):
    def __init__(self):
        pass

    def __call__(self, PIL_img):
        img = np.asarray(PIL_img, dtype=np.float32)
        img /= 127.5
        img -= 1.0
        return img.transpose((2, 0, 1))


def quantize_image(img):
    img = torch.clamp(img, -1, 1)
    img += 1
    img = torch.round(img)
    img = img.to(torch.uint8)
    return img


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        return out


def train(train_data_path, lmbda, lr, batch_size, checkpoint_dir, weight_path, is_high, post_processing):
    print("zhi xing le ")
    train_data = DIV2KDataset(train_data_path, transform=tv.transforms.Compose([
        tv.transforms.RandomCrop(256),
        Preprocess()
    ]))

    training_loader = torch.utils.data.DataLoader(train_data,
                                                  batch_size=batch_size,

                                                  shuffle=True,
                                                  num_workers=8)
    print("数据集大小：", len(training_loader))
    net = Net((batch_size, 256, 256, 3), (1, 256, 256, 3), is_high, post_processing).cuda()
    # print("net---------------------------------:\n", net)

    # def weight_init(m):
    #
    #     print(" m的值为:{}  \n".format(m))
    #
    #     print("-----------------------------")
    #     # if isinstance(m, nn.Linear):
    #     #     nn.init.xavier_uniform_(m.weight)
    #     #     nn.init.constant_(m.bias, 0)
    #     # elif isinstance(m, nn.Conv2d):
    #     #     nn.init.xavier_uniform_(m.weight)
    #     #     nn.init.constant_(m.bias, 0)

    if weight_path != "":
        print("--------------------已加载预训练模型---------------------")
        net.load_state_dict(torch.load(weight_path), strict=True)
    # else:
    #     print("--------------------未加载预训练模型---------------------")
    #     net.apply(weight_init)

    if post_processing:  # Only Train Post Processing Module
        # opt = optim.Adam(net.post_processing_params(), lr=lr)

        opt = optim.AdamW(net.post_processing_params(), lr=lr)
        sch = optim.lr_scheduler.MultiStepLR(opt, [1200, 1350], 0.5)
        train_epoch = 1500
    else:
        opt = optim.Adam(net.base_params(), lr=lr)
        sch = optim.lr_scheduler.MultiStepLR(opt, [1500, 2500, 3500, 4000], 0.5)
        train_epoch = 5000
        # entropy_optimizer = torch.optim.AdamW([
        #     {'params': net.cit_he.parameters(), 'lr': opt.lr},
        #     {'params': net.cit_hd.parameters(), 'lr': opt.lr},
        #     {'params': net.cit_ar.parameters(), 'lr': opt.lr}
        # ], eps=1e-8, weight_decay=opt.wd)
        # lr_step = list(np.linspace(opt.epoch_pretrained, opt.nEpochs, 6, dtype=int))[1:]
        #
        # lr_scheduler = LearningRateScheduler(mode='stagedecay',
        #                                      lr=opt.lr,
        #                                      num_training_instances=len(training_loader),
        #                                      stop_epoch=opt.nEpochs,
        #                                      warmup_epoch=opt.nEpochs * opt.warmup,
        #                                      stage_list=lr_step,
        #                                      stage_decay=opt.lr_decay)
        #
        # lr_scheduler.update_lr(opt.epoch_pretrained * len(training_loader))

    net = nn.DataParallel(net)

    # 设置超先验网络的学习率

    start_time = time.time()
    print("开始训练的时间为：{}".format(start_time))

    for epoch in range(0, train_epoch):
        net.train()
        list_train_loss = 0.
        list_train_bpp = 0.
        list_train_mse = 0.

        cnt = 0

        for i, data in enumerate(training_loader, 0):
            # Updata lr
            # lr_scheduler.update_lr(batch_size=batch_size)
            # current_lr = lr_scheduler.get_lr()
            # for param_group in entropy_optimizer.param_groups:
            #     param_group['lr'] = current_lr

            x = data.cuda()
            opt.zero_grad()
            # print("x.shape ---------------{}".format(x.shape))
            train_bpp, train_mse = net(x, 'train')
            # print("train_bpp:{} train_mse:{}".format(train_bpp,train_mse))
            # out_criterion = criterion(out_net, d)
            train_loss = lmbda * 255 ** 2 * train_mse + train_bpp
            # mse = 10 * np.log10(255 * 255 / train_mse.item())
            # mse = torch.tensor(mse)
            # mse.cuda()
            # train_loss = mse + train_bpp * 100
            train_loss = train_loss.mean()
            train_bpp = train_bpp.mean()
            train_mse = train_mse.mean()
            # print("train_loss: ",train_loss.item())
            if np.isnan(train_loss.item()):
                raise Exception('NaN in loss')

            list_train_loss += train_loss.item()
            list_train_bpp += train_bpp.item()
            list_train_mse += train_mse.item()

            train_loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), 10)
            nn.utils.clip_grad_norm_(net.parameters(), 1)

            opt.step()
            cnt += 1
            # if i % 200 == 0:
            #     print('[Epoch %04d batch:%04d,TRAIN] Loss: %.4f bpp: %.4f mse: %.4f  ' % (
            #         epoch,
            #         i,
            #         train_loss,
            #         train_bpp,
            #         list_train_mse
            #     )
            #           )

        print('[Epoch %04d TRAIN] Loss: %.4f bpp: %.4f mse: %.4f  ' % (
            epoch,
            list_train_loss / cnt,
            list_train_bpp / cnt,
            list_train_mse / cnt
        )
              )

        sch.step()
        # entropy_optimizer.step()
        if (epoch % 100 == 99):

            print('[INFO] Saving')
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            torch.save(net.module.state_dict(), '%s/%04d.ckpt' % (checkpoint_dir, epoch))

        #
        # print('[INFO] Saving')
        # if not os.path.isdir(checkpoint_dir):
        #     os.mkdir(checkpoint_dir)
        # torch.save(net.module.state_dict(), '%s/%04d.ckpt' % (checkpoint_dir, epoch))

        with open(os.path.join(checkpoint_dir, 'train_log.txt'), 'a') as fd:
            fd.write('[Epoch %04d TRAIN] Loss: %.4f bpp: %.4f mse: %.4f \n' % (
                epoch, list_train_loss / cnt, list_train_bpp / cnt, list_train_mse / cnt))
        fd.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        # "--train_data_path", default="/media/yang/Pytorch/buxiaobu/code/STF-main/openimages/train/data/*",
        # help="Directory of Testset Images")
        "--train_data_path", default="/media/yang/Pytorch/buxiaobu/code/My_dataset/DIV2K_train_HR/train/*",
        help="Directory of Testset Images")
    parser.add_argument(
        "--weight_path", default="",
        help="Path of Pretrained Checkpoint")
    # parser.add_argument(
    #     "--weight_path", default="/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/Net_WAM_CTM/Net_WAM_CTM_Spilt/checkpoint/checkpoint_orign/2199.ckpt",
    #     help="Path of Pretrained Checkpoint")
    # parser.add_argument(
    #     "--checkpoint_dir", default="/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/Add_WAM/Add_WAM_full",
    #     help="Directory of Saved Checkpoints")
    # "--checkpoint_dir", default="/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/image_scaling",
    # help="Directory of Saved Checkpoints")

    parser.add_argument(
        "--checkpoint_dir",
        default="/media/yang/Pytorch/buxiaobu/code/Neural_Syntax/Net_WAM_CTM/Net_WAM_CTM_Spilt/net_unet/net_uet",
        help="Directory of Saved Checkpoints")
    parser.add_argument(
        "--high", action="store_true",
        help="Using High Bitrate Model")
    parser.add_argument(
        "--post_processing", action="store_true",
        help="Using Post Processing")
    parser.add_argument(
        "--lambda", type=float, default=0.0025, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    # parser.add_argument(
    #     "--lambda", type=float, default=0.05, dest="lmbda",
    #     help="Lambda for rate-distortion tradeoff.")
    # parser.add_argument(
    #     "--lr", type=float, default=1e-4,
    #     help="Learning Rate")
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning Rate")
    # parser.add_argument(
    #     "--lr", type=float, default=0.0002,
    #     help="Learning Rate")
    parser.add_argument(
        "--batch_size", type=float, default=8,
        help="Batch Size")

    args = parser.parse_args()

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.current_device())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.zeros(3).to(device)
    print(a)
    print("a",a.get_device())

    train(args.train_data_path, args.lmbda, args.lr, args.batch_size, args.checkpoint_dir, args.weight_path, args.high,
          args.post_processing)