import torch
import torch.nn as nn
import math
from pathlib import Path
import glob
import os
from PIL import ImageOps, Image
import numpy as np
import cv2
import hashlib
import random
import pkg_resources as pkg
import platform
import subprocess
from utils.general import methods
from utils.loggers import Loggers
from train import parse_opt
from train import LOGGER
from models.yolo import Model
from utils.autobatch import autobatch, check_train_batch_size
import yaml
import wandb
import torch.nn.functional as F
from tqdm import tqdm


def demo():
    # 模型
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
    # 图像
    img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
    # 推论
    results = model(img)
    # 结果
    results.print()


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def lookRect(img_file=r'G:\\Turingdataset\\coco128\\images\\train2017\\000000000089.jpg'):
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    lb_file = sb.join(img_file.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt'
    with open(lb_file) as f:
        lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
        # lb: [['45', '0.479492', '0.688771', '0.955609', '0.5955'],
        #      ['45', '0.736516', '0.247188', '0.498875', '0.476417'],
        #      ['50', '0.637063', '0.732938', '0.494125', '0.510583']]
    a = cv2.imread(img_file)
    H, W, C = a.shape
    print(a.shape)
    lb = np.array(lb, dtype=np.float32)
    for c, xc, yc, w, h in lb:
        # xc,yc,w,h = lb[1][1],lb[1][2],lb[1][3],lb[1][4]
        x1, y1, x2, y2 = (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H
        print(f'{x1} {y1} {x2} {y2}')
        cv2.rectangle(a, (int(x1), int(y1), int(w * W), int(h * H)), thickness=2, color=(255, 255, 0))
        # rect函数的输入为（左上角x，左上角y，宽，高）
        # x = 100
        # wh = 100
        # cv2.rectangle(a, (x+100, x, wh+100, wh), thickness=2, color=(255, 255, 0))
        cv2.imshow('wdcnm', a)
        cv2.waitKey()


def abm():
    import albumentations as A

    # Declare an augmentation pipeline
    transform = A.Compose([
        # A.Blur(p=0.01),
        # A.RandomCrop(width=256, height=256),
        # A.HorizontalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
        A.ToGray(p=1),
        A.CLAHE(tile_grid_size=(64, 64), p=1),
    ])

    # Read an image with OpenCV and convert it to the RGB colorspace
    image = cv2.imread(
        r'G:\Turingdataset\bv\images\aaahandbags\false\Bottega Veneta\Bottega Veneta Arco 48 Bag Grainy Calfskin (Varied Colors)_262\5939175.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Augment an image
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    print(np.max(transformed_image))
    print(transformed_image.shape)
    print((transformed_image[:, :, 0] == transformed_image[:, :, 1]).sum())
    cv2.imshow('wd1', transformed_image)
    cv2.imshow('hs', cv2.equalizeHist(cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)))
    cv2.waitKey()


def autoanchor():
    from utils.general import check_yaml
    from utils.autoanchor import check_anchors
    from utils.dataloaders import LoadImagesAndLabels
    model = Model('models/yolov5n.yaml').to(torch.device('cuda'))
    hyp = check_yaml('E:\裂缝\yolo\myolov5\data\hyps\hyp.scratch-high.yaml', suffix=('.yaml', '.yml'))
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    dataset = LoadImagesAndLabels(path=r'E:\裂缝\yolo\datasets\coco128\images\train2017',
                                  img_size=640,
                                  batch_size=2,
                                  augment=True,
                                  hyp=hyp,
                                  rect=True,
                                  # 当augment=True时，设置rect为True即可不使用mosaic数据增强，self.rect为True表明每批batch数据需要自适应缩放尺寸
                                  image_weights=False,
                                  cache_images=None,
                                  single_cls=False,
                                  stride=32,
                                  pad=0,
                                  prefix='train')
    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=640)


def testwandb():
    import wandb

    wandb.init(project="yue")
    wandb.config = {
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 128
    }
    epochs = 300
    for e in range(epochs):
        wandb.log({"loss": (torch.randint(10, (1,)) + e) / epochs})


class TestCNN(nn.Module):
    nl = 3

    def __init__(self, i, o):
        super().__init__()
        a = i
        b = o

    def forward(self, x):
        return x


def yue(a):
    return (1, 2, *(a.cpu() / 3).tolist()), 3, 4


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # model = Model(cfg='E:\裂缝\yolo\myolov5\models\yolov5n.yaml',ch=3,nc=80).to(torch.device('cuda'))
    # print(check_train_batch_size(model,640,True))
    # print(math.cos(math.pi))
    # autoanchor()
    # testwandb()
    # print(torch.randint(10,(1,)))
    a = './fah/sfa'
    print(a.startswith('./'))
