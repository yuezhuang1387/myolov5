# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Image augmentation functions
"""

import math
import random

import cv2
import numpy as np

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box
from utils.metrics import bbox_ioa


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            # 可认为是空域增强-基于灰度变换
            T = [
                A.Blur(p=0.01), # 有0.01的概率使用随机size内核(范围(3,7))对图像进行均值模糊
                A.MedianBlur(p=0.01), # 有0.01的概率使用随机size内核(范围(3,7))对图像进行中值模糊
                A.ToGray(p=0.01), # 有0.01的概率将输入RGB图像转为灰度图，灰度图shape: (H,W,3)，次函数实际内部调用cv2.cvtColor函数，先cv2.COLOR_RGB2GRAY，然后cv2.COLOR_GRAY2RGB，一个通道直接复制3份
                A.CLAHE(p=0.01), # 有0.01的概率进行分区自适应直方图均衡化，默认将图像分为8*8一共64个区域后再直方图均衡化
                A.RandomBrightnessContrast(p=0.0), # 有p概率随机改变图像的亮度和对比度(+-0.2)
                A.RandomGamma(p=0.0), # 有p的概率对图像进行随机gamma变换(gamma系数默认从0.8~1.2之间随机)
                A.ImageCompression(quality_lower=75, p=0.0)]  # 有p概率减小jpeg图像的压缩（啥意思，提高图像质量？）
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            LOGGER.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # albumentations package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(colorstr('albumentations: ') + f'{e}')

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    '''
    HSV(色调、饱和度、明度)颜色空间增强，介绍：https://zhuanlan.zhihu.com/p/67930839
    :param im:(H,W,3)
    :param hgain: 调整系数取值在[1-hgain,1+hgain]
    :param sgain: 调整系数取值在[1-sgain,1+sgain]
    :param vgain: 调整系数取值在[1-vgain,1+vgain]
    :return:(H,W,3)
    '''
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        # OpenCV中HSV三个分量的范围为：
        # H = [0,179]
        # S = [0,255]
        # V = [0,255]
        # hue.shape: (H,W)
        dtype = im.dtype  # uint8
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype) # 取值0~179
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        # cv2.LUT函数为查表函数：https://blog.csdn.net/weixin_41010198/article/details/111634487
        # 具体解析：cv2.LUT(hue, lut_hue)返回值和hue同尺寸的，令返回为r1，则r1[i,j] = lut_hue[hue[i,j]]
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed，输出即为im


def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    # Replicate labels
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    '''
    将输入im保持宽高比缩放到new_shape范围内，空白处填充color
    :param im: 按长宽比例缩放后的应图像，确保最长边达到640(另一个短边是按原图比例缩放得到，且不一定能被32整除), im.shape:(h_resized,w_resized,3)
    :param new_shape: shape为当前index对应batch中图像需要统一缩放到的尺寸 (H,W), shape.shape: (2,) (H和W中一个为640，另一个短边能被32整除)
    :param color: 按默认(114, 114, 114)
    :param auto: 设置为False
    :param scaleFill: 按默认False
    :param scaleup: 设置为self.augment
    :param stride: 按默认32
    :return:
            im: (new_shape_H,new_shape_W,3)将输入im保持图像宽高比缩放到当前index对应batch的统一尺寸new_shape(H,W)内，空白处填color(该尺寸H和W中一个为640，另一个短边能被32整除)
            ratio：(w_r,h_r)输入图像im最终缩小的系数
            (dw, dh): 输入im缩小到new_shape范围内后，dw或dh其中之一为0，另一个为需要填充的宽度/2
    '''
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # 按原图比例缩放后
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding，有一个是0
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding，np.mod表示按元素求余
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR) # 将原图保持宽高比缩放
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1)) # (+-0.1)是为了应对dw或者dh中为奇数的情况，使得扩充的总长度仍然为dw或者dh
    # copyMakeBorder函数解析 https://blog.csdn.net/qq_35037684/article/details/107792712?
    # 给im上下左右进行扩充，top, bottom, left, right为上下左右要扩展的像素宽度，cv2.BORDER_CONSTANT表示扩展区域填充常量value
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # im: (new_shape_H,new_shape_W,3)将输入im保持图像宽高比缩放到当前index对应batch的统一尺寸new_shape(H,W)内，空白处填color(该尺寸H和W中一个为640，另一个短边能被32整除)
    # ratio：(w_r,h_r)输入图像im最终缩小的系数
    # (dw, dh): 输入im缩小到new_shape范围内后，dw或dh其中之一为0，另一个为需要填充的宽度/2
    return im, ratio, (dw, dh)


def random_perspective(im,
                       targets=(),
                       segments=(),
                       degrees=10,
                       translate=.1,
                       scale=.1,
                       shear=10,
                       perspective=0.0,
                       border=(0, 0)):
    '''
    解析：http://t.zoukankan.com/shuimuqingyang-p-14595210.html
    随机仿射变换数据增强,包含缩放裁剪旋转等等，实际上不会使用仿射和旋转，因为这样做后图像的边框就成了倾斜的
    :param im: 复制粘贴了一些物体后的新mosaic画布，shape: (2*640,2*640,3)
    :param targets: shape: (画布上扩充后的全部nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
    :param segments: segments为list,len(segments)=画布上扩充后的全部标记物体数量，segments[0].shape: (num_pixels, 2)，对应图像移到画布上后segments在画布坐标系上的实际位置（没有归一化）
    :param degrees: self.hyp['degrees']，图像的旋转角度，参数设置只有0
    :param translate: self.hyp['translate']，0.0459/0.1/0.0902，图像translate的比例？
    :param scale: shear=self.hyp['scale'],图像缩放系数，(+-)
    :param shear: self.hyp['shear'], 图像剪切，参数设置只有0
    :param perspective: perspective=self.hyp['perspective'], 仿射系数，参数设置只有0
    :param border: [-640//2,-640//2], 通过对border参数设置为(0,0)即可实现法仿射变换输入输出尺寸相同,不为(0,0)则输入输出尺寸不同
    :return:
            im: 仿射变换后数据增强了的画布，训练阶段输出shape：(640,640,3)
            targets: 仿射变换后画布上的相应labels，框个数可能会减少，shape: (全部nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
    '''
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2 # 640
    width = im.shape[1] + border[1] * 2 # 640

    # Center
    C = np.eye(3) # np.eye生成对角矩阵
    # [[1. 0. 0.]
    #  [0. 1. 0.]
    #  [0. 0. 1.]]
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)
    # [[1. 0. -640.]
    #  [0. 1. -640.]
    #  [0. 0. 1.]]

    # Perspective
    P = np.eye(3) # np.eye生成对角矩阵
    # [[1. 0. 0.]
    #  [0. 1. 0.]
    #  [0. 0. 1.]]
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
    # [[1. 0. 0.]
    #  [0. 1. 0.]
    #  [p(0). p(0). 1.]]

    # Rotation and Scale
    R = np.eye(3)
    # [[1. 0. 0.]
    #  [0. 1. 0.]
    #  [0. 0. 1.]]
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114)) # 仿射变换函数，输出(640,640)

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    # im: 仿射变换后数据增强了的画布，训练阶段输出shape：(640,640,3)
    # targets: 仿射变换后画布上的相应labels，框个数可能会减少，shape: (全部nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    '''
    对当前输入的画布im，n为画布上物体总数，取0.1*n个物体，判断这些物体边界框在水平翻转后(以图像中轴为基准)的位置是否大致位于背景，是的话则将图像此处也添加一个现有物体，并对当前画布的labels和segments进行更新
    此法为针对小目标提出的一种数据增强方法，复制粘贴数据增强，解析：https://blog.csdn.net/qq_42722197/article/details/111351368?
    Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    :param im: mosaic画布，shape: (2*640,2*640,3)
    :param labels: shape: (画布上全部nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
    :param segments: segments为list,len(segments)=画布上全部标记物体数量，segments[0].shape: (num_pixels, 2)，对应图像移到画布上后segments在画布坐标系上的实际位置（没有归一化）
    :param p: 取值有0/0.1
    :return: im: 复制粘贴了一些物体后的新mosaic画布，shape: (2*640,2*640,3)
             labels: shape: (画布上扩充后的全部nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
             segments: segments为list,len(segments)=画布上扩充后的全部标记物体数量，segments[0].shape: (num_pixels, 2)，对应图像移到画布上后segments在画布坐标系上的实际位置（没有归一化）
    '''
    n = len(segments) # 当前画布上物体总个数
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        # random.sample() 从序列中随机截取指定个数的元素，下面意思为从n从随机挑选p*n个元素
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4] # 这一步实际上以画布中轴线为参考，对边框坐标进行水平翻转
            ioa = bbox_ioa(box1=box, box2=labels[:, 1:5],eps=1E-7)  # shape: (n,) labels中每个边框和box框的交集在labels每个框面积中的占比
            if (ioa < 0.30).all():
                # .all()表示做逻辑与运算，若对象全为True，则.all()返回True；否则返回False
                # 此处在判断若当前这个边框box和labels中全部框的交集在labels每个框面积中的占比均不超过0.3时，可认为此处为背景，没有物体，可以在此处复制粘贴额外添加一个物体
                labels = np.concatenate((labels, [[l[0], *box]]), 0) # 额外添加的边框
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1)) # 额外添加的segments区域
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED) # 在新图in_new(作为mask)上原始segments[j]区域处填充元素(255,255,255)，标记-这些区域可复制粘贴到别处

        # cv2.bitwise_and按位与运算，im_new作为mask，im中im_new上255元素对应位置值保留；im中im_new上0元素对应位置值置0，结果存在result
        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)，妙啊，直接对保留了原始各个segments[j]区域的result图像进行翻转,产生需额外添加到图像上的物体（呼应上面对坐标框进行以画布中轴线为基准进行水平翻转的操作）
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        im[i] = result[i]  # 将原始im画布中的大致背景区域添加上额外的物体，cv2.imwrite('debug.jpg', im)  # debug
    return im, labels, segments


def cutout(im, labels, p=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    '''
    对图像进行叠加，有意思，图像叠加，labels继续concat
    对im和im2进行mixup数据增强，im1和im2均是mosaic数据增强后得到的
    :param im: mosaic数据增强+copy paste数据增强+仿射变换(随机缩放/裁剪）后的画布，训练阶段输出labels继续concat
    :param labels: mosaic数据增强+copy paste数据增强+仿射变换(随机缩放/裁剪）后画布上的相应labels，框个数可能会减少，shape: (全部nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
    :param im2: mosaic数据增强+copy paste数据增强+仿射变换(随机缩放/裁剪）后的画布，训练阶段输出shape：(640,640,3)
    :param labels2: mosaic数据增强+copy paste数据增强+仿射变换(随机缩放/裁剪）后画布上的相应labels，框个数可能会减少，shape: (全部nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
    :return:
           im: mosaic数据增强+copy paste数据增强+仿射变换+mixup融合后的图像，shape：(640,640,3)
           labels：融合后concat两图像的labels，shape：(融合后图像nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
    '''
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0，beat分布，抛一枚硬币，正面32次，反面32次，硬币正面概率，返回r值应该接近0.5
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
