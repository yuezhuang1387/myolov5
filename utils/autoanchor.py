# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
AutoAnchor utils
"""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.general import LOGGER, colorstr, emojis

PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    '''
    检查当前描框设置和数据集是否契合，若不太契合尝试使用kmean重新生成描框(结果不一定更好)去更新yolo模型最后detect部分的anchors参数
    :param dataset: 训练Dataset
    :param model:
    :param thr: 4/3.44
    :param imgsz: 640
    '''
    # Check anchor fit to data, recompute if necessary
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    # 先对数据集进行缩放
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1,
                                                         keepdims=True)  # shapes.shape: (训练集图片总数, 2), 2对应W和H，此时W和H按照图像长宽比缩放，最大的为640
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()
    # wh.shape: torch.Size([训练集图片总数, 2]),2对应W和H,此时的W和H均乘了一个随机(0.9~1.1)的缩放系数（数值为实际长宽，不是归一化）

    def metric(k):  # compute metric
        '''
        best中的值越大(接近1)，说明当前设置的描框组合和图像的契合尺度越好，实际可理解为描框和图像尺寸接近（不管是大还是小）
        :param k: 描框,shape: torch.Size([9,2]),9表示三层（每层像素点对应3个描框）
        :return:bpr：best>0.25的占比
                    best中的值越大(接近1)，说明当前设置的描框组合和图像的契合尺度越好，实际可理解为描框和图像尺寸接近（不管是大还是小）
                    显然和图像契合程度占比越高越好
        '''
        # print(k)
        # wh[:, None].shape: torch.Size([训练集图片总数,1,2]),
        # k[None].shape: torch.Size([1,9,2])
        r = wh[:, None] / k[None] # r.shape: torch.Size([训练集图片总数,9,2])
        x = torch.min(r, 1 / r).min(2)[0]  # 对每一个描框，取[图像W/描框W, 图像H/描框H, 描框W/图像W, 描框H/图像H]的最小值，x的shape为torch.Size([训练集图片总数, 9])
        best = x.max(1)[0]  # best.shape: torch.Size([训练集图片总数])，best中的值越大(接近1)，说明当前设置的描框组合和图像的契合尺度越好，实际可理解为描框和图像尺寸接近（不管是大还是小）
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1 / thr).float().mean()  # best possible recall, best>0.25的占比
        return bpr, aat

    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # m.stride: tensor([ 8., 16., 32.]
    anchors = m.anchors.clone() * stride  # current anchors, shape: torch.Size([3(层),3(描框数),2(描框宽长)])
    # print(anchors)
    # tensor([[[ 10.,  13.],
    #          [ 16.,  30.],
    #          [ 33.,  23.]],
    #
    #         [[ 30.,  61.],
    #          [ 62.,  45.],
    #          [ 59., 119.]],
    #
    #         [[116.,  90.],
    #          [156., 198.],
    #          [373., 326.]]], device='cuda:0')
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    if bpr > 0.98:
        # 直接使用当前描框 threshold to recompute
        LOGGER.info(emojis(f'{s}Current anchors are a good fit to dataset ✅'))
    else:
        # 不太契合，需要尝试一下kmean重新生成描框
        LOGGER.info(emojis(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...'))
        na = m.anchors.numel() // 2  # number of anchors
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            LOGGER.info(f'{PREFIX}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= stride
            s = f'{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)'
        else:
            s = f'{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)'
        LOGGER.info(emojis(s))


def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset
        通过将数据集每张图的长宽作为kmean聚类输入，最终将其聚为9个类，每一类聚类中心就对应一个描框
        Arguments:
            dataset: path to data.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    npr = np.random
    thr = 1 / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for x in k:
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.dataloaders import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh
    # wh0.shape: torch.Size([训练集图片总数, 2]),2对应W和H,此时的W和H均乘了一个随机(0.9~1.1)的缩放系数（数值为实际长宽，不是归一化）

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f'{PREFIX}WARNING: Extremely small objects found: {i} of {len(wh0)} labels are < 3 pixels in size')
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans init
    try:
        LOGGER.info(f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...')
        assert n <= len(wh)  # apply overdetermined constraint
        s = wh.std(0)  # sigmas for whitening
        k = kmeans(wh / s, n, iter=30)[0] * s  # points
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        LOGGER.warning(f'{PREFIX}WARNING: switching strategies from kmeans to random init')
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)

    return print_results(k)
