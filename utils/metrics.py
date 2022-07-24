# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def fitness(x):
    '''
    Model fitness as a weighted combination of metrics
    :param x: shape: (1,7)，7对应(mp, mr, map50, map, box-loss, conf-loss, cls-loss):
    :return: 加权后的指标值，shape: (1,)
    '''
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def smooth(y, f=0.05):
    '''
    对y做一个卷积操作，相当于均值平滑了
    :param y: shape: (1000, )，置信度阈值从0~1(取1000个值)时指标(recall/precision/F1)的对应取值
    :param f: 默认0.05
    :return: 平滑后的数组，shape: (1000, )
    '''
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    '''
    计算验证集各类指标(average precision, AP)，保存iou@0.5的P/R/P-R/F1曲线
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    :param tp: shape: (每张图nms最终筛完预测框数量(不超过300)×验证集图像总数, 10)，每一行看为是某个预测框在不同iouv[i]阈值下在该图中能否有匹配的标签框(True/False)，（每一列中最终能和标签框匹配上的预测框数<=当前验证集全部图像标签框个数）
    :param conf: shape: (每张图nms最终筛完预测框数量(不超过300)×验证集图像总数,)，该预测框概率
    :param pred_cls: shape: (每张图nms最终筛完预测框数量(不超过300)×验证集图像总数,)，该预测框类别
    :param target_cls: (每个图像标签框数×验证集图像总数,)，标签框类别
    :param plot: 默认False
    :param save_dir: 'runs\train\exp12'
    :param names: 类别字典，{0:'person', 1:'bicycle', ..., 79:'toothbrush'}
    :param eps: 默认值1e-16
    :return: The average precision as computed in py-faster-rcnn. (tp, fp, p, r, f1, ap, unique_classes)
             tp: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的TP框个数(每类别实际标签框个数×recall)，IoU@0.5
             tp: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的FP框个数(TP/precision-TP，相当于每个类别预测框个数-TP)，IoU@0.5
             p: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的precision，IoU@0.5
             r: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的recall，IoU@0.5
             f1: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的F1，IoU@0.5
             ap: shape: (nc, 10), nc为验证集标签框类别数(去重后)，10对应各类别在10个iou阈值下的AP
             unique_classes: shape: (nc,)，nc为验证集标签框类别数(去重后)，对应各个类别值
    '''
    # 1、根据置信度排序
    i = np.argsort(-conf) # 返回按照预测框置信度从大到小排序后的索引
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 2、Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    # unique_classes为标签框类别去重后按照从小到大的排的新list，shape:(nc,)
    # nt表示各个类别标签框在验证集中出现的总个数，shape: (nc,)
    nc = unique_classes.shape[0]  # 验证集标签框类别去重后类别的个数

    # 3、每类构造PR曲线且计算AP
    px, py = np.linspace(0, 1, 1000), []  # for plotting，(1000,), []
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000)) # (nc,10),(nc,1000),(nc,1000)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c # 验证集全部预测框中和当前标签框类别c相等的索引
        n_l = nt[ci]  # number of labels，验证集全部标签框中类别为c的标签框个数
        n_p = i.sum()  # number of predictions，验证集全部预测框中类别为c的预测框个数
        if n_p == 0 or n_l == 0:
            continue

        # 3.1 计算验证集中c类别的 FPs and TPs，TP和FP和FN的划分参考：https://zhuanlan.zhihu.com/p/443499860
        # cumsum(0)为数组按行累加 https://blog.csdn.net/cyj5201314/article/details/104595351
        fpc = (1 - tp[i]).cumsum(0) # shape: (类别为c的预测框个数, 10)，每一行表示类别为c的预测框在该行对应的置信度阈值下在不同的iou阈值下没匹配到(类别为c)标签框的个数，FP
        tpc = tp[i].cumsum(0) # shape: (类别为c的预测框个数, 10)，每一行表示类别为c的预测框在该行对应的置信度阈值conf下在不同iou阈值下能匹配到(类别为c)标签框的个数，TP（小于验证集中c类标签框总数）

        # 3.2 Recall
        recall = tpc / (n_l + eps)  # recall curve，(类别为c的预测框个数, 10)，每一行表示类别为c的预测框在该行对应的置信度阈值conf下在不同的iou阈值下在整个验证集上的recall
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
        # recall[:, 0]代表整个数据集中c类框，在不同的置信度阈值下的recall（交并比阈值为第一列对应0.5），置信度阈值越大recall越小

        # 3.3 Precision
        precision = tpc / (tpc + fpc)  # precision curve，(类别为c的预测框个数, 10)，每一行表示类别为c的预测框在该行对应的置信度阈值conf下在不同的iou阈值下在整个验证集上的precision
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # 3.4 AP from recall-precision curve
        for j in range(tp.shape[1]):
            # # recall[:, j].shape: (类别为c的预测框个数,)，在当前第j个iou阈值下，按照预测框置信度阈值从大到小排序，在每个置信度阈值下c类框的recall（置信度阈值越小recall越大）
            # precision[:, j].shape: (类别为c的预测框个数,)，在当前第j个iou阈值下，按照预测框置信度阈值从大到小排序，在每个置信度阈值下c类框的precision（置信度阈值越小precision越小，因为此时FP一般太多）
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            # ap[ci, j]表示当前类别在第j个iou阈值下的AP
            # mpre: shape: (类别为c的预测框个数+2,)，在当前第j个iou阈值下，按照预测框置信度阈值从大到小排序，在每个置信度阈值下c类框的precision（置信度阈值越小precision越小，因为此时FP一般太多）
            # mrec: shape: (类别为c的预测框个数+2,)，在当前第j个iou阈值下，按照预测框置信度阈值从大到小排序，在每个置信度阈值下c类框的recall（置信度阈值越小recall越大）
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5
                # py的最终长度为nc（验证集标签框类别去重后类别的个数），py[i].shape: (1000,) py[i]对应某类下np.linspace(0, 1, 1000)作为recall线性插值出的precision,iou@0.5

    # Compute F1 (harmonic mean of precision and recall)
    # p.shape: (nc,1000), p[i]对应某类别下置信度阈值从0~1(取1000个值)时precision的对应取值（precision从小到大）IoU@0.5
    # r.shape: (nc,1000), r[i]对应某类别下置信度阈值从0~1(取1000个值)时recall的对应取值（recall从大到小）IoU@0.5
    # ap.shape: (nc, 10), 10对应各类别在10个iou阈值下的AP
    f1 = 2 * p * r / (p + r + eps) # f1.shape: (nc,1000), f1[i]对应某类别下置信度阈值从0~1(取1000个值)时F1的对应取值 IoU@0.5
    names = [v for k, v in names.items() if k in unique_classes]  # list: 当前验证集所包含的框类别名称
    names = dict(enumerate(names))  # to dict，当前验证集中存在的类别，{0:'person', 1:'bicycle', ..., nc-1:'toothbrush'}
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names) # 绘制并保存各类别P-R曲线图iou@0.5
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1') # 绘制并保存各类别P/R/F1-confidence曲线图(iou@0.5)
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # 各类别平均F1取最大值时的置信度阈值索引
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    # tp: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的TP框个数(每类别实际标签框个数×recall)，IoU@0.5
    # tp: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的FP框个数(TP/precision-TP，相当于每个类别预测框个数-TP)，IoU@0.5
    # p: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的precision，IoU@0.5
    # r: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的recall，IoU@0.5
    # f1: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的F1，IoU@0.5
    # ap: shape: (nc, 10), nc为验证集标签框类别数(去重后)，10对应各类别在10个iou阈值下的AP
    # unique_classes: shape: (nc,)，nc为验证集标签框类别数(去重后)，对应各个类别值
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    '''
    计算以recall和precision为横、纵坐标的曲线的面积，AP
    :param recall: shape: (类别为c的预测框个数,)，在某个iou阈值下，按照预测框置信度阈值从大到小排序，在每个置信度阈值下c类框的recall（置信度阈值越小recall越大）
    :param precision: shape: (类别为c的预测框个数,)，在某个iou阈值下，按照预测框置信度阈值从大到小排序，在每个置信度阈值下c类框的precision（置信度阈值越小precision越小，因为此时FP一般太多）
    :return: (ap, mpre, mrec)
              ap: float类型AP值
              mpre: shape: (类别为c的预测框个数+2,)，在某个iou阈值下，按照预测框置信度阈值从大到小排序，在每个置信度阈值下c类框的precision（置信度阈值越小precision越小，因为此时FP一般太多）
              mrec: shape: (类别为c的预测框个数+2,)，在某个iou阈值下，按照预测框置信度阈值从大到小排序，在每个置信度阈值下c类框的recall（置信度阈值越小recall越大）
    '''

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision 包络, np.flip()对一维数组进行翻转，相当于mpre[::-1]
    # np.maximum.accumulate计算累积最大值 np.maximum.accumulate(np.array([2, 0, 3, -4, -2, 7, 9])) = [2 2 3 3 3 7 9]
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(y=np.interp(x, mrec, mpre), x=x)  # ap为float类型数值，计算给定点围成的梯形面积：https://blog.csdn.net/qq_38253797/article/details/119706121
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        '''
        :param nc: 80
        :param conf: 默认0.25
        :param iou_thres: 0.45
        '''
        # 混淆矩阵
        self.matrix = np.zeros((nc + 1, nc + 1)) # self.matrix中分为三类情况，参见process_batch函数
        self.nc = nc  # number of classes
        self.conf = conf # 0.25
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        '''
        对nms后的predict框和labels框进行匹配
        :param detections: shape: torch.Size([当前图像nms最终筛完的预测框数量(不超过300),6])
                           6中0:4表示映射到原图尺寸的实际预测框坐标(x1,y1,x2,y2)
                           6中4表示当前的预测概率值
                           6中5表示当前的预测类别(0~79)
        :param labels: torch.Size([当前图像标签框数,5]),5对应框类别+框映射到原图尺寸的实际坐标(x1,y1,x2,y2)
        :return:
        '''
        """
        Return intersection-over-union (Jaccard index) of boxes.
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf] # 筛出置信度大于0.25的
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        # x为元组，保存交并比满足阈值的索引,len(x)=2, x[0]对应标签框索引、x[1]对应预测框索引，shape均为torch.Size([交并比满足阈值且类别匹配的框数])
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            # matches.shape: (交并比满足阈值的框对数, 3) 3对应当前所匹配上的两框(标签框索引, 预测框索引, 两框交并比)
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]] # 按照匹配上的两框交并比从大到小排序
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]] # 去除预测框索引重复的部分(只保留第一个)（因为一个预测框可能和多个标签框对应）
                matches = matches[matches[:, 2].argsort()[::-1]] # 按照匹配上的两框交并比从大到小排序
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]] # 去除标签框索引重复的部分(只保留第一个)（一个标签框也可能和多个预测框对应）
                # 最终剩余的matches.shape[0]（即最终匹配框个数） <= 当前图像标签框的个数
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int) # (标签框索引, 预测框索引, 两框交并比)，m0.shape=m1.shape=(最终匹配框个数,)
        for i, gc in enumerate(gt_classes):
            j = m0 == i # j.shape=(最终匹配框个数,)，其中最多只有一个True，即m0中等于i处的位置为True
            if n and sum(j) == 1:
                # 1、成功匹配，TP
                self.matrix[detection_classes[m1[j]], gc] += 1
                # self.matrix[k,l]表示k类预测框和l类标签框成功匹配，数值+1
            else:
                # 2、当前gt框没有匹配上的预测框（因为IoU不满足阈值），在当前gt框对应的类别上+1，最下面一行
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    # 3、当前预测框没匹配上标签框时，在预测框对应的类别上+1，最右边一列
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='', names=()):
        # 在最后一轮绘制一个混淆矩阵
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            nc, nn = self.nc, len(names)  # number of classes, names
            sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
            labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(array,
                           annot=nc < 30,
                           annot_kws={
                               "size": 8},
                           cmap='Blues',
                           fmt='.2f',
                           square=True,
                           vmin=0.0,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    '''
    Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
    :param box1: predicted box, shape: torch.Size([正样本网格总数,4]), 4表示predict框中心坐标减当前正样本网格左上角坐标的结果(x,y)+predict框的(w,h)
    :param box2: target box，shape为：torch.Size([正样本网格总数, 4])，4表示当前正样本网格需预测标记框的中心点坐标减正样本网格左上角坐标结果(x,y)(取值范围-0.5-1.5)+当前正样本网格需预测标记框的实际(w,h)
    :param xywh: 默认True
    :param GIoU:
    :param DIoU:
    :param CIoU:
    :param eps:
    :return: IoU/GIoU/DIoU/CIoU，shape: torch.Size([正样本网格总数, 1])，当前正样本网格所预测出的predict框和实际分配标记框的交并比
    '''

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1) # torch.Size([正样本网格总数, 1])
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area，box1是predict框中心坐标减当前正样本网格左上角坐标的结果，box2是target坐标中心坐标减当前正样本网格左上角坐标的结果，box1-box2就相当于predict坐标减实际target坐标
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def bbox_ioa(box1, box2, eps=1E-7):
    """
    返回box2中每个边框和box1框的交集在box2每个框面积中的占比
    intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array shape: (4,)
    box2:       np.array shape: (画布上全部nums_objects数量n, 4)，4 对应 各物体边框在画布上的实际位置[x1, y1, x2, y2]（没有归一化）
    returns:    np.array shape: (n,) box2中每个边框和box1框的交集在box2每个框面积中的占比
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1   # b1_x1为纯数
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T # T 表示转置，b2_x1.shape(n,)

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0) # clip(0)表示截断，负数取0

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------


def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    '''
    绘制并保存各类别P-R曲线图iou@0.5
    :param px: np.linspace(0, 1, 1000)
    :param py: list，长度为nc（验证集标签框类别去重后类别的个数），py[i].shape: (1000,) py[i]对应某类下np.linspace(0, 1, 1000)作为recall线性插值出的precision,iou@0.5
    :param ap: shape: (nc, 10), 10对应各类别在10个iou阈值下的AP
    :param save_dir: 保存路径，Path('runs\train\exp12') / 'PR_curve.png'
    :param names: dict，当前验证集中存在的类别，{0:'person', 1:'bicycle', ..., nc-1:'toothbrush'}
    :return:
    '''
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1) # (1000, nc)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric'):
    '''
    绘制并保存各类别P/R/F1-confidence曲线图(iou@0.5)
    :param px: np.linspace(0, 1, 1000)
    :param py: shape: (nc,1000), py[i]对应某类别下置信度阈值从0~1(取1000个值)时指标(recall/precision/F1)的对应取值 IoU@0.5
    :param save_dir: 保存路径，Path('runs\train\exp12') / 'P/R/F1_curve.png'
    :param names:
    :param xlabel: 默认'Confidence'
    :param ylabel: Precision/Recall/F1
    :return:
    '''
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05) # 对数组平滑一下
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close()
