# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    '''
    :param eps: 超参数中设置的'label_smoothing'，默认0.0
    :return: return positive, negative label smoothing BCE targets
    '''
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction  # 默认为'mean'
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        '''
        :param pred: shape: torch.Size([正样本网格总数,80]) 表示正样本网格实际模型各类别推理结果logit
        :param true: shape: torch.Size([正样本网格总数,80])，每个正样本网格需要预测的边界框类别处填充0.95，其余为0.05(即每行的80个元素一个为0.95，其余均为0.05)
        :return: 单一的tensor数
        '''
        loss = self.loss_fcn(pred,
                             true)  # torch.Size([正样本网格总数,80])，因为设置了self.loss_fcn.reduction = 'none'，表示每个正样本网格每个类别的nn.BCEWithLogitsLoss
        # loss 实际 = -true*torch.log(pred.sigmoid())-(1-true)*torch.log(1-pred.sigmoid())
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        # 因为true中取值为(1或0)（考虑标签平滑时为0.95或0.05），true为1时alpha_factor=0.25/true为0时alpha_factor=0.75
        # aplha_factor此处的设置和理论有点矛盾，按道理应该是true为1时aplha_factor的值应该更大，但原论文中实验发现aplha=0.25效果却更好
        # （一些博客解释是由于正样本一般是难分样本，负样本是易分样本，modulating_factor会过大的加剧两者之间的差异（即希望加剧差异，但不希望太大），故又通过aplha_factor平衡回去，服了）
        modulating_factor = (1.0 - p_t) ** self.gamma  # 给置信度高的一个更小的权值因子
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria，BCELoss本身为二分类，而BCEWithLogitsLoss可为单标签二分类或多标签二分类
        # 此处pos_weight用法解析：https://blog.csdn.net/zfhsfdhdfajhsr/article/details/118221229?
        # pytorch官方文档：https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
        # pos_weight相当于在当前类别的正样本loss上乘了一个权值，默认1
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['cls_pw']], device=device))  # 相当于nn.Sigmoid+BCELoss(BCEloss本身分为单标签二分类和多标签二分类)
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['obj_pw']], device=device))  # 相当于nn.Sigmoid+BCELoss(BCEloss本身分为单标签二分类和多标签二分类)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # 0.95, 0.05

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma，默认0.0
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        # self.balance: [4.0, 1.0, 0.4]
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index，默认0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors，3
        self.nc = m.nc  # number of classes，80
        self.nl = m.nl  # number of layers，3
        self.anchors = m.anchors
        # 在yolo部分中的初始化后为：
        # tensor([[[ 1.25000,  1.62500],
        #          [ 2.00000,  3.75000],
        #          [ 4.12500,  2.87500]],
        #         [[ 1.87500,  3.81250],
        #          [ 3.87500,  2.81250],
        #          [ 3.68750,  7.43750]],
        #         [[ 3.62500,  2.81250],
        #          [ 4.87500,  6.18750],
        #          [11.65625, 10.18750]]], device='cuda:0')
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        '''
        :param p: 长度为3的list，保存3个tensor
                    torch.Size([N, 3, 80, 80, 85])
                    torch.Size([N, 3, 40, 40, 85])
                    torch.Size([N, 3, 20, 20, 85])
        :param targets: shape: torch.Size([N个图像标签中框总数,6]) 第一列表明该框所在的图像是当前batch中的第几张图，第二列为框类别，后四列为各框归一化坐标(x_center, y_center, w, h)
        :return:
                (lbox + lobj + lcls) * bs :
                    shape: torch.Size([1])，当前batchsize下全部box+置信度+分类损失之和
                torch.cat((lbox, lobj, lcls)).detach():
                    shape: torch.Size([3])，(box-loss,置信度loss,分类loss)，均为平均值
        '''
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # len(tcls)=len(tbox)=len(indics)=len(anch)=3，3表示取了三层特征图
        #    tcls[0]: shape为：torch.Size([某层(尺寸)下正样本网格总数])，表示每个正样本网格需预测的边界框类别
        #    tbox[0]: shape为：torch.Size([某层(尺寸)下正样本网格总数, 4])，4表示当前正样本网格需预测标记框的中心点坐标减正样本网格左上角坐标结果(x,y)(取值范围-0.5-1.5)+当前正样本网格需预测标记框的实际(w,h)
        #    indices[0]: 为一个元组(b,a,gj,gi)
        #         b.shape: torch.Size([ 正样本网格总数 ])，当前正样本网格需要预测的边界框属于batch中第几张图
        #         a.shape: torch.Size([ 正样本网格总数 ])，当前正样本网格需要预测的边界框对应的描框anchors(因为一个网格对应三个anchors，只能取0/1/2)
        #         gj.shape:torch.Size([ 正样本网格总数 ])，当前正样本网格左上角y坐标 (取值范围0~H-1)
        #         gi.shape:torch.Size([ 正样本网格总数 ])，当前正样本网格左上角x坐标 (取值范围0~W-1)
        #    anchors[0]: shape为：torch.Size([正样本网格总数, 2])，2表示当前正样本网格需预测的标记框对应的描框(w,h)，在当前特征图上的尺寸

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            # b.shape: torch.Size([ 正样本网格总数 ])，当前正样本网格需要预测的边界框属于batch中第几张图
            # a.shape: torch.Size([ 正样本网格总数 ])，当前正样本网格需要预测的边界框对应的描框anchors(因为一个网格对应三个anchors，只能取0/1/2)
            # gj.shape:torch.Size([ 正样本网格总数 ])，当前正样本网格左上角y坐标 (取值范围0~H-1)
            # gi.shape:torch.Size([ 正样本网格总数 ])，当前正样本网格左上角x坐标 (取值范围0~W-1)
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype,
                               device=self.device)  # target obj，shape: torch.Size([N,3,H.W])

            n = b.shape[0]  # number of targets，正样本网格总数
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
                # pi[b, a, gj, gi].shape: torch.Size([正样本网格总数,85])，表示正样本网格的实际模型推理结果
                # pxy.shape: torch.Size([正样本网格总数,2]) 表示正样本网格实际模型推理结果的(px,py)
                # pwh.shape: torch.Size([正样本网格总数,2]) 表示正样本网格实际模型推理结果的(pw,ph)
                # _.shape: torch.Size([正样本网格总数,1]) 表示正样本网格实际模型推理结果置信度
                # pcls.shape: torch.Size([正样本网格总数,80]) 表示正样本网格实际模型各类别推理结果

                # 1、Regression损失
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh),
                                 1)  # predicted box, shape: torch.Size([正样本网格总数,4]), 4表示predict框中心坐标减当前正样本网格左上角坐标的结果(x,y)+predict框的(w,h)
                iou = bbox_iou(pbox, tbox[i],
                               CIoU=True).squeeze()  # torch.Size([正样本网格总数])，当前正样本网格所预测出的predict框和实际分配标记框的交并比
                lbox += (1.0 - iou).mean()  # iou loss, (1.0 - iou).mean()为一个单独的tensor数，shape：torch.Size([])

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()  # torch.Size([正样本网格总数]) ，返回iou从小到大排序的索引
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:  # self.gr默认=1
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio，torch.Size([N,3,H,W])，其中有（正样本网格总数）个元素值不为0，其余为iou，
                # tobj[i,j,k,l]中，i表示某一正样本网格需要预测的边界框属于batch中第几张图、j表示正样本网格需要预测的边界框对应的描框anchors值(取0/1/2)、k表示正样本网格左上角y坐标 (取值范围0~H-1)、l表示正样本网格左上角x坐标 (取值范围0~W-1)
                # tobj[i,j,k,l]表示当前正样本网格predic框和target框的交并比

                # 2、Classification损失
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # shape为torch.Size([正样本网格总数,80])，全部填充0.05
                    t[range(n), tcls[
                        i]] = self.cp  # targets，t.shape: torch.Size([正样本网格总数,80])，每个正样本网格需要预测的边界框类别处填充0.95，其余为0.05(即每行的80个元素一个为0.95，其余均为0.05)
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            # 3、置信度损失
            obji = self.BCEobj(pi[..., 4], tobj)
            # pi[..., 4]：shape：torch.Size([N,3,H,W])，表示当前网格下模型推理置信度
            # tobj：iou ratio，torch.Size([N,3,H,W])，其中有（正样本网格总数）个元素值不为0，其余为iou，
            # tobj[i,j,k,l]中，i表示某一正样本网格需要预测的边界框属于batch中第几张图、j表示正样本网格需要预测的边界框对应的描框anchors值(取0/1/2)、k表示正样本网格左上角y坐标 (取值范围0~H-1)、l表示正样本网格左上角x坐标 (取值范围0~W-1)
            # tobj[i,j,k,l]表示当前正样本网格predic框和target框的交并比
            lobj += obji * self.balance[i]  # obj loss，self.balance: [4.0, 1.0, 0.4]
            if self.autobalance: # 默认False
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance: # 默认False
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box'] # 0.05 shape: torch.Size([1])
        lobj *= self.hyp['obj'] # 0.7  shape: torch.Size([1])
        lcls *= self.hyp['cls'] # 0.3  shape: torch.Size([1])
        bs = tobj.shape[0]  # batch size

        # (lbox + lobj + lcls) * bs : shape: torch.Size([1])，当前batchsize下全部box+置信度+分类损失之和
        # torch.cat((lbox, lobj, lcls)).detach(): shape: torch.Size([3])，(box-loss,置信度loss,分类loss)，均为平均值
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        '''
        为每个标记框匹配合适的描框（可能有多个），然后在此基础上为每个标记框分配3个正样本网格
        跨网格匹配：一个标记框对应了三个正样本网格，从标记框中心坐标所处网格的上、下、左、右四个网格中找到离中心点最近的两个网格，再加上当前网格共三个网格进行匹配，增大了正样本的数量
        :param p: 长度为3的list，保存3个tensor
                    torch.Size([N, 3, 80, 80, 85])
                    torch.Size([N, 3, 40, 40, 85])
                    torch.Size([N, 3, 20, 20, 85])
        :param targets: shape: torch.Size([N个图像标签中框总数,6]) 第一列表明该框所在的图像是当前batch中的第几张图，第二列为框类别，后四列为各框归一化坐标(x_center, y_center, w, h)
        :return: tcls, tbox, indices, anch
                len(tcls)=len(tbox)=len(indics)=len(anch)=3，3表示取了三层特征图
                tcls[0]: shape为：torch.Size([某层(尺寸)下正样本网格总数])，表示每个正样本网格需预测的边界框类别
                tbox[0]: shape为：torch.Size([某层(尺寸)下正样本网格总数, 4])，4表示当前正样本网格需预测标记框的中心点坐标减正样本网格左上角坐标结果(x,y)(取值范围-0.5-1.5)+当前正样本网格需预测标记框的实际(w,h)
                indices[0]: 为一个元组(b,a,gj,gi)
                      b.shape: torch.Size([ 正样本网格总数 ])，当前正样本网格需要预测的边界框属于batch中第几张图
                      a.shape: torch.Size([ 正样本网格总数 ])，当前正样本网格需要预测的边界框对应的描框anchors(因为一个网格对应三个anchors，只能取0/1/2)
                      gj.shape:torch.Size([ 正样本网格总数 ])，当前正样本网格左上角y坐标 (取值范围0~H-1)
                      gi.shape:torch.Size([ 正样本网格总数 ])，当前正样本网格左上角x坐标 (取值范围0~W-1)
                anch[0]: shape为：torch.Size([正样本网格总数, 2])，2表示当前正样本网格需预测的标记框对应的描框(w,h)，在当前特征图上的尺寸
        '''
        na, nt = self.na, targets.shape[0]  # number of anchors, N个图像标签中框总数
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1,
                                                                             nt)  # same as .repeat_interleave(nt), torch.Size([3,N个图像标签中框总数])
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        # targets.repeat(na, 1, 1): shape: torch.Size([na(3),N个图像标签中框总数,6])
        # ai[..., None]): shape: torch.Size([3,N个图像标签中框总数,1])
        # targets: shape: torch.Size([3,N个图像标签中框总数,7]，7中第一列表明该框所在的图像是当前batch中的第几张图、第二列为标注框的类别、[:, :, 2:6]为各框归一化坐标(x_center, y_center, w, h)、最后一列表示标记框对应的anchors(因为一个网格对应三个anchors，只能取0/1/2)
        # 举例：(N=2,N个图像标签中框总数也=2，刚好一个图像上一个类别的框时)
        # tensor([[[0., 0., 0.56360, 0.04329, 0.57948, 0.54632, 0.00000],
        #          [1., 1., 0.87120, 0.73670, 0.16823, 0.00972, 0.00000]],
        #
        #         [[0., 0., 0.56360, 0.04329, 0.57948, 0.54632, 1.00000],
        #          [1., 1., 0.87120, 0.73670, 0.16823, 0.00972, 1.00000]],
        #
        #         [[0., 0., 0.56360, 0.04329, 0.57948, 0.54632, 2.00000],
        #          [1., 1., 0.87120, 0.73670, 0.16823, 0.00972, 2.00000]]])

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],  # j
                [0, 1],  # k
                [-1, 0],  # l
                [0, -1],  # m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            # anchors：wh
            # tensor([[ 1.25000,  1.62500],
            #         [ 2.00000,  3.75000],
            #         [ 4.12500,  2.87500]], device='cuda:0')
            # shape: torch.Size([N, 3, 80, 80, 85])
            gain[2:6] = torch.tensor(shape)[
                [3, 2, 3, 2]]  # tensor([0, 0, 80, 80, 80, 80, 0]) WHWH, shape: torch.Size([7])

            # Match targets to anchors
            t = targets * gain
            # t.shape,torch.Size([3, N个图像标签中框总数, 7]),7中第一列表明该框所在的图像是当前batch中的第几张图、第二列为标注框的类别、[:,:,2:6]为框在特征图上的实际(x_center, y_center, w, h)坐标、最后一列表示标记框对应的anchors(因为一个网格对应三个anchors，只能取0/1/2)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio, shape: torch.Size([3,N个图像标签中框总数,2])，每个框的实际wh除描框的wh，有三个描框
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp[
                    'anchor_t']  # compare, torch.Size([3,N个图像标签中框总数])，从[labels框w/描框w,labels框h/描框h,描框w/labels框w,描框h/labels框h]中选的最大值
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[
                    j]  # 筛选出边框/描框尺寸ratio小于'anchor_t'的框(比例差异过大就丢弃掉), shape: torch.Size([3×N个图像标签中框总数 中 边框/描框尺寸ratio小于'anchor_t'的个数, 7])

                # Offsets
                # 跨网格匹配：一个标记框对应了三个正样本网格，从标记框中心坐标所处网格的上、下、左、右四个网格中找到离中心点最近的两个网格，再加上当前网格共三个网格进行匹配，增大了正样本的数量
                gxy = t[:,
                      2:4]  # grid xy，torch.Size([3×N个图像标签中框总数 中 边框/描框尺寸ratio小于'anchor_t'的个数, 2])，2对应框在特征图上的实际(x_center, y_center)坐标
                gxi = gain[[2, 3]] - gxy  # inverse, tensor([80,80])-gxy,
                j, k = ((gxy % 1 < g) & (gxy > 1)).T  # 描框x坐标相对当前网格左上角x坐标距离是否小于0.5, 描框y坐标相对当前网格左上角y坐标距离是否小于0.5
                l, m = ((gxi % 1 < g) & (gxi > 1)).T  # 描框x坐标相对当前网格右下角x坐标距离是否小于0.5, 描框y坐标相对当前网格右下角y坐标距离是否小于0.5
                # j和l大部分情况取值相反(一个为True另一个为False，但当x坐标处于第一个网格前0.5和最后一个网格后0.5时，两者均取False)
                # k和m大部分情况取值相反(一个为True另一个为False，但当y坐标处于第一个网格前0.5和最后一个网格后0.5时，两者均取False)
                j = torch.stack((torch.ones_like(j), j, k, l,
                                 m))  # j.shape: torch.Size([5, 3×N个图像标签中框总数 中 边框/描框尺寸ratio小于'anchor_t'的个数])，值全部为True或False
                # j中第一行全部为True，即后续会保留全部的gtbox
                # j中第二行只有中心 横坐标靠近方格左边 的gtbox为True，后续作为索引时保留
                # j中第三行只有中心 纵坐标靠近方格上边 的gtbox为True，后续作为索引时保留
                # j中第四行只有中心 横坐标靠近方格右边 的gtbox为True，后续作为索引时保留
                # j中第五行只有中心 纵坐标靠近方格下边 的gtbox为True，后续作为索引时保留
                t = t.repeat((5, 1, 1))[j]
                # 这里将t复制5个，然后使用j来过滤
                # 第一个t是保留所有的gtbox，因为上一步里面增加了一个全为true的维度，
                # 第二个t保留了靠近方格左边的gtbox，
                # 第三个t保留了靠近方格上方的gtbox，
                # 第四个t保留了靠近方格右边的gtbox，
                # 第五个t保留了靠近方格下边的gtbox，
                # t.repeat((5, 1, 1)).shape: torch.Size([5, 3×N个图像标签中框总数 中 边框/描框尺寸ratio小于'anchor_t'的个数, 7])
                # t.shape: torch.Size([3×N个图像标签中框总数 中 边框/描框尺寸ratio小于'anchor_t'的数 × (1+(0~4), (0~4)取几决定于当前标记框的x/y坐标相对于网格左上/右下坐标小于0.5的个数，正常为2), 7])
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                # (torch.zeros_like(gxy)[None] + off[:, None]).shape: torch.Size([5, 3×N个图像标签中框总数 中 边框/描框尺寸ratio小于'anchor_t'的个数, 2])，在5个off上为全部边界框中心坐标加上偏置
                # offsets.shape: torch.Size([3×N个图像标签中框总数 中 边框/描框尺寸ratio小于'anchor_t'的数 × (1+(0~4), (0~4)取几决定于当前标记框的x/y坐标相对于网格左上/右下坐标小于0.5的个数，正常为2), 2])，2表示保留不同位置的gtbox所需要添加的off
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            # bc.shape: torch.Size([3×N框总数中满足ratio条件×(1+2), 2]) (1+2)中的2表示两个临近网格，后面2对应(标记框所在的图像是当前batch中的第几张图, 标注框类别)
            # gxy.shape: torch.Size([3×N框总数中满足ratio条件×(1+2), 2]) (1+2)中的2表示两个临近网格，2对应标记框在特征图上的实际中心坐标(x_center, y_center)
            # ghw.shape: torch.Size([3×N框总数中满足ratio条件×(1+2), 2]) (1+2)中的2表示两个临近网格，2对应标记框在特征图上的实际wh(w, h)
            # a.shape: torch.Size([3×N中满足ratio条件×(1+2), 1]) (1+2)中的2表示两个临近网格，1对应标记框对应的anchors(因为一个网格对应三个anchors，只能取0/1/2)
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            # a.shape: torch.Size([ 3×N框总数中满足ratio条件×(1+2) ])，边界框对应的描框anchors(因为一个网格对应三个anchors，只能取0/1/2)
            # b.shape: torch.Size([ 3×N框总数中满足ratio条件×(1+2) ])，边界框属于当前batch中第几张图
            # c.shape: torch.Size([ 3×N框总数中满足ratio条件×(1+2) ])，边界框类别

            # 跨网格匹配：一个标记框对应了三个正样本网格，从标记框中心坐标所处网格的上、下、左、右四个网格中找到离中心点最近的两个网格，再加上当前网格共三个网格进行匹配，增大了正样本的数量
            gij = (
                    gxy - offsets).long()  # 向下取整，torch.Size([3×N框总数中满足ratio条件×(1+2), 2])，对每个标记框，可得到该框中心点所在网格左上角坐标、和距离中心点最近的两个相邻网格的左上角坐标
            gi, gj = gij.T  # grid indices，torch.Size([3×N框总数中满足ratio条件×(1+2)]) 正样本网格左上角坐标(x,y)

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            # indices.append添加一个元组(b,a,gj,gi)
            # -------------------------显然：(3×N框总数中满足ratio条件×(1+2)) = 和标记框对应的正样本网格总数----------------------
            # b.shape: torch.Size([ 正样本网格总数 ])，当前正样本网格需要预测的边界框属于batch中第几张图
            # a.shape: torch.Size([ 正样本网格总数 ])，当前正样本网格需要预测的边界框对应的描框anchors(因为一个网格对应三个anchors，只能取0/1/2)
            # gj.shape:torch.Size([ 正样本网格总数 ])，当前正样本网格左上角y坐标 (取值范围0~H-1)
            # gi.shape:torch.Size([ 正样本网格总数 ])，当前正样本网格左上角x坐标 (取值范围0~W-1)

            tbox.append(torch.cat((gxy - gij, gwh), 1))
            # tbox.append的shape为：torch.Size([正样本网格总数, 4])，4表示当前正样本网格需预测标记框的中心点坐标减正样本网格左上角坐标结果(x,y)(取值范围-0.5-1.5)+当前正样本网格需预测标记框的实际(w,h)
            anch.append(anchors[a])
            # anch.append的shape为：torch.Size([正样本网格总数, 2])，2表示当前正样本网格需预测的标记框对应的描框(w,h)，在当前特征图上的尺寸
            tcls.append(c)
            # tcls.append的shape为：torch.Size([正样本网格总数])，表示正样本网格需预测的边界框类别

        return tcls, tbox, indices, anch
        # len(tcls)=len(tbox)=len(indics)=len(anch)=3，3表示取了三层特征图
        #       tcls[0]: shape为：torch.Size([某层(尺寸)下正样本网格总数])，表示每个正样本网格需预测的边界框类别
        #       tbox[0]: shape为：torch.Size([某层(尺寸)下正样本网格总数, 4])，4表示当前正样本网格需预测标记框的中心点坐标减正样本网格左上角坐标结果(x,y)(取值范围-0.5-1.5)+当前正样本网格需预测标记框的实际(w,h)
        #       indices[0]: 为一个元组(b,a,gj,gi)
        #             b.shape: torch.Size([ 正样本网格总数 ])，当前正样本网格需要预测的边界框属于batch中第几张图
        #             a.shape: torch.Size([ 正样本网格总数 ])，当前正样本网格需要预测的边界框对应的描框anchors(因为一个网格对应三个anchors，只能取0/1/2)
        #             gj.shape:torch.Size([ 正样本网格总数 ])，当前正样本网格左上角y坐标 (取值范围0~H-1)
        #             gi.shape:torch.Size([ 正样本网格总数 ])，当前正样本网格左上角x坐标 (取值范围0~W-1)
        #       anch[0]: shape为：torch.Size([正样本网格总数, 2])，2表示当前正样本网格需预测的标记框对应的描框(w,h)，在当前特征图上的尺寸
