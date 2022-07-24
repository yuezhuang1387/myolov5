# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory E:\裂缝\yolo\myolov5
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build，self.stride 通过在m.stride的初始化部分进行设置，最终为tensor([ 8., 16., 32.])
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    # self.anchors 在yolo部分中的初始化后为,wh
    # tensor([[[ 1.25000,  1.62500],
    #          [ 2.00000,  3.75000],
    #          [ 4.12500,  2.87500]],
    #
    #         [[ 1.87500,  3.81250],
    #          [ 3.87500,  2.81250],
    #          [ 3.68750,  7.43750]],
    #
    #         [[ 3.62500,  2.81250],
    #          [ 4.87500,  6.18750],
    #          [11.65625, 10.18750]]], device='cuda:0')
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        '''
        经过FPN和PAN后融合的三个特征图，通过detect进行输出(包含三组head)
        :param nc: 类别：80
        :param anchors:wh, [[10, 13, 16, 30, 33, 23],
                         [30, 61, 62, 45, 59, 119],
                         [116, 90, 156, 198, 373, 326]]
        :param ch: 三个输入特征图的通道数，[128, 256, 512]
        :param inplace:
        '''
        super().__init__()
        self.nc = nc  # 类别数
        self.no = nc + 5  # 类别数+5
        self.nl = len(anchors)  # 使用了几层的输出特征图，3
        self.na = len(anchors[0]) // 2  # number of anchors，3
        self.grid = [torch.zeros(1)] * self.nl  # init grid [tensor([0.]), tensor([0.]), tensor([0.])]
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment), yolo部分初始化为True

    def forward(self, x):
        '''
        :param x: list,len=3
                torch.Size([1, 128, 80, 80])
                torch.Size([1, 256, 40, 40])
                torch.Size([1, 512, 20, 20])
        :return: 训练时，list,len=3
                    torch.Size([1, 3, 80, 80, 85])
                    torch.Size([1, 3, 40, 40, 85])
                    torch.Size([1, 3, 20, 20, 85])
        '''
        # self.training为Module模块的参数，当设置model.train()时self.training为True，设置model.eval()时self.training为False
        # self.export: 一直为False
        z = []  # inference output
        # print(f'self.grid: {self.grid}')
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # torch.Size([N,85*3,H,W])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4,
                                                                   2).contiguous()  # torch.Size([N,85*3,H,W]) -> torch.Size([N,3,H,W,85])

            # 只有设置model.eval()才会执行下面程序
            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                    # self.grid[i].shape: torch.Size([1,3,H,W,2]), torch.Size([1,k,i,j,l]),表示(i,j)处的网格的第k个描框对应的坐标(l=0为x坐标, l=1为y坐标)(都减去了0.5的偏置，k取0/1/2都一样)
                    # self.anchor_grid[i].shape: torch.Size([1,3,H,W,2]), torch.Size([1,k,i,j,l]),表示(i,j)处的网格的第k个描框尺寸(l=0为w, l=1为h)(i,j取任意值都一样)

                y = x[i].sigmoid()
                if self.inplace: # 默认为True
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy, shape: torch.Size([N,3,H,W,2]) 2表示predict框实际中心坐标xy(映射到yolo模型实际输入图像尺寸上(640,640)或(672,另一个可被32整除))
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh, shape: torch.Size([N,3,H,W,2]) 2表示predict框实际wh(映射到yolo模型实际输入图像尺寸上(640,640)或(672,另一个可被32整除))
                    # torch.Size([N,3,H,W,4])表示predict框的置信度
                    # torch.Size([N,3,H,W,5:])表示predict框对80个类别的预测概率
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no)) # torch.Size([N, 3, H, W, 85]) -> torch.Size([N, 3*H*W, 85])
                # z[i].shape: torch.Size([N, 3*H*W, 85])

        # self.training为Module模块的参数，当设置model.train()时self.training为True，设置model.eval()时self.training为False
        # self.export: 一直为False
        # (torch.cat(z, 1), x):
        #       torch.cat(z, 1): shape: torch.Size([N, 全部预测先验框个数(3*H1*W1+3*H2*W2+3*H3*W3), 85])
        #                               85中0:2表示每个predict框实际中心坐标xy(映射到yolo模型实际输入图像尺寸上(640,640)或(672,另一个可被32整除))
        #                               85中2:4表示predict框实际wh(映射到yolo模型实际输入图像尺寸上(640,640)或(672,另一个可被32整除))
        #                               85张5表示predict框的置信度
        #                               85中5:85表示predict框对80个类别的预测概率
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
        # 最后一句return等价于
        # if self.training:
        #     return x
        # else:
        #     if self.export:
        #         return (torch.cat(z, 1),)
        #     else:
        #         return (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        '''
        :param nx: 某层特征图的W
        :param ny: 某层特征图的H
        :param i: 第几层特征图
        :return: grid.shape: torch.Size([1,3,H,W,2]), torch.Size([1,k,i,j,l]),表示(i,j)处的网格的第k个描框对应的坐标(l=0为x坐标, l=1为y坐标)(都减去了0.5的偏置，k取0/1/2都一样)
                 anchor_grid.shape: torch.Size([1,3,H,W,2]), torch.Size([1,k,i,j,l]),表示(i,j)处的网格的第k个描框实际尺寸(l=0为w, l=1为h)(i,j取任意值都一样)
        '''
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape (1,3,H,W,2)
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t) # shape: torch.Size([H/W])
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij') # yv.shape = xv.shape = torch.Size([H,W])
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        # grid.shape: torch.Size([1,3,H,W,2]), torch.Size([1,k,i,j,l]),表示(i,j)处的网格的第k个描框对应的坐标(l=0为x坐标, l=1为y坐标)(都减去了0.5的偏置，k取0/1/2都一样)
        # anchor_grid.shape: torch.Size([1,3,H,W,2]), torch.Size([1,k,i,j,l]),表示(i,j)处的网格的第k个描框实际尺寸(l=0为w, l=1为h)(i,j取任意值都一样)
        return grid, anchor_grid


class Model(nn.Module):
    # YOLOv5 model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        '''
        :param cfg:
        :param ch: 输入通道3
        :param nc: 输出通道
        :param anchors: 此处不为None时，会覆盖掉yolov5s.yaml中的anchors数据
        '''
        super().__init__()
        if isinstance(cfg, dict):
            # model dict
            self.yaml = cfg
        else:
            # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name  # yolov5s.yaml
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict
                # print(self.yaml)
                # {
                #   'nc': 80,
                #   'depth_multiple': 0.33,
                #   'width_multiple': 0.5,
                #   'anchors':
                #       [
                #           [10, 13, 16, 30, 33, 23],
                #           [30, 61, 62, 45, 59, 119],
                #           [116, 90, 156, 198, 373, 326]
                #       ],
                #   'backbone':
                #       [
                #           [-1, 1, 'Conv', [64, 6, 2, 2]],
                #           [-1, 1, 'Conv', [128, 3, 2]],
                #           [-1, 3, 'C3', [128]],
                #           [-1, 1, 'Conv', [256, 3, 2]],
                #           [-1, 6, 'C3', [256]],
                #           [-1, 1, 'Conv', [512, 3, 2]],
                #           [-1, 9, 'C3', [512]],
                #           [-1, 1, 'Conv', [1024, 3, 2]],
                #           [-1, 3, 'C3', [1024]],
                #           [-1, 1, 'SPPF', [1024, 5]]],
                #    'head': ......后续部分见yaml文件
        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels = 3
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names 0~79
        self.inplace = self.yaml.get('inplace', True)  # True

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride  # tensor([ 8., 16., 32.])
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        '''
        :param x:
        :param augment: 是否使用数据增强的推理
        :param profile: 是否查看模型每层推理耗时、GFLOPs、Params
        :param visualize: 可视化选项
        :return: list,len=3
                 torch.Size([N, 3, 80, 80, 85])
                 torch.Size([N, 3, 40, 40, 85])
                 torch.Size([N, 3, 20, 20, 85])
        '''
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer, 只有concat和detect层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # 当m.i为[6, 4, 14, 10, 17, 20, 23]几层中的输出时，save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch=[3]):
    '''
    统计模型每层参数和输入并打印log
    :param d: model_dict
            {
                  'nc': 80,
                  'depth_multiple': 0.33,
                  'width_multiple': 0.5,
                  'anchors':
                      [
                          [10, 13, 16, 30, 33, 23],
                          [30, 61, 62, 45, 59, 119],
                          [116, 90, 156, 198, 373, 326]
                      ],
                  'backbone':
                      [
                          [-1, 1, 'Conv', [64, 6, 2, 2]],
                          [-1, 1, 'Conv', [128, 3, 2]],
                          [-1, 3, 'C3', [128]],
                          [-1, 1, 'Conv', [256, 3, 2]],
                          [-1, 6, 'C3', [256]],
                          [-1, 1, 'Conv', [512, 3, 2]],
                          [-1, 9, 'C3', [512]],
                          [-1, 1, 'Conv', [1024, 3, 2]],
                          [-1, 3, 'C3', [1024]],
                          [-1, 1, 'SPPF', [1024, 5]]],
                   'head': ......后续部分见yaml文件
            }
    :param ch: input_channels(list), [3]
    :return: nn.Sequential(*layers)：25层，每层为n个对应操作block(Conv/C3/SPPF/Concat/Upsample/Detect)组成的nn.Sequential
             sorted(save)：4个concat和一个detect对应的索引，[6, 4, 14, 10, 17, 20, 23]

    '''
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors = 3
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) = 3*(80+5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # list相加会直接拼接在一起，len(d['backbone'])=10, len(d['head'])=15, len(d['backbone']+d['head'])=25
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m,
                                  str) else m  # eval strings,eval函数：https://www.runoob.com/python/python-func-eval.html
        # m 通过eval函数后 = m字符实际对应的模块类(或者对象),'Conv'->Conv，在models.common.py文件中定义的类别
        for j, a in enumerate(args):
            # a为args如[64, 6, 2, 2]中的每个元素
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass
        # len(d['backbone']+d['head'])=25次循环中，只有最后一次循环的args会发生改变：['nc', 'anchors'] -> [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]]
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # print(n)
        if m in (Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x):
            c1, c2 = ch[f], args[0]  # c1=3,c2=args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)  # 8的倍数，向上取整

            args = [c1, c2, *args[1:]]  # [in_ch, out_ch, *args[1:]]
            # m = Conv时，args = [3, 32, 6, 2, 2]，[in_ch,out_ch,kernel,stride,padding]
            # m = C3时，args = [64, 64, 1]，[in_ch,out_ch,number]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost, C3x]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)  # f 例如[-1, 6]，即取-1和6的通道数
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type：'models.common.Conv'
        n_params = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, n_params  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{n_params:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    # nn.Sequential(*layers)：25层，每层为n个对应操作block(Conv/C3/SPPF/Concat/Upsample/Detect)组成的nn.Sequential
    #           sorted(save)：4个concat和一个detect对应的索引，[6, 4, 14, 10, 17, 20, 23]
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    # 笔记本上，用于测试git
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=10, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)
    # Options
    # if opt.line_profile:  # profile layer by layer
    #     _ = model(im, profile=True)
    #
    # elif opt.profile:  # profile forward-backward
    #     results = profile(input=im, ops=[model], n=3)
    #
    # elif opt.test:  # test all models
    #     for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
    #         try:
    #             _ = Model(cfg)
    #         except Exception as e:
    #             print(f'Error in {cfg}: {e}')
    #
    # else:  # report fused model summary
    #     model.fuse()
    model.eval()
    out, train_out = model(im.cuda())
    print(len(out))
    print(len(train_out))
    yy = model(im.cuda(), profile=False)  # list,len=3
    model.state_dict()
    for y in yy:
        print(y.shape)
        # torch.Size([1, 3, 80, 80, 85])
        # torch.Size([1, 3, 40, 40, 85])
        # torch.Size([1, 3, 20, 20, 85])

    print(model.model[-1].stride)  #
    print(model.model[-1].anchors)

    for k, v in model.named_parameters():
        v.requires_grad = True  # 默认训练所有层
        # print(f'{k}')
        # model.0.conv.weight
        # model.0.bn.weight
        # model.0.bn.bias
        # ......
        # model.2.cv1.conv.weight
        # model.2.cv1.bn.weight
        # ......
        # model.24.m.2.weight
        # model.24.m.2.bias

    for i, (k, v) in enumerate(model.named_modules()):
        a = 1
        # print(k)
        # print(v)
        # v为模块的名字
        # model
        # model.0
        # model.0.conv
        # model.0.bn
        # model.0.act
        # ......
        # model.24
        # model.24.m
        # model.24.m.0
        # model.24.m.1
        # model.24.m.2
    # print(model.names)
    # for i,n in enumerate(model.names):
    #     print(f'i={i},n={n}')
