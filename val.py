# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/val.py --weights yolov5s.pt                 # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (macOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, emojis, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    '''
    对nms后的predict框和labels框进行匹配
    :param detections: shape: torch.Size([当前图像nms最终筛完的预测框数量(不超过300),6])
                       6中0:4表示映射到原图尺寸的实际预测框坐标(x1,y1,x2,y2)
                       6中4表示当前的预测概率值
                       6中5表示当前的预测类别(0~79)
    :param labels: torch.Size([当前图像标签框数,5]),5对应框类别+框映射到原图尺寸的实际坐标(x1,y1,x2,y2)
    :param iouv: tensor([0.50000, 0.55000, 0.60000, 0.65000, 0.70000, 0.75000, 0.80000, 0.85000, 0.90000, 0.95000])
    :return: 令返回值为return，shape：torch.Size([当前图像nms最终筛完的预测框数量(不超过300), 10])
             return中每一列表示: 在该列对应阈值iou[i]下，最终和标签框匹配上的预测框索引处置True（最终能和标签框匹配上的预测框数<=当前图像标签框个数）
             也可将return中每一行看为是某个预测框在不同iou[i]阈值下能否有匹配的标签框(True/False)
    '''
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4]) # torch.Size([当前图像标签框数，当前图像nms筛完剩余的预测框数])
    correct_class = labels[:, 0:1] == detections[:, 5]
    # correct_class.shape: torch.Size([当前图像标签框数，当前图像nms筛完剩余的预测框数]) correct_class[i][j]表示第i个标签框类别是否等于第j个预测框的类别
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        # x为元组，保存交并比满足阈值且类别匹配的索引,len(x)=2, x[0]对应标签框索引、x[1]对应预测框索引，shape均为torch.Size([交并比满足阈值且类别匹配的框数])
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            # matches.shape: (交并比满足阈值且类别匹配的框数, 3) 3对应当前所匹配上的两框(标签框索引, 预测框索引, 两框交并比)
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]] # 按照匹配上的两框交并比从大到小排序
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]] # 去除预测框索引重复的部分(只保留第一个)（因为一个预测框可能和多个标签框对应）
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]] # 去除标签框索引重复的部分(只保留第一个)（一个标签框也可能和多个预测框对应）
                # 最终剩余的matches.shape[0]（即最终匹配框个数） <= 当前图像标签框的个数
            correct[matches[:, 1].astype(int), i] = True # 当前iou[i]这个阈值下最终和标签框匹配上的预测框索引处置True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@torch.no_grad()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):
    '''

    :param data:
            {'path': '../datasets/coco128', 'train': 'E:\\裂缝\\yolo\\datasets\\coco128\\
             images\\train2017', 'val': 'E:\\裂缝\\yolo\\datasets\\coco128\\images\\train2017', 'test': None, 'nc':
              80, 'names': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
             'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
              'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'su
             itcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'ska
             teboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl
             ', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
              'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'v
             ase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'], 'download': 'https://ultralytics.com/asse
             ts/coco128.zip'}
    :param weights: 默认None
    :param batch_size: batch_size // WORLD_SIZE * 2
    :param imgsz: 640
    :param conf_thres: 默认0.001,  # confidence threshold
    :param iou_thres: 默认0.6
    :param task: 'val',  # train, val, test, speed or study
    :param device:
    :param workers:
    :param single_cls: None
    :param augment:
    :param verbose:
    :param save_txt:
    :param save_hybrid: 默认False
    :param save_conf:
    :param save_json:
    :param project:
    :param name:
    :param exist_ok:
    :param half:
    :param dnn:
    :param model:
    :param dataloader: val_loader
    :param save_dir: 'runs\train\exp12'
    :param plots: False
    :param callbacks:
    :param compute_loss: ComputeLoss
    :return:  (mp, mr, map50, map, box-loss, conf-loss, cls-loss), maps, t
              (mp, mr, map50, map, box-loss, conf-loss, cls-loss):
                  mp: float数，在(各类别平均)F1取最大的置信度阈值下，各个类别precision(IoU@0.5)的平均值
                  mr: float数，在(各类别平均)F1取最大的置信度阈值下，各个类别recall(IoU@0.5)的平均值
                  map50: float数，各类别(在IoU@0.5阈值下)AP的平均值
                  map: float数，各类别(在10个IoU阈值0.5:0.95平均值下)AP的平均值
                  box-loss: float数，平均值
                  conf-loss: float数，平均值
                  cls-loss: float数，平均值
              maps: shape；(80,) 表示每个类别的平均AP
                  当类别在验证集标签框中存在时，为该类在10个iou阈值0.5:0.95下的平均AP
                  当类别在验证集标签框中不存在时，为存在的各类别(在10个IoU阈值0.5:0.95平均值下)AP的平均值
              t: (平均每张图预处理耗时ms, 平均每张图推理耗时ms，平均每张图NMS耗时ms)
    '''
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset, False
    nc = 1 if single_cls else int(data['nc'])  # number of classes,80
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95，torch.Size([10])
    niou = iouv.numel() # 10

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights[0]} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad = 0.0 if task in ('speed', 'benchmark') else 0.5
        rect = False if task == 'benchmark' else pt  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0 # 统计val集图片数量
    confusion_matrix = ConfusionMatrix(nc=nc) # 混淆矩阵
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)} # model.names为list，表示各类名称
    # names: 类别字典，{0:'person', 1:'bicycle', ..., 79:'toothbrush'}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000)) # [0, 1, 2, ..., 999]
    s = ('%20s' + '%11s' * 6) % ('Class', 'ImageNums', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # im: shape: torch.Size([N,3,H,W])
        # targets: shape: torch.Size([N个图像标签中框总数,6]) 第一列表明该框所在的图像是当前batch中的第几张图，第二列为框类别，后四列为各框归一化坐标(x_center, y_center, w, h)
        # paths: 元组，len(path)=batchsize, path[0]为当前图片绝对路径,'E:\裂缝\yolo\datasets\coco128\images\train2017\000000000357.jpg'
        # shapes: 元组，len(shapes)=batchsize, shapes[0]:
        #         当使用mosaic数据增强时，为None
        #         当不用mosaic数据增强时，为(h0, w0), ((h / h0, w / w0), pad)，显然此处没用使用mosaic数据增强
        #           其中(h0, w0)为图像最原始尺寸
        #           其中(h, w)为图像第一次缩放后的尺寸，h和w中最大值为640(另一个短边是按原图比例缩放得到，且不一定能被32整除)
        #           其中pad: (dw, dh), 输入img第二次缩小到new_shape范围内后，(相对h,w)需要填充的宽度，dw或dh其中之一为0，另一个为需要填充的宽度/2
        callbacks.run('on_val_batch_start')
        t1 = time_sync()
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1 # 1、累积处理数据to('cuda')时间

        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        # out: shape: torch.Size([N, 全部预测先验框个数(3*H1*W1+3*H2*W2+3*H3*W3), 85])
        #      85中0:2表示每个predict框实际中心坐标xy(映射到yolo模型实际输入图像尺寸上(640,640)或(672,另一个可被32整除))
        #      85中2:4表示predict框实际wh(映射到yolo模型实际输入图像尺寸上(640,640)或(672,另一个可被32整除))
        #      85张5表示predict框的置信度
        #      85中5:85表示predict框对80个类别的预测概率
        # train_out: list,len=3
        #            torch.Size([1, 3, 80, 80, 85])
        #            torch.Size([1, 3, 40, 40, 85])
        #            torch.Size([1, 3, 20, 20, 85])
        dt[1] += time_sync() - t2 # 2、累积模型推理的时间

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # torch.Size([3])，(box-loss, 置信度loss, 分类loss)，均为平均值

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        # targets.shape: torch.Size([N个图像标签中框总数,6]) 第一列表明该框所在的图像是当前batch中的第几张图，第二列为框类别，后四列为各框实际坐标(x_center, y_center, w, h)
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling，默认[]
        t3 = time_sync()

        # 关键操作1：对batch中每张图像预测框进行NMS
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        # len(out)=batchsize
        #    out[i].shape：torch.Size([当前图像nms最终筛完的预测框数量(不超过300),6])
        #    6中0:4表示预测框坐标(x1, y1, x2, y2)-均为实际尺寸坐标(映射到yolo模型实际输入图像尺寸上(640,640)或(672,另一个可被32整除)
        #    6中4表示当前的预测概率值
        #    6中5表示当前的预测类别(0~79)
        dt[2] += time_sync() - t3 # 3、累积NMS时间

        # 4、Metrics
        for si, pred in enumerate(out):
            # pred.shape: torch.Size([当前图像nms最终筛完的预测框数量(不超过300),6])
            #    6中0:4表示预测框坐标(x1, y1, x2, y2)-均为实际尺寸坐标(映射到yolo模型实际输入图像尺寸上(640,640)或(672,另一个可被32整除)
            #    6中4表示当前的预测概率值
            #    6中5表示当前的预测类别(0~79)
            labels = targets[targets[:, 0] == si, 1:] # torch.Size([当前图像标签框数,5]),5对应框类别+框实际坐标(x_center, y_center, w, h)
            nl, npr = labels.shape[0], pred.shape[0]  # 当前图像有的标签框数, 当前图像预测出来的框数
            path, shape = Path(paths[si]), shapes[si][0] # shape=(h0,w0)(当前图像的原始尺寸)
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1 # 统计测val集图片总数量

            if npr == 0:
                # 如果预测为空，则添加空的信息到stats里
                if nl:
                    stats.append((correct, *torch.zeros((3, 0), device=device)))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # 将预测框坐标predn映射到原图尺寸

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes.shape: torch.Size([当前图像标签框数,4]),4对应标签框实际坐标(x1,y1,x2,y2)
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # 将标签框坐标tbox映射到原图尺寸
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # torch.Size([当前图像标签框数,5]),5对应框类别+框映射到原图尺寸的实际坐标(x1,y1,x2,y2)

                # 关键操作2：对batch中每张图nms后的predict框和labels框进行匹配
                correct = process_batch(predn, labelsn, iouv)
                # correct.shape：torch.Size([当前图像nms最终筛完的预测框数量(不超过300), 10])，10=len(iouv)
                # correct中每一列表示: 在该列对应阈值iouv[i]下，最终和标签框匹配上的预测框索引处置True（每一列中最终能和标签框匹配上的预测框数<=当前图像标签框个数）
                # 也可将correct中每一行看为是某个预测框在不同iouv[i]阈值下能否有匹配的标签框(True/False)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:  # 默认False
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json: # 默认False
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary

            # 每隔30个epoch（大于0），将当前epoch前16张val图像和预测结果以wandb.Image形式存入list中
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # 将当前这个batch中前16张图上绘制相应labels框并保存到fname路径
            plot_images(im, output_to_target(out), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # 将当前这个batch中前16张图上绘制相应pred框并保存到fname路径

        callbacks.run('on_val_batch_end') # utils/loggers/__init__.py没有此函数的实现

    # stats为list，长度为val_dataloader全部图片数
    # stats[i]: (correct, pred[:, 4], pred[:, 5], labels[:, 0])
    #            correct: shape：torch.Size([当前图像nms最终筛完的预测框数量(不超过300), 10])，10=len(iouv)
    #                     correct中每一列表示: 在该列对应阈值iouv[i]下，最终和标签框匹配上的预测框索引处置True（每一列中最终能和标签框匹配上的预测框数<=当前图像标签框个数）
    #                     也可将correct中每一行看为是某个预测框在不同iouv[i]阈值下能否有匹配的标签框(True/False)
    #            pred[:, 4]: 框预测概率，shape: torch.Size([当前图像nms最终筛完的预测框数量(不超过300)])
    #            pred[:, 5]: 框预测类别，shape: torch.Size([当前图像nms最终筛完的预测框数量(不超过300)])
    #            labels[:, 0]): 标签框类别，shape: torch.Size([当前图像标签框数])

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # 对原始stats进行解压
    # stats为长度为4的list，
    # stats[0]: shape: (每张图nms最终筛完预测框数量(不超过300)×验证集图像总数, 10)，每一行看为是某个预测框在不同iouv[i]阈值下能否有匹配的标签框(True/False)，（每一列中最终能和标签框匹配上的预测框数<=当前验证集全部图像标签框个数）
    # stats[1]: shape: (每张图nms最终筛完预测框数量(不超过300)×验证集图像总数,)，该预测框概率
    # stats[2]: shape: (每张图nms最终筛完预测框数量(不超过300)×验证集图像总数,)，该预测框类别
    # stats[3]: shape: (每个图像标签框数×验证集图像总数,)，标签框类别
    if len(stats) and stats[0].any():
        # 关键操作3：计算整个数据集的各个指标
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names) # 计算验证集各类指标，保存iou@0.5的P/R/P-R/F1曲线
        # tp: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的TP框个数(每类别实际标签框个数×recall)，IoU@0.5
        # tp: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的FP框个数(TP/precision-TP，相当于每个类别预测框个数-TP)，IoU@0.5
        # p: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的precision，IoU@0.5
        # r: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的recall，IoU@0.5
        # f1: shape: (nc,) nc为验证集标签框类别数(去重后)，表示在(各类别平均)F1取最大的置信度阈值下，各个类别的F1，IoU@0.5
        # ap: shape: (nc, 10), nc为验证集标签框类别数(去重后)，10对应各类别在10个iou阈值下的AP
        # ap_class: shape: (nc,)，nc为验证集标签框类别数(去重后)，对应各个类别值
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # mp: float数，在(各类别平均)F1取最大的置信度阈值下，各个类别precision(IoU@0.5)的平均值
        # mr: float数，在(各类别平均)F1取最大的置信度阈值下，各个类别recall(IoU@0.5)的平均值
        # map50: float数，各类别(在IoU@0.5阈值下)AP的平均值
        # map: float数，各类别(在10个IoU阈值0.5:0.95平均值下)AP的平均值
        nt = np.bincount(stats[3].astype(int), minlength=nc)  # 验证集中每类标签出现的个数
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        # 展示平均每张图 预处理/推理/NMS 三个阶段的耗时
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        # 绘制混淆矩阵并保存到'confusion_matrix.png'
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map # 此处nc=80；map为float数，各类别(在10个IoU阈值0.5:0.95平均值下)AP的平均值
    for i, c in enumerate(ap_class): # ap_class: shape: (nc,)，nc为验证集标签框类别数(去重后)，对应各个类别值
        # ap: shape: (nc,), nc为验证集标签框类别数(去重后)，为每类(在10个iou阈值下的平均)AP
        maps[c] = ap[i]
        # maps: shape；(80,) 表示每个类别的平均AP
        #       当类别在验证集标签框中存在时，为该类在10个iou阈值0.5:0.95下的平均AP
        #       当类别在验证集标签框中不存在时，为存在的各类别(在10个IoU阈值0.5:0.95平均值下)AP的平均值

    # return: (mp, mr, map50, map, box-loss, conf-loss, cls-loss), maps, t
    #         (mp, mr, map50, map, box-loss, conf-loss, cls-loss):
    #             mp: float数，在(各类别平均)F1取最大的置信度阈值下，各个类别precision(IoU@0.5)的平均值
    #             mr: float数，在(各类别平均)F1取最大的置信度阈值下，各个类别recall(IoU@0.5)的平均值
    #             map50: float数，各类别(在IoU@0.5阈值下)AP的平均值
    #             map: float数，各类别(在10个IoU阈值0.5:0.95平均值下)AP的平均值
    #             box-loss: float数，平均值
    #             conf-loss: float数，平均值
    #             cls-loss: float数，平均值
    #         maps: shape；(80,) 表示每个类别的平均AP
    #             当类别在验证集标签框中存在时，为该类在10个iou阈值0.5:0.95下的平均AP
    #             当类别在验证集标签框中不存在时，为存在的各类别(在10个IoU阈值0.5:0.95平均值下)AP的平均值
    #         t: (平均每张图预处理耗时ms, 平均每张图推理耗时ms，平均每张图NMS耗时ms)
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(emojis(f'WARNING: confidence threshold {opt.conf_thres} > 0.001 produces invalid results ⚠️'))
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
