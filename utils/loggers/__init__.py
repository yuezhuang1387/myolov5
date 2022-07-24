# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Logging utils
"""

import os
import warnings

import pkg_resources as pkg
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.general import colorstr, cv2, emojis
from utils.loggers.wandb.wandb_utils import WandbLogger
from utils.plots import plot_images, plot_results
from utils.torch_utils import de_parallel

LOGGERS = ('csv', 'tb', 'wandb')  # text-file, TensorBoard, Weights & Biases
RANK = int(os.getenv('RANK', -1))

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
    if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.2') and RANK in {0, -1}:
        try:
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:  # known non-TTY terminal issue
            wandb_login_success = False
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None


class Loggers():
    # YOLOv5 Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = [
            'train/box_loss',
            'train/obj_loss',
            'train/cls_loss',  # train loss
            'metrics/precision',
            'metrics/recall',
            'metrics/mAP_0.5',
            'metrics/mAP_0.5:0.95',  # metrics
            'val/box_loss',
            'val/obj_loss',
            'val/cls_loss',  # val loss
            'x/lr0',
            'x/lr1',
            'x/lr2']  # params
        self.best_keys = ['best/epoch', 'best/precision', 'best/recall', 'best/mAP_0.5', 'best/mAP_0.5:0.95']
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # Message
        if not wandb:
            prefix = colorstr('Weights & Biases: ')
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)"
            self.logger.info(emojis(s))

        # TensorBoard
        s = self.save_dir
        if 'tb' in self.include and not self.opt.evolve:
            prefix = colorstr('TensorBoard: ')
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

        # W&B
        if wandb and 'wandb' in self.include:
            wandb_artifact_resume = isinstance(self.opt.resume, str) and self.opt.resume.startswith('wandb-artifact://')
            run_id = torch.load(self.weights).get('wandb_id') if self.opt.resume and not wandb_artifact_resume else None
            self.opt.hyp = self.hyp  # add hyperparameters
            self.wandb = WandbLogger(self.opt, run_id)
            # temp warn. because nested artifacts not supported after 0.12.10
            if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.11'):
                self.logger.warning(
                    "YOLOv5 temporarily requires wandb version 0.12.10 or below. Some features may not work as expected."
                )
        else:
            self.wandb = None

    def on_train_start(self):
        # Callback runs on train start
        pass

    def on_pretrain_routine_end(self):
        # Callback runs on pre-train routine end
        paths = self.save_dir.glob('*labels*.jpg')  # training labels
        if self.wandb:
            # 当self.save_dir=Path('runs\train\exp2')时，下方列表为空
            self.wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]})

    def on_train_batch_end(self, ni, model, imgs, targets, paths, plots):
        '''
        tensorboard可视化模型结构+生成(0~2)train-batch的数据图像保存到本地+将图像展示在wandb中
        :param ni: 从训练开始一直到当前的总iterator数
        :param model:
        :param imgs: shape: torch.Size([N,3,H,W])，float32, 归一化0~1
        :param targets: shape: torch.Size([N个图像标签中框总数,6]) 第一列表明该框所在的图像是当前batch中的第几张图，第二列为框类别，后四列为各框归一化坐标(x_center, y_center, w, h)
        :param paths: 元组，len(path)=batchsize, path[0]为当前图片绝对路径,'E:\裂缝\yolo\datasets\coco128\images\train2017\000000000357.jpg'
        :param plots: True
        :return:
        '''
        # Callback runs on train batch end
        if plots:
            if ni == 0:
                if not self.opt.sync_bn:  # --sync known issue https://github.com/ultralytics/yolov5/issues/3754
                    # 一般不会开多卡sync_bn
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')  # suppress jit trace warning
                        self.tb.add_graph(torch.jit.trace(de_parallel(model), imgs[0:1], strict=False), [])
                        # add_graph函数：使用tensorboard可视化模型结构，https://blog.csdn.net/weixin_43183872/article/details/108329776
            if ni < 3:
                f = self.save_dir / f'train_batch{ni}.jpg'  # filename
                plot_images(imgs, targets, paths, f) # 展示当前batch的训练数据图，存到'train_batch0.jpg'等图片中

            if self.wandb and ni == 10:
                files = sorted(self.save_dir.glob('train*.jpg'))
                # 将图像展示在wandb中
                self.wandb.log({'Mosaics': [wandb.Image(str(f), caption=f.name) for f in files if f.exists()]})

    def on_train_epoch_end(self, epoch):
        '''
        更新self.wandb中的current_epoch
        :param epoch:
        :return:
        '''
        if self.wandb:
            self.wandb.current_epoch = epoch + 1

    def on_val_image_end(self, pred, predn, path, names, im):
        # 每隔30个epoch（大于0），将当前epoch前16张val图像和预测结果以wandb.Image形式存入list中
        if self.wandb:
            self.wandb.val_one_image(pred, predn, path, names, im)

    def on_val_end(self):
        # val集中前三个batch的pred/labels结果（每个batch只前取16张图）存到wandb字典的"Validation"字段
        if self.wandb:
            files = sorted(self.save_dir.glob('val*.jpg'))
            self.wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in files]})

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        '''
        每一个epoch结束后将train+val的13个指标追加到'results.csv'文件中，并将指标存到tensorboard和wandb字典 + 更新当前epoch全部字典值到wandb网页
        :param vals: 保存各指标的list，长度为13，均为float类型，对应self.keys:
        [
            'train/box_loss',
            'train/obj_loss',
            'train/cls_loss',  # train loss
            'metrics/precision',
            'metrics/recall',
            'metrics/mAP_0.5',
            'metrics/mAP_0.5:0.95',  # metrics
            'val/box_loss',
            'val/obj_loss',
            'val/cls_loss',  # val loss
            'x/lr0',
            'x/lr1',
            'x/lr2']
        :param epoch:
        :param best_fitness:
        :param fi:
        :return:
        '''
        # Callback runs at the end of each fit (train+val) epoch
        x = dict(zip(self.keys, vals)) # key为指标名，value为数值

        # 1、保存csv文件
        if self.csv:
            file = self.save_dir / 'results.csv'
            n = len(x) + 1  # number of cols，14
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')  # add header, rstrip(',')删除末尾的','
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        # 2、存到Tensorboard，会实时同步
        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)

        # 3、将指标存到wandb字典，此处还没实时同步
        if self.wandb:
            if best_fitness == fi:
                # 两者相等说明出现的新的最大值
                best_results = [epoch] + vals[3:7]
                for i, name in enumerate(self.best_keys): # self.best_keys = ['best/epoch', 'best/precision', 'best/recall', 'best/mAP_0.5', 'best/mAP_0.5:0.95']
                    self.wandb.wandb_run.summary[name] = best_results[i]  # log best results in the summary
            self.wandb.log(x)
            self.wandb.end_epoch(best_result=best_fitness == fi) # 将每隔30个epoch（大于0）的val图像预测结果存到wandb字典中 + 更新当前epoch全部字典值到wandb网页

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        '''
        :param last: 'runs\train\exp12\weights\last.pt'
        :param epoch: 当前轮数
        :param final_epoch: True/False
        :param best_fitness: 最高精度
        :param fi: 当前精度
        :return:
        '''
        # Callback runs on model save event
        if self.wandb:
            if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1: # 因为opt.save_period默认=-1，下方函数不会被调用
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)

    def on_train_end(self, last, best, plots, epoch, results):
        '''
        绘制csv对应图像，并将csv图像/混淆矩阵图像/P/R/F1/P-R图像传到wandb字典中的"Results"字段下，最后更新全部值到wandb网页
        :param last: 'runs\train\exp12\weights\last.pt'
        :param best: 'runs\train\exp12\weights\best.pt'
        :param plots: True
        :param epoch: 299
        :param results: (mp, mr, map50, map, box-loss, conf-loss, cls-loss)
                         mp: float数，在(各类别平均)F1取最大的置信度阈值下，各个类别precision(IoU@0.5)的平均值
                         mr: float数，在(各类别平均)F1取最大的置信度阈值下，各个类别recall(IoU@0.5)的平均值
                         map50: float数，各类别(在IoU@0.5阈值下)AP的平均值
                         map: float数，各类别(在10个IoU阈值0.5:0.95平均值下)AP的平均值
                         box-loss: float数，平均值
                         conf-loss: float数，平均值
                         cls-loss: float数，平均值
        :return:
        '''
        if plots:
            plot_results(file=self.save_dir / 'results.csv')  # save results.png
        files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter
        self.logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")

        if self.tb:
            for f in files:
                self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC')

        if self.wandb:
            self.wandb.log(dict(zip(self.keys[3:10], results)))
            self.wandb.log({"Results": [wandb.Image(str(f), caption=f.name) for f in files]})
            # Calling wandb.log. TODO: Refactor this into WandbLogger.log_model
            if not self.opt.evolve:
                wandb.log_artifact(str(best if best.exists() else last),
                                   type='model',
                                   name=f'run_{self.wandb.wandb_run.id}_model',
                                   aliases=['latest', 'best', 'stripped'])
            self.wandb.finish_run() # 展示当前epoch中wandb字典中存储的全部字段到wandb网页

    def on_params_update(self, params):
        # Update hyperparams or configs of the experiment
        # params: A dict containing {param: value} pairs
        if self.wandb:
            self.wandb.wandb_run.config.update(params, allow_val_change=True)
