# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           cv2, is_colab, is_kaggle, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # -1,  https://pytorch.org/docs/stable/elastic/run.html

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    '''
    获取对paths(list)中全部路径加密后的16进制字符串
    Returns a single hash value of a list of paths (files or dirs)
    :pa ram paths:  self.label_files + self.im_files
                    ['E:\\裂缝\\yolo\\datasets\\coco128\\labels\\train2017\\000000000000.txt', ......,
                    'E:\\裂缝\\yolo\\datasets\\coco128\\labels\\train2017\\0000000000127.txt',

                    'E:\\裂缝\\yolo\\datasets\\coco128\\images\\train2017\\000000000000.jpg', ......,
                    'E:\\裂缝\\yolo\\datasets\\coco128\\images\\train2017\\0000000000127.jpg']
                    ]
    :return: '52aafc425bda444a4ef27b73dd43b7c8'
    '''
    # os.path.getsize返回文件大小，以字节为单位
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # 训练数据中images+labels.txt文件的总大小
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash '52aafc425bda444a4ef27b73dd43b7c8'


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # PIL格式size为(width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    except Exception:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90, }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(path,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False):
    '''
    :param path: train和val图片所在文件夹('E:\\裂缝\\yolo\\datasets\\coco128\\images\\train2017')，或者txt文件路径
    :param imgsz: 需要压缩到的图片尺寸，一般为640，会确保能被32整除
    :param batch_size:
    :param stride: 最大下采样倍率，32
    :param single_cls: 将多类数据作为单类进行训练，默认False
    :param hyp: 超参数的yaml设置文件，train文件中读取后传入为字典
    :param augment: train.py中train_dataloader会设置为True
    :param cache: 是否提前缓存图片到内存，以加快训练速度，默认None
    :param pad: 默认为0
    :param rect: rectangular training， 默认False, 是否需要为每批batch数据自适应缩放尺寸，缩放后该批batch图像长宽其一为640，另一个短边能被32整除（当前index所处batch需要的尺寸）
    :param rank:进程序号，用于多卡训练，默认-1
    :param workers: 默认8线程
    :param image_weights: 默认False，是否使用加权图像选择进行训练
    :param quad: 默认False，将批次从 16x3x640x640 重塑为 4x3x1280x1280，感觉有点像focus反向操作，但也不是
    :param prefix: 带颜色的'train: '字符或'val: '字符
    :param shuffle: 训练时为True
    :return:
    '''
    print(f'这里是dataloader............................................dataloader')
    # train.py中shuffle为True
    if rect and shuffle:
        # rect和shuffle会产生冲突，因为使用rect后，dataloader按照index顺序以batchsize为间隔加载到中才能保证一个batch中的图片尺寸一致，而shuffle会随机挑选
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation，train.py中设置为True
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    # 最终线程nw为从[系统每个显卡能分到的最大线程数, batchsize, 用户设置]里面的最小值，说明线程数设置最好小于batchsize(才能明显起加速作用)
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)  # rank == -1表示只有单卡
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self.new_video(path)
                ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    # YOLOv5 local webcam dataloader, i.e. `python detect.py --source 0`
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        s = f'webcam {self.count}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None, s

    def __len__(self):
        return 0


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl==2020.12.2'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0:
                assert not is_colab(), '--source 0 webcam unsupported on Colab. Rerun command in a local environment.'
                assert not is_kaggle(), '--source 0 webcam unsupported on Kaggle. Rerun command in a local environment.'
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    '''
    根据images路径输出labels路径
    :param img_paths:  ['E:\\裂缝\\yolo\\datasets\\coco128\\images\\train2017\\000000000000.jpg', ......,
                           'E:\\裂缝\\yolo\\datasets\\coco128\\images\\train2017\\0000000000127.jpg']
    :return:           ['E:\\裂缝\\yolo\\datasets\\coco128\\labels\\train2017\\000000000000.txt', ......,
                           'E:\\裂缝\\yolo\\datasets\\coco128\\labels\\train2017\\0000000000127.txt']
    '''
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    # rsplit：从右向左开始分割，1表示只分割一段出来，https://www.wolai.com/gobsYeiMNSmpjMjAFWNHUb#iT2AD8HRYGpBET7q4oXztg
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 prefix=''):
        '''
        :param path: # 训练时图片所在路径，E:\裂缝\yolo\datasets\coco128\images\train2017
        :param img_size: # 需缩放到的图片尺寸，一般为640，会确保能被32整除
        :param batch_size:
        :param augment: # 数据增强，训练集时设置为True
        :param hyp: 超参数的yaml设置文件，读取后传入为字典
        :param rect: rectangular training， 默认False, 是否需要为每批batch数据自适应缩放尺寸，缩放后该批batch图像长宽其一为640，另一个短边能被32整除（当前index所处batch需要的尺寸）
        :param image_weights: 默认False，训练时，是否根据GT框的数量分布权重来选择图片，如果图片权重大，则被抽到的次数多
        :param cache_images: 是否提前缓存图片到内存，以加快训练速度，默认None，其他参数包括'disk'和'ram'
        :param single_cls: 将多类数据作为单类进行训练，默认False
        :param stride: 最大下采样倍率，32
        :param pad: 默认0
        :param prefix: 带颜色的'train: '字符或'val: '字符
        '''
        self.img_size = img_size  # 640
        self.augment = augment
        self.hyp = hyp  # 超参数的yaml设置文件，读取后传入为字典
        self.image_weights = image_weights # 这个参数就没有使用过
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]  # [-640//2,-640//2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None

        # 1、获取全部图片路径
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic, <class 'pathlib.WindowsPath'>类型
                if p.is_dir():
                    # 是否文件夹
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # 通过glob函数获取path目录下(包括当前文件夹下以及全部子文件下)全部图片路径到一个list中
                    # f = list(p.rglob('*.*'))  # pathlib
                    # f = ['E:\\裂缝\\yolo\\datasets\\coco128\\images\\train2017\\000000000000.jpg', ......,
                    #       'E:\\裂缝\\yolo\\datasets\\coco128\\images\\train2017\\0000000000127.jpg']
                elif p.is_file():  # file, p: E:\裂缝\yolo\datasets\coco128\train.txt
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep # 'E:\裂缝\yolo\datasets\coco128'
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.im_files = sorted(
                x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)  # 从f中筛选出个格式为图片的路径,进行排序
            # ['E:\\裂缝\\yolo\\datasets\\coco128\\images\\train2017\\000000000000.jpg', ......,
            #  'E:\\裂缝\\yolo\\datasets\\coco128\\images\\train2017\\0000000000127.jpg']
            # os.sep为当前系统(Windows/Linux)下文件的路径分隔符，str.lower()是将字符串中全部大写字符转为小写
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        # 2、Check cache
        self.label_files = img2label_paths(self.im_files)  # labels
        # ['E:\\裂缝\\yolo\\datasets\\coco128\\labels\\train2017\\000000000000.txt', ......,
        #  'E:\\裂缝\\yolo\\datasets\\coco128\\labels\\train2017\\0000000000127.txt']
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        # cache_path: 回到图片所在目录的上级 E:\裂缝\yolo\datasets\coco128\labels\train2017.cache (with_suffix是在添加新的后缀)
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # same version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # same hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache为字典
            # cache[每个images的路径]: [lb, shape, segemnts]
            # # lb.shape: (nums_objects, 1 + 4)，对应当前图片中各物体的类别和归一化坐标(x_center, y_center, w, h)
            # # shape: (W, H), 因为是读取的PIL格式图像
            # # segments: list(segments) = 当前图像中物体个数，segments[0].shape: (num_pixels, 2)，对应某物体像素级标注的xy坐标
            # cache['hash']: 对全部labels + images路径加密后的16进制字符串
            # cache['results']: nf, nm, ne, nc, len(self.im_files)
            # cache['msgs']: msgs  # warnings信息
            # cache['version']: self.cache_version  # cache version

        # 3、Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        # 只有在train模型下设置augment=True且nf<=0才会报错
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

        # 4、Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(
            labels)  # len(self.labels)=labels图片个数，self.labels[0].shape: (nums_objects, 1 + 4)，对应当前图片中各物体的类别和归一化坐标(x_center, y_center, w, h)
        self.shapes = np.array(shapes, dtype=np.float64)  # self.shapes.shape: (labels图片个数, 2)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index，向下取整 (number of images,)
        nb = bi[-1] + 1  # number of batches，总batches数量
        self.batch = bi  # batch index of image, shape: (n,) __getitem__函数中index索引处的图像所属的batch编号
        self.n = n  # number of images
        self.indices = range(n)  # 可迭代对象，0~n-1

        # 5、（正常这一步直接跳过）Update labels，只检测特定的类别（可选）
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)  # shape: (1,0)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        # 6、Rectangular Training，默认False，设置每批batch数据需要自适应缩放到的尺寸
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # self.shapes.shape: (labels图片个数, 2) 2对应当前image的W和H
            ar = s[:, 1] / s[:, 0]  # aspect ratio(H/W), ar.shape: (labels图片个数,)
            irect = ar.argsort()  # 按照ar从小到大排序，输出对应索引 (labels图片个数,)
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb  # nb：number of batches，总batches数量
            for i in range(nb):
                # bi: batchsize index, shape: (number of images,)
                ari = ar[bi == i]  # 第i个batch对应的images的ratio(H/W), shape:(batch_size,)
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    # 极端1：当前batch中图片均呈现H<W, 最终当前batch中数据缩放到(640*H_max/W_max,640)
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    # 极端2：当前batch中图片均呈现H>W，最终当前batch中数据缩放到(640,640*W_min/H_min)
                    shapes[i] = [1, 1 / mini]
                # 正常3：其他情况batch，一个batch中有H<W，也同时存在H>W的图片，最终这些batch中数据直接缩放到(640,640)

            # self.batch_shapes保存每批batch数据需要自适应缩放到的尺寸
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride
            # self.batch_shapes.shape: (总batches数量nb,2)，每个batchsize对应的尺寸都不一样

        # 7、缓存图片到内存（可选，默认None）Cache images into RAM/disk for faster training (WARNING: large datasets may exceed system resources)
        self.ims = [None] * n  # n 为image个数
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]  # 生成npy缓存文件路径
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n, bar_format=BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == 'disk':
                    gb += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    # x[0]: 按图像比例缩放的图像，确保最长边达到640, x[0].shape:(h_resized,w_resized,3)
                    # x[1]: 图像原始尺寸，(h_original,w_original)
                    # x[2]: 图像缩放后的尺寸，(h_resized,w_resized)
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.ims[i].nbytes  # 总共需要的字节数
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        '''
        :param path: E:\裂缝\yolo\datasets\coco128\labels\train2017.cache,Path类型
        :param prefix: 带颜色的'train: '字符或'val: '字符
        :return:x，字典，(k,v)包含：
                        x[每个images的路径]: [lb,shape,segemnts]
                        # lb.shape: (nums_objects, 1 + 4)，对应当前图片中各物体的类别和归一化坐标(x_center, y_center, w, h)
                        # shape: (W, H), 因为是读取的PIL格式图像
                        # segments: list(segments) = 当前图像中物体个数，segments[0].shape: (num_pixels, 2)，对应某物体像素级标注的xy坐标
                        x['hash']: 对全部labels+images路径加密后的16进制字符串
                        x['results']: nf, nm, ne, nc, len(self.im_files)
                        x['msgs']: msgs  # warnings信息
                        x['version']: self.cache_version  # cache version
        '''
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        # path:                         E:\裂缝\yolo\datasets\coco128\labels\train2017.cache
        # path.parent:                  E:\裂缝\yolo\datasets\coco128\labels，所在文件夹
        # path.stem:                    train2017，文件名(不含路径和格式)
        # f'{path.parent / path.stem}': E:\裂缝\yolo\datasets\coco128\labels\train2017
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                        desc=desc,
                        total=len(self.im_files),
                        bar_format=BAR_FORMAT)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                # im_file: 'E:\\裂缝\\yolo\\datasets\\coco128\\images\\train2017\\000000000000.jpg'
                # lb.shape: (nums_objects, 1 + 4)，对应当前图片中各物体的类别和归一化坐标(x_center, y_center, w, h)
                # shape: (W, H)
                # 因为是读取的PIL格式图像
                # segments: list(segments) = 当前图像中物体个数，segments[0].shape: (num_pixels, 2)，对应某物体像素级标注的xy坐标
                # nm_f: lb_file缺失为1，正常为0
                # nf_f: lb_file缺失为0，正常为1
                # ne_f: lb_file（txt文件）中不存在标记框数据时为1，正常为0
                # nc_f：image / labels文件存在错误时为1，正常为0
                # msg：str字符，表明纠正了JPEG文件或者去掉了当前图片中几个重复标注的物体数据
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(
            self.label_files + self.im_files)  # 对输入list中全部路径加密后的16进制字符串，'52aafc425bda444a4ef27b73dd43b7c8'
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time，将x字典保存到cache文件中
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def __len__(self):
        return len(self.im_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        '''
        :return:
               torch.from_numpy(img): img中像素值没有归一化(仍在0~255),dtype=torch.uint8
                       当经过  mosaic数据增强+copy paste数据增强+仿射变换+mixup融合+空域增强+HSV增强后时，shape：torch.Size([3,640,640])
                       当只经过 仿射变换+空域增强+HSV增强后时，shape：torch.Size([3,newshape_H,newshape_W])，长宽其一为640，另一个短边能被32整除（当前index所处batch需要的尺寸）
               labels_out:
                       shape: torch.Size([当前图像中总框数, 1+1+4])，后5列中1对应框类别，4对应各框归一化坐标(x_center, y_center, w, h)
               self.im_files[index]:
                       当前图片绝对路径,'E:\裂缝\yolo\datasets\coco128\images\train2017\000000000357.jpg'
               shape:
                       当使用mosaic数据增强时，为None
                       当不用mosaic数据增强时，为(h0, w0), ((h / h0, w / w0), pad)
                           其中(h0, w0)为图像最原始尺寸
                           其中(h, w)为图像第一次缩放后的尺寸，h和w中最大值为640(另一个短边是按原图比例缩放得到，且不一定能被32整除)
                           其中pad: (dw, dh), 输入img第二次缩小到new_shape范围内后，(相对h,w)需要填充的宽度，dw或dh其中之一为0，另一个为需要填充的宽度/2
        '''
        # self.indices：range(n),n为图像总数,可将self.indices看为list
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp  # 超参数的yaml设置文件，读取后传入为字典
        mosaic = self.mosaic and random.random() < hyp['mosaic']  # 训练时self.mosaic=True, hyp['mosaic']一般为1
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            # img: mosaic数据增强+copy paste数据增强+仿射变换(随机缩放/裁剪）后的画布，训练阶段输出shape：(640,640,3)
            # labels: mosaic数据增强+copy paste数据增强+仿射变换(随机缩放/裁剪）后画布上的相应labels，框个数可能会减少，shape: (全部nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:  # hyp['mixup']可取0/0.04/0.1，有较小概率对当前图像进行mixup数据增强，从图集中再随便挑一张
                img, labels = mixup(img, labels, *self.load_mosaic(
                    random.randint(0, self.n - 1)))  # self.n为训练图像总数，
                # img: mosaic数据增强+copy paste数据增强+仿射变换+mixup融合后的图像，shape：(640,640,3)
                # labels：融合后concat两图像的labels，shape：(融合后图像nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)
            # img: 按长宽比例缩放后的应图像，确保最长边达到640(另一个短边是按原图比例缩放得到，且不一定能被32整除)
            # (h0, w0): 图像原始尺寸，(h_original,w_original)
            # (h, w): 图像缩放后的尺寸，(h_resized,w_resized)

            # 当augment=True时，设置rect为True即可不使用mosaic数据增强，self.rect为True表明每批batch数据需要自适应缩放尺寸
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            # shape为当前index对应batch中图像需要统一缩放到的尺寸 (H,W), shape.shape: (2,) (H和W中一个为640，另一个短边能被32整除)
            img, ratio, pad = letterbox(img, new_shape=shape, auto=False, scaleup=self.augment)
            # letterbox函数是将输入im保持宽高比继续缩放到new_shape范围内，空白处填充color=(114, 114, 114)
            # img: (new_shape_H,new_shape_W,3)将输入im保持图像宽高比缩放到当前index对应batch的统一尺寸new_shape(H,W)内，空白处填color(该尺寸H和W中一个为640，另一个短边能被32整除)
            # ratio：(w_r,h_r)输入图像im最终缩小的系数
            # pad: (dw, dh), 输入img缩小到new_shape范围内后，需要填充的宽度，dw或dh其中之一为0，另一个为需要填充的宽度/2
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[
                index].copy()  # labels.shape: (nums_objects, 1 + 4)，对应当前图片中各物体的类别和归一化坐标(x_center, y_center, w, h)
            if labels.size:  # normalized xywh to pixel xyxy format
                # 更新通过letterbox又一次缩放图像后的labels坐标
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                # labels.shape: (nums_objects,1+4)，对应当前图片中各物体的类别和实际位置[x1, y1, x2, y2]（没有归一化）

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'],
                                                 border=(0, 0)  # 通过对border参数设置为(0,0)即可实现法仿射变换输入输出尺寸相同,不为(0,0)则输入输出尺寸不同
                                                 )
                # img: 仿射变换数据增强，shape: (new_shape_H,new_shape_W,3)，和输入尺寸相同
                # labels: 仿射变换后图像上相应labels，框个数可能会减少，shape: (全部nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
        nl = len(labels)  # 当前图像上框的数量

        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)
            # 返回坐标归一化后的图像labels，shape:(当前图像全部框数量, 1+4)，4对应各框归一化坐标(x_center, y_center, w, h)

        if self.augment:
            # 空域增强(灰度变换)，Albumentations
            # img: 此时img分两类，
            #      当经过mosaic数据增强+copy paste数据增强+仿射变换+mixup融合后时，shape：(640,640,3)
            #      当只经过仿射变换数据增强时，shape：(new_shape_H,new_shape_W,3)，长宽其一为640，另一个短边能被32整除（当前index所处batch需要的尺寸）
            # labels: 坐标归一化后的图像框，shape:(当前图像全部框数量, 1+4)，4对应各框归一化坐标(x_center, y_center, w, h)
            img, labels = self.albumentations(img, labels)  # 空域增强(灰度变换)经过后输入和输出尺寸完全一样
            nl = len(labels)  # 当前图像中的框数

            # HSV color-space，对输入img进行颜色空间增强，输出通道仍为BGR
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
            # labels_out.shape: torch.Size([当前图像中总框数, 1+1+4])，后5列中1对应框类别，4对应各框归一化坐标(x_center, y_center, w, h)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # (H,W,3) to (3,H,W), BGR to RGB, img像素值还没有归一化(仍在0~255)
        img = np.ascontiguousarray(img)  # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快，参考https://zhuanlan.zhihu.com/p/148413517

        # return
        #       torch.from_numpy(img): img中像素值没有归一化(仍在0~255), dtype=torch.uint8
        #               当经过  mosaic数据增强+copy paste数据增强+仿射变换+mixup融合+空域增强+HSV增强后时，shape：torch.Size([3,640,640])
        #               当只经过 仿射变换+空域增强+HSV增强后时，shape：torch.Size([3,newshape_H,newshape_W])，长宽其一为640，另一个短边能被32整除（当前index所处batch需要的尺寸）
        #       labels_out:
        #               shape: torch.Size([当前图像中总框数, 1+1+4])，后5列中1对应框类别，4对应各框归一化坐标(x_center, y_center, w, h)
        #       self.im_files[index]:
        #               当前图片绝对路径,'E:\裂缝\yolo\datasets\coco128\images\train2017\000000000357.jpg'
        #       shapes:
        #               当使用mosaic数据增强时，为None
        #               当不用mosaic数据增强时，为(h0, w0), ((h / h0, w / w0), pad)
        #                   其中(h0, w0)为图像最原始尺寸
        #                   其中(h, w)为图像第一次缩放后的尺寸，h和w中最大值为640(另一个短边是按原图比例缩放得到，且不一定能被32整除)
        #                   其中pad: (dw, dh), 输入img第二次缩小到new_shape范围内后，需要填充的宽度，dw或dh其中之一为0，另一个为需要填充的宽度/2
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    def load_image(self, i):
        '''
        Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        :param i: 索引index
        :return: im: 按长宽比例缩放后的应图像，确保最长边达到640(另一个短边是按原图比例缩放得到，且不一定能被32整除)
                 (h0, w0): 图像原始尺寸，(h_original,w_original)
                 im.shape[:2]: 图像缩放后的尺寸，(h_resized,w_resized)
        '''
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig h0, w0
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
                # 按图像比例缩放，确保最长边达到640
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        else:
            return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images_to_disk(self, i):
        # Saves an image as an *.npy file for faster loading
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def load_mosaic(self, index):
        '''
        YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        mosaic数据增强解析：https://blog.csdn.net/weixin_46142822/article/details/123805663
        :param index: int类型
        :return:
                img4: mosaic数据增强+copy paste数据增强+仿射变换(随机缩放/裁剪）后的画布，训练阶段输出shape：(640,640,3)
                labels4: mosaic数据增强+copy paste数据增强+仿射变换(随机缩放/裁剪）后画布上的相应labels，框个数可能会减少，shape: (全部nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
        '''
        labels4, segments4 = [], []  # labels4、segment4分别存储当前画布上全部更新后的labels和segments
        s = self.img_size
        # self.mosaic_border:[-640//2,-640//2]
        # random.uniform(x, y)方法将随机生成一个在[x,y]范围内的实数，
        # 1、在mosaic画布上随机生成一个点，范围在(320,2*640-320)，mosaic画布的长宽为2*self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 从self.indices中再随机挑选3个额外的图像索引，len(indices)=4
        random.shuffle(indices)  # 将indices随机打乱
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)
            # img: 按长宽比例缩放后的应图像，确保最长边达到640(另一个短边是按原图比例缩放得到，且不一定能被32整除)
            # _: 图像原始尺寸，(h_original,w_original)
            # (h, w): 图像缩放后的尺寸，(h_resized,w_resized)

            # 2、围绕当前随机点放置4块拼图，place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                # base image with 4 tiles，设置基础画布(取值为114)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                # 画布区域 xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                # 图片区域 xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            # 3、将图像上这块区域的像素移到mosaic画布上
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            # 画布上空余没有被图像填充的部分填充
            padw = x1a - x1b  # padw大于0说明画布上存在没有被图像填充的区域
            padh = y1a - y1b  # padh大于0说明画布上存在没有被图像填充的区域

            # 4、更新labels和segments
            # labels.shape: (nums_objects, 1 + 4)，对应当前图片中各物体的类别和归一化坐标(x_center, y_center, w, h)
            # segments: list(segments) = 当前图像中物体个数，segments[0].shape: (num_pixels, 2)，对应某物体像素级标注的xy坐标
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                # 2.1 更新labels
                # xywhn2xyxy输出对应图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                # 2.2 更新segments
                # xyn2xy输出segments[0].shape: (num_pixels, 2)，对应图像移到画布上后segments在画布坐标系上的实际位置（没有归一化）
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)  # labels4存储当前画布上的全部更新后的labels
            segments4.extend(segments)  # segments存储当前画布上的全部更新后的segments
        # labels4为list,len(labels4)=4，labels4[0].shape: (nums_objects, 1 + 4)，对应当前图片(画布需要的4张图片中的某一张)中各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
        # segments为list,len(segments)=当前画布上的全部标记物体总数，segments[0].shape: (num_pixels, 2)，对应图像移到画布上后segments在画布坐标系上的实际位置（没有归一化）

        # Concat/clip labels
        labels4 = np.concatenate(labels4,
                                 axis=0)  # labelss.shape: (画布上全部nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
        for x in (labels4[:, 1:], *segments4):
            np.clip(a=x, a_min=0, a_max=2 * s,
                    out=x)  # 将数组a中的所有数限定到范围a_min和a_max中, clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # 5.1 copy_paste数据增强（只有p>0才会启用）
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
        # img4: 复制粘贴了一些物体后的新mosaic画布，shape: (2*640,2*640,3)
        # labels4: shape: (画布上扩充后的全部nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
        # segments4: segments为list,len(segments)=画布上扩充后的全部标记物体数量，segments[0].shape: (num_pixels, 2)，对应图像移到画布上后segments在画布坐标系上的实际位置（没有归一化）

        # 5.2 随机仿射变换数据增强（包括缩放裁剪旋转等等）
        img4, labels4 = random_perspective(img4,
                                           labels4,
                                           segments4,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border
                                           # 通过对border参数设置为(0,0)即可实现法仿射变换输入输出尺寸相同,不为(0,0)则输入输出尺寸不同
                                           )
        # img4: mosaic数据增强+copy paste数据增强+仿射变换(随机缩放/裁剪）后的画布，训练阶段输出shape：(640,640,3)
        # labels4: mosaic数据增强+copy paste数据增强+仿射变换(随机缩放/裁剪）后画布上的相应labels，框个数可能会减少，shape: (全部nums_objects数量, 1 + 4)，1+4 对应 各物体的类别和图像移到画布上后labels在画布坐标系上的实际位置[x1, y1, x2, y2]（没有归一化）
        return img4, labels4

    def load_mosaic9(self, index):
        # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        img9, labels9 = random_perspective(img9,
                                           labels9,
                                           segments9,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img9, labels9

    @staticmethod
    def collate_fn(batch):
        '''
        对dataloader中每个batch的返回值做一些定制：https://zhuanlan.zhihu.com/p/493400057
        正常使用dataloader时collate_fn函数无需设置，默认的collate_fn函数相当于:
            collate_fn = lambda x: (torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0) for j in range(len(x[0])))
        :param batch: 输入为list，包含batchsize个元组，每个元组形式为(im, label, path, shapes)
        :return:
                torch.stack(im, 0): shape: torch.Size([N,3,H,W])
                torch.cat(label, 0): shape: torch.Size([N个图像标签中框总数,6]) 第一列表明该框所在的图像是当前batch中的第几张图，第二列为框类别，后四列为各框归一化坐标(x_center, y_center, w, h)
                path: 元组，len(path)=batchsize, path[0]为当前图片绝对路径,'E:\裂缝\yolo\datasets\coco128\images\train2017\000000000357.jpg'
                shapes: 元组，len(shapes)=batchsize, shape[0]参见__getitem__输出
        '''
        im, label, path, shapes = zip(*batch)  # transposed
        # im为元组，len(im)=len(label)=len(path)=len(shapes)=batchsize, im[0]/label[0]/path[0]/shapes[0]的尺寸和__getitem__中对应位置的返回尺寸一致
        for i, lb in enumerate(label):
            # lb: shape: torch.Size([当前图像中总框数, 1+1+4])，后5列中1对应框类别，4对应各框归一化坐标(x_center, y_center, w, h)
            lb[:, 0] = i  # add target image index for build_targets(),第一列表明该框所在的图像是当前batch中的第几张图
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        '''
        将批次从 16x3x640x640 重塑为 4x3x1280x1280
        :param batch:  输入为list，包含batchsize个元组，每个元组形式为(im, label, path, shapes)
        :return:
                torch.stack(im4, 0): shape: torch.Size([N/4,3,H*2,W*2])
                torch.cat(label4, 0): shape: torch.Size([N/4个新图像中框的总数,6]) 第一列表明该框所在的图像是当前新batch中的第几张图，第二列为框类别，后四列为各框归一化坐标(x_center, y_center, w, h)
                path: 元组，len(path)=batchsize, path[0]为当前图片绝对路径,'E:\裂缝\yolo\datasets\coco128\images\train2017\000000000357.jpg'
                shapes: 元组，len(shapes)=batchsize, shape[0]参见__getitem__输出

        '''
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4 # 新batchsize
        im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            # 图像可能是取第4*i张图放大，也可能是相近的四张拼一起，服了，这不浪费了很多数据
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear',
                                   align_corners=False)[0].type(img[i].type()) # 取第4*i张图放大
                lb = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2) # 4张图拼在一起
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s # 更新labels坐标
            im4.append(im)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path=DATASETS_DIR / 'coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(str(path) + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path=DATASETS_DIR / 'coco128'):  # from utils.dataloaders import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path=DATASETS_DIR / 'coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.dataloaders import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


def verify_image_label(args):
    '''
    使用PIL库对images和labels的完整性(尺寸、格式)进行验证，验证完cache_labels函数会生成缓存文件，下次训练时就不会再次验证
    :param args:
            im_file: 'E:\\裂缝\\yolo\\datasets\\coco128\\images\\train2017\\000000000000.jpg'
            lb_file: 'E:\\裂缝\\yolo\\datasets\\coco128\\labels\\train2017\\000000000000.txt'
            prefix: 带颜色的'train: '字符或'val: '字符
    :return:
            im_file: 'E:\\裂缝\\yolo\\datasets\\coco128\\images\\train2017\\000000000000.jpg'
            lb.shape: (nums_objects,1+4)，对应当前图片中各物体的类别和归一化坐标(x_center,y_center,w,h)
            shape: (W,H) 因为是读取的PIL格式图像
            segments: list(segments)=当前图像中物体个数，segments[0].shape: (num_pixels, 2)，对应某物体像素级标注的xy坐标
            nm: lb_file缺失为1，正常为0
            nf: lb_file缺失为0，正常为1
            ne: lb_file（txt文件）中不存在标记框数据时为1，正常为0
            nc：image/labels文件存在错误时为1，正常为0
            msg：str字符，表明纠正了JPEG文件或者去掉了当前图片中几个重复标注的物体数据
    '''
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        print(f'shape:')
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'

        # 下面部分是为了矫正JPEG格式图像exif方向问题，即防止读取的图像和系统图片查看器显示的图像方向上产生90度旋转，将其矫正回来，太细节了，https://zhuanlan.zhihu.com/p/85923289/
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                # lb: [['45', '0.479492', '0.688771', '0.955609', '0.5955'],
                #      ['45', '0.736516', '0.247188', '0.498875', '0.476417'],
                #      ['50', '0.637063', '0.732938', '0.494125', '0.510583']]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                    # list(segments)=当前图像中物体个数，segments[0].shape: (num_pixels, 2)，对应某物体像素级标注的xy坐标
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)),
                                        1)  # shape: (nums_objects,1+4)
                lb = np.array(lb, dtype=np.float32)
                # lb.shape: (nums_objects,1+4)，对应当前图片中各物体的类别和归一化坐标(x_center,y_center,w,h)
            nl = len(lb)  # 当前图片中label框出的物体个数
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)  # 去掉lb中的重复标注行，返回排序后的索引i
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = segments[i]
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
                    # 去掉了当前图片中几个重复标注的物体数据
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)  # lb:[] lb.shape: (0,5)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
        # im_file: 'E:\\裂缝\\yolo\\datasets\\coco128\\images\\train2017\\000000000000.jpg'
        # lb.shape: (nums_objects,1+4)，对应当前图片中各物体的类别和归一化坐标(x_center,y_center,w,h)
        # shape: (W,H) 因为是读取的PIL格式图像
        # segments: list(segments)=当前图像中物体个数，segments[0].shape: (num_pixels, 2)，对应某物体像素级标注的xy坐标
        # nm: lb_file缺失为1，正常为0
        # nf: lb_file缺失为0，正常为1
        # ne: lb_file（txt文件）中不存在标记框数据时为1，正常为0
        # nc：image/labels文件存在错误时为1，正常为0
        # msg：str字符，表明纠正了JPEG文件或者去掉了当前图片中几个重复标注的物体数据

    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.dataloaders import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.dataloaders import *; dataset_stats('path/to/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def _round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

    def _find_yaml(dir):
        # Return data.yaml file
        files = list(dir.glob('*.yaml')) or list(dir.rglob('*.yaml'))  # try root level first and then recursive
        assert files, f'No *.yaml file found in {dir}'
        if len(files) > 1:
            files = [f for f in files if f.stem == dir.stem]  # prefer *.yaml files that match dir name
            assert files, f'Multiple *.yaml files found in {dir}, only 1 *.yaml file allowed'
        assert len(files) == 1, f'Multiple *.yaml files found: {files}, only 1 *.yaml file allowed in {dir}'
        return files[0]

    def _unzip(path):
        # Unzip data.zip
        if str(path).endswith('.zip'):  # path is data.zip
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            ZipFile(path).extractall(path=path.parent)  # unzip
            dir = path.with_suffix('')  # dataset directory == zip name
            assert dir.is_dir(), f'Error unzipping {path}, {dir} not found. path/to/abc.zip MUST unzip to path/to/abc/'
            return True, str(dir), _find_yaml(dir)  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def _hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
        f_new = im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, 'JPEG', quality=75, optimize=True)  # save
        except Exception as e:  # use OpenCV
            print(f'WARNING: HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f_new), im)

    zipped, data_dir, yaml_path = _unzip(Path(path))
    try:
        with open(check_yaml(yaml_path), errors='ignore') as f:
            data = yaml.safe_load(f)  # data dict
            if zipped:
                data['path'] = data_dir  # TODO: should this be dir.resolve()?`
    except Exception:
        raise Exception("error/HUB/dataset_stats/yaml_load")

    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}  # statistics dictionary
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # load dataset
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)  # shape(128x80)
        stats[split] = {
            'instance_stats': {
                'total': int(x.sum()),
                'per_class': x.sum(0).tolist()},
            'image_stats': {
                'total': dataset.n,
                'unlabelled': int(np.all(x == 0, 1).sum()),
                'per_class': (x > 0).sum(0).tolist()},
            'labels': [{
                str(Path(k).name): _round_labels(v.tolist())} for k, v in zip(dataset.im_files, dataset.labels)]}

        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(_hub_ops, dataset.im_files), total=dataset.n, desc='HUB Ops'):
                pass

    # Profile
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file) as f:
                x = json.load(f)  # load hyps dict
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

    # Save, print and return
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)  # save stats.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats


if __name__ == '__main__':
    from utils.general import check_yaml

    hyp = check_yaml('E:\裂缝\yolo\myolov5\data\hyps\hyp.scratch-high.yaml', suffix=('.yaml', '.yml'))
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    a = LoadImagesAndLabels(path=r'E:\裂缝\yolo\datasets\coco128\images\train2017',
                            img_size=640,
                            batch_size=2,
                            augment=True,
                            hyp=hyp,
                            rect=True,  # 当augment=True时，设置rect为True即可不使用mosaic数据增强，self.rect为True表明每批batch数据需要自适应缩放尺寸
                            image_weights=False,
                            cache_images=None,
                            single_cls=False,
                            stride=32,
                            pad=0,
                            prefix='train')
    cnm = []
    for i in range(10):
        cnm.append(a.__getitem__(0))
    LoadImagesAndLabels.collate_fn(cnm)
