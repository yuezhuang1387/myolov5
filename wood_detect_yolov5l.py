import argparse
import os
import torch.backends.cudnn as cudnn

# from utils.datasets import *
from utils.torch_utils import *
from utils.general import *
from models.yolo import Model
from models.experimental import attempt_load
import pdb
from tqdm import tqdm


def time_synchronized():
    # 基于pytorch的精准时间测量
    if torch.cuda.is_available():  # 如果cuda可用则执行synchronize函数
        # 该函数会等待当前设备上的流中的所有核心全部完成 这样测量时间便会准确 因为pytorch中程序为异步执行
        torch.cuda.synchronize()
    return time.time()


def delete_catelog(catelog):
    if os.path.exists(catelog):
        shutil.rmtree(catelog)
    else:
        pass


def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class yolov5_v4_DetectionModel:
    def __init__(self, weights=None, cfg_file='./models/yolov5l_turing.yaml', img_size=640, iou_thres=0.5, augment=True,
                 agnostic_nms=False,
                 names=None, device='0'):
        if names is None:
            names = ['Hediao', 'Zhuchuan', 'Hetao', 'Shouchuanxijie']
        self.img_size = img_size
        self.iou_thres = iou_thres
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.device = select_device(device)

        print(self.device)
        t0 = time.time()
        model = Model(cfg_file, nc=len(names))
        # ckpt = torch.load(weights, map_location=device)
        ckpt = torch.load(weights)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        model.to(self.device)
        model.eval()
        print('Loading time: ', time.time() - t0)
        self.model = model
        self.names = names  # model.names if hasattr(model, 'names') else model.modules.names
        self.classes = len(self.names)
        self.prepare()

    def prepare(self):
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        ret = self.model(img) if self.device.type != 'cpu' else None
        return ret

    def predict(self, image, threshold):
        if isinstance(image, str):
            im0 = cv2.imread(image)  # BGR
        else:
            im0 = image
        assert im0 is not None, 'Image Not Found ' + image
        # Padded resize
        img = letterbox(im0, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        # pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        pred = non_max_suppression(pred, threshold, self.iou_thres, agnostic=self.agnostic_nms)
        # pred为当前batchsize全部图像筛完后的预测框，len(pred)=batchsize
        #              pred[i].shape：torch.Size([当前图像nms最终筛完的预测框数量(不超过300),6])
        #              6中0:4表示预测框坐标(x1, y1, x2, y2)-均为实际尺寸坐标(映射到yolo模型实际输入图像尺寸上(640,640)或(672,另一个可被32整除)
        #              6中4表示当前的预测概率值
        #              6中5表示当前的预测类别(0~79)
        t2 = time_synchronized()
        det = pred[0]
        # Process detections
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
        results = []
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                results.append({
                    'class_id': int(cls),
                    'label': self.names[int(cls)],
                    'score': float(conf),
                    'ymin': int(xyxy[1]),
                    'xmin': int(xyxy[0]),
                    'ymax': int(xyxy[3]),
                    'xmax': int(xyxy[2]),
                })

        return results


def det_box(res):
    return res['ymin'], res['xmin'], res['ymax'], res['xmax']


def get_save_path(cut_bag_dir, label, img_file):
    result_list = ['HeTao', 'GanLanHe', 'PuTi', 'HaiNanHuangHuaLi', 'XiaoYeZiTan']
    img_id = img_file.split('\\')[-1].split('.')[0]

    if img_file.split('\\')[-5] in result_list:
        category = img_file.split('\\')[-5]
        img_name = category + '__' + img_file.split('\\')[-4] + '__' + img_file.split('\\')[-3] + '__' + img_id

    if img_file.split('\\')[-4] in result_list:
        category = img_file.split('\\')[-4]
        img_name = category + '__' + img_file.split('\\')[-3] + '__' + img_file.split('\\')[-2] + '__' + img_id
    if img_file.split('\\')[-3] in result_list:
        category = img_file.split('\\')[-3]
        img_name = category + '__' + img_file.split('\\')[-2] + '__' + img_id
    if img_file.split('\\')[-2] in result_list:
        category = img_file.split('\\')[-2]
        img_name = category + '__' + img_id

    save_path = os.path.join(cut_bag_dir, category)
    return save_path, img_name


if __name__ == '__main__':
    model = yolov5_v4_DetectionModel(weights='./models/checkpoints/wood_detect_update1.pt',
                                     device='0',
                                     names=['Hediao', 'Zhuchuan', 'Hetao', 'Shouchuanxijie']
                                     )

    img_dir = r'G:\Turingdataset\鸡翅木菩提\菩提原图'
    # cut_dir = img_dir+'_cut_7_6'
    cut_dir = 'G:\Turingdataset\鸡翅木菩提\菩提'
    realname = '菩提'
    i = 0

    for dir, sub_dirs, files in os.walk(img_dir):
        folder = os.path.split(dir)[-1]
        if len(files) > 0:
            i = i + 1
            for file in tqdm(files):
                if file.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png']:
                    print(file)
                    continue
                img_file = os.path.join(dir, file)
                try:
                    img = cv2.imread(img_file)
                    results = model.predict(img_file, 0.3)
                except:
                    continue
                file = file.replace('.png', '.jpg')
                if len(results) == 0:
                    continue
                results = sorted(results, key=lambda x: x["score"], reverse=True)
                img_h = img.shape[0]
                res = results[0]
                # print(res)
                ymin, xmin, ymax, xmax = det_box(res)
                label, score = res['label'], res['score']
                re_img = img[ymin:ymax, xmin:xmax, :]
                # save_path = get_save_path(cut_bag_dir, label, img_file)

                # try:
                # save_path = dir.replace(img_dir,cut_dir) # 保存1：按照原始路径存
                # save_path = os.path.join(cut_dir,label) # 保存2：按照检测输出的类别文件夹保存
                save_path = cut_dir # 保存3：将全部图片存到一个文件夹中
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                write_name = os.path.join(save_path, realname+'___'+'___'.join(dir.split(os.sep)[len(img_dir.split(os.sep)):])+'___'+''.join(file.split('.')[:-1]) + '.jpg')
                if not os.path.exists(write_name):
                    cv2.imwrite(write_name, re_img)
            # except:
            #     print('------------------------------------------------')
