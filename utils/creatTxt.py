import sys
import os.path as osp
import os
import json
from tqdm import tqdm
import cv2
import numpy as np


def json2labelTxt():
    '''
    将json格式的labels转为
    '''
    img_dir = r'G:\Turingdataset\bv\annotations'
    label2Num = {'刻字锁': 0, '刻字铆钉': 1, '刻字搭扣': 2, 'Logo压印': 3, '刻字链接块': 4, '锁扣头': 5, '白标': 6, '刻字拉链头': 7, '刻字': 8,
                 '刻印皮标': 9, '拉链头': 10, '数字卡片': 11}
    label_set = set()
    type_set = set()
    nums = 0  # 总共8151张图（jpg格式，每张图至少一个框）, 8791个标签框
    imgnum = 0
    for dir, sub_dirs, files in os.walk(img_dir):
        if len(files) > 0:
            for file in files:
                if file.split('.')[-1].lower() not in ['json']:
                    print(f'不是json文件')
                    continue
                nums += 1
                json_path = os.path.join(dir, file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    dict_list = json.load(f)  # [{'label': 'Logo压印', 'type': 'rect', 'points': [287, 400, 417, 423]}]
                img = cv2.imdecode(
                    np.fromfile(json_path.replace('annotations', 'images').replace('.json', '.jpg'), dtype=np.uint8),
                    1)  # 可读取中文路径图片
                H, W, C = img.shape
                txt_path = json_path.replace('annotations', 'labels').replace('.json', '.txt')
                txt_dir = str(os.sep).join(txt_path.split(os.sep)[:-1])
                if not os.path.exists(txt_dir):
                    os.makedirs(txt_dir)
                for data in dict_list:
                    label_num = label2Num[data['label']]
                    x1, y1, x2, y2 = data['points']  # (x1,y1,x2,y2)
                    x_center = (x1 + x2) / 2 / W
                    y_center = (y1 + y2) / 2 / H
                    w = (x2 - x1) / W
                    h = (y2 - y1) / H
                    newbox = (x_center, y_center, w, h)
                    with open(txt_path, 'a', encoding='utf-8') as f:
                        f.write(
                            str(label_num) + ' ' + str(round(x_center, 6)) + ' ' + str(round(y_center, 6)) + ' ' + str(
                                round(w, 6)) + ' ' + str(round(h, 6)) + ' ' + '\n')  # score中包含了换行字符
                    # cv2.rectangle(img, (int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])), thickness=2, color=(255, 255, 0))
                    # rect函数的输入为（左上角x，左上角y，宽，高）
                # cv2.imshow('cnm',cv2.resize(img,(640,640),interpolation=cv2.INTER_LINEAR))
                # cv2.waitKey()
                # print(label_set)
                # print(type_set)
                # return 0
    print(nums)
    print(imgnum)


def readfromtxt():
    img_dir = r'G:\Turingdataset\bv\labels'
    label2Num = {'刻字锁': 0, '刻字铆钉': 1, '刻字搭扣': 2, 'Logo压印': 3, '刻字链接块': 4, '锁扣头': 5, '白标': 6, '刻字拉链头': 7, '刻字': 8,
                 '刻印皮标': 9, '拉链头': 10, '数字卡片': 11}
    label_set = set()
    type_set = set()
    nums = 0  # 总共8151张图（jpg格式，每张图至少一个框）, 8791个标签框
    imgnum = 0
    for dir, sub_dirs, files in os.walk(img_dir):
        if len(files) > 0:
            for file in files:
                if file.split('.')[-1].lower() not in ['txt']:
                    print(f'不是txt文件')
                    continue
                nums += 1
                txt_path = os.path.join(dir, file)
                with open(txt_path, 'r', encoding='utf-8') as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    # lb: [['45', '0.479492', '0.688771', '0.955609', '0.5955'],
                    #      ['45', '0.736516', '0.247188', '0.498875', '0.476417'],
                    #      ['50', '0.637063', '0.732938', '0.494125', '0.510583']]
                img = cv2.imdecode(
                    np.fromfile(txt_path.replace('labels', 'images').replace('.txt', '.jpg'), dtype=np.uint8),
                    1)  # 可读取中文路径图片
                H, W, C = img.shape
                lb = np.array(lb, dtype=np.float32)
                # print(lb)
                for c, xc, yc, w, h in lb:
                    # xc,yc,w,h = lb[1][1],lb[1][2],lb[1][3],lb[1][4]
                    x1, y1, x2, y2 = (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H
                    # print(f'{x1} {y1} {x2} {y2}')
                    cv2.rectangle(img, (int(x1), int(y1), int(w * W), int(h * H)), thickness=2, color=(255, 255, 0))
                    # rect函数的输入为（左上角x，左上角y，宽，高）
                cv2.imshow('cnm', cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR))
                cv2.waitKey()
                # print(label_set)
                # print(type_set)
                # return 0
    print(nums)
    print(imgnum)


def splitTrainAndVal(ratio=5):
    '''
    将train和val划分为5:1
    :param ratio:
    :return:
    '''
    img_dir = r'G:\Turingdataset\bv\labels'
    label2Num = {'刻字锁': 0, '刻字铆钉': 1, '刻字搭扣': 2, 'Logo压印': 3, '刻字链接块': 4, '锁扣头': 5, '白标': 6, '刻字拉链头': 7, '刻字': 8,
                 '刻印皮标': 9, '拉链头': 10, '数字卡片': 11}
    label_set = set()
    type_set = set()
    nums = 0  # 总共8151张图（jpg格式，每张图至少一个框）, 8791个标签框
    nums1 = 0
    nums2 = 0
    for dir, sub_dirs, files in os.walk(img_dir):
        if len(files) > 0:
            for file in files:
                if file.split('.')[-1].lower() not in ['txt']:
                    print(f'不是txt文件')
                    continue
                txt_path = os.path.join(dir, file)
                img_path = txt_path.replace('labels', 'images').replace('.txt', '.jpg')
                # print(img_path) # G:\Turingdataset\bv\images\aaahandbags\false\Bottega Veneta\Bottega Veneta Arco 48 Bag Grainy Calfskin (Varied Colors)_262\1718637.jpg
                img_path = str(os.sep).join(img_path.split(os.sep)[3:])
                # print(img_path) # images\aaahandbags\false\Bottega Veneta\Bottega Veneta Arco 48 Bag Grainy Calfskin (Varied Colors)_262\1718637.jpg
                if nums%(ratio+1):
                    with open('G:\Turingdataset/bv/train.txt', 'a', encoding='utf-8') as f:
                        f.write('./' + img_path + '\n')  # score中包含了换行字符
                    nums1 += 1
                else:
                    with open('G:\Turingdataset/bv/val.txt', 'a', encoding='utf-8') as f:
                        f.write('./' + img_path + '\n')  # score中包含了换行字符
                    nums2 += 1
                nums += 1
                # return 0
    print(nums1)
    print(nums2)


if __name__ == '__main__':
    splitTrainAndVal()
