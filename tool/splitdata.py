import os
import cv2
import numpy as np
from tqdm import tqdm
'''
为什么在此处的opencv无法读取中文路径，在detect函数中的就可读取中文路径？？？
将图片划分为4:1的训练和验证数据
'''
if __name__ == '__main__':
    path = r'G:\Turingdataset\交趾黄檀手串_cut_7_4'
    save = r'G:\Turingdataset\cut_7_4'
    ratio = 5
    trainpath = save + r'\train'
    valpath = save + r'\val'
    if not os.path.exists(trainpath):
        os.makedirs(trainpath)
    if not os.path.exists(valpath):
        os.makedirs(valpath)
    print(len(os.listdir(path)))

    for i, imgname in enumerate(tqdm(os.listdir(path))):
        imgfile = os.path.join(path, imgname)
        # img = cv2.imread(imgfile)
        img = cv2.imdecode(np.fromfile(imgfile,dtype=np.uint8),-1)
        if i%ratio==0:
            cv2.imwrite(os.path.join(valpath,imgname),img)
        else:
            cv2.imwrite(os.path.join(trainpath, imgname), img)
