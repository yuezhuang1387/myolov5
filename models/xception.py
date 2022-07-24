from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

#去除了maxpooling
class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        rep = []

        filters = in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        #if strides != 1:
            #rep.append(DeformConv2d(out_filters, out_filters, 3, 1, 1, modulation=True))
            #rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        #x += skip
        return x,skip


class BCcls(nn.Module):

    def __init__(self, in_channels, out_channels,num_classes, **kwargs):
        super(BCcls, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.fc=nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x=self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        cls_feat = F.normalize(x, p=2, dim=1)
        x = self.fc(x)
        return x,cls_feat

class BCft(nn.Module):

    def __init__(self, in_channels, out_channels,num_classes, **kwargs):
        super(BCft, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.fc=nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x=self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        ft_feat = F.normalize(x, p=2, dim=1)
        x = self.fc(x)
        return x,ft_feat

class BCseg(nn.Module):

    def __init__(self, low_channels, high_channels,num_classes, **kwargs):
        super(BCseg, self).__init__()
        #self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(low_channels, low_channels, kernel_size=(1,1), **kwargs)
        self.conv2 = nn.Conv2d(high_channels, high_channels, kernel_size=(1, 1), **kwargs)

        self.conv3 = nn.Conv2d(low_channels+high_channels, num_classes, kernel_size=(3, 3),padding=1, **kwargs)
        self.bn3 = nn.BatchNorm2d(num_classes)
        self.relu3 = nn.ReLU(inplace=True)
        #self.fc=nn.Linear(num_classes*28*28, num_classes)

    def forward(self, xseglow,xseghigh):
        H=xseglow.size()[2]
        xseglow = self.conv1(xseglow)

        xseghigh=self.conv2(xseghigh)
        xseghigh=F.upsample_bilinear(xseghigh,size=(H, H))
        x=torch.cat((xseglow,xseghigh),1)

        x=self.conv3(x)
        x=self.bn3(x)
        x=self.relu3(x)
        return x

class Xception(nn.Module):

    def __init__(self, num_classes_cls=1000,num_classes_ft=2):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes_cls = num_classes_cls
        self.num_classes_ft=num_classes_ft

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.max1=nn.MaxPool2d(3, 2, 1)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.max2 = nn.MaxPool2d(3, 2, 1)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.max3 = nn.MaxPool2d(3, 2, 1)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.max4 = nn.MaxPool2d(3, 2, 1)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        #self.relu4=nn.ReLU(inplace=True)
        self.cls = BCcls(in_channels=2048, out_channels=2048, num_classes=self.num_classes_cls)
        self.ft = BCft(in_channels=2048, out_channels=2048, num_classes=self.num_classes_ft)
        self.seg = BCseg(low_channels=728, high_channels=2048,num_classes=self.num_classes_ft)




        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x,skip = self.block1(x)
        x=self.max1(x)
        x=x+skip

        x,skip = self.block2(x)
        x=self.max2(x)
        x=x+skip

        x,skip = self.block3(x)
        xseglow=x
        x=self.max3(x)
        x=x+skip

        x ,skip= self.block4(x)
        x=x+skip
        x ,skip= self.block5(x)
        x = x + skip
        x ,skip= self.block6(x)
        x = x + skip
        x ,skip= self.block7(x)
        x = x + skip
        #x = self.block8(x)
        #x = self.block9(x)
        #x = self.block10(x)
        #x = self.block11(x)
        x,skip = self.block12(x)
        x=self.max4(x)
        x=x+skip


        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        xseghigh = x
        xsrc = x

        return xseglow,xseghigh,xsrc

    def forward(self, input):
        xseglow, xseghigh, xsrc= self.features(input)
        y_cls,cls_feat= self.cls(xsrc)
        y_ft ,ft_feat= self.ft(xsrc)
        y_seg = self.seg(xseglow, xseghigh)

        return y_cls,y_ft,y_seg,cls_feat,ft_feat
if __name__ == '__main__':
    model = Xception(num_classes_cls=11,num_classes_ft=2)
    x = torch.rand(1,3,224,224)
    y = model(x)
    for yy in y:
        print(yy.shape)
        # torch.Size([1, 11])
        # torch.Size([1, 2])
        # torch.Size([1, 2, 28, 28])
        # torch.Size([1, 2048])
        # torch.Size([1, 2048])