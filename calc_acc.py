import os
import json
import math
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import GAHRNet
from torchvision import transforms
from PIL import Image, ImageDraw
import datetime
from my_dataset import DataKeyPoint5,DataKeyPoint2,DataKeyPoint22,DataKeyPoint222
from my_dataset import DataKeyPoint4
from torch.utils import data
import distributed_utils as utils


def euclidean_distance(tensor1, tensor2, dim, device):
   
    # 计算差值的平方
    squared_difference = (tensor1.to(device) - tensor2.to(device))**2
    
    # 沿着指定的维度求和
    sum_along_dim = squared_difference.sum(dim)
    
    # 取平方根
    distance = torch.sqrt(sum_along_dim)
    
    return distance


def get_max_preds(batch_heatmaps):

    batch_size, num_joints, h, w = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    maxvals = maxvals.unsqueeze(dim=-1)
    idx = idx.float()

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps)

    preds[:, :, 0] = idx % w  # column 对应最大值的x坐标
    preds[:, :, 1] = torch.floor(idx / w)  # row 对应最大值的y坐标

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    preds *= pred_mask
    return preds 

device = torch.device("cuda:1")
print(f"using device: {device}")
weights_path = '/home/jxzhu/MBSI/new_weight/SwinTransformerSys/model-9.58057373046875.pth' #convsthrnet
val_dataset = DataKeyPoint222('/home/jxzhu/MBSI/data', "test2", fixed_size=[512, 512])
val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      pin_memory=True)




model = GAHRNet(num_joints=29,base_channel=32)
# model = SwinTransformerSys(img_size=512, window_size=8, num_classes=29, in_chans=3)
weights = torch.load(weights_path, map_location=device)
weights = weights if "model" not in weights else weights["model"]
model.load_state_dict(weights)
model.to(device)
model.eval()
L = []
SD = []
L1 = []
L2 = []
L3 = []
L4 = []
SD1 = []
for i in range(19):
    L1.append(0)
    L2.append(0)
    L3.append(0)
    L4.append(0)
    SD1.append(0)
metric_logger = utils.MetricLogger(delimiter="  ")
for i, [images, targets, points, height, width] in enumerate(metric_logger.log_every(val_data_loader,80)):
    images = torch.stack([image.to(device) for image in images])
    results = model(images)

    point1 = get_max_preds(results)[0].to(device)
    # point2 = get_max_preds(targets)[0].to(device)
    y = torch.tensor([1935 / 3360,2400 / 3360]).to(device)
    point1 = point1 * y
    point2 = points[0].to(device)
    point2 = point2 * y
    # point2 = point2 * y
    # print(point1.size())[29,2]
    # print(point2.size())[29,2]
    y = euclidean_distance(point1,point2,1,device)  
    y2 = y.sum(dim=0) / 19.0
    y3 = (y-y2) ** 2
    y3 = math.sqrt(y3.sum(dim=0) / 19.0)
    L.append(y2)
    SD.append(y3)
             
    # print('y=',y.size())
    y = y.tolist()
    for i in range(len(y)):
        SD1.append(math.sqrt((i-y2) ** 2))
        if y[i] < 4.0:
            L1[i] += 1.0
            if y[i] < 3.0:
                L2[i] += 1.0
                if y[i] < 2.5:
                    L3[i] += 1.0
                    if y[i] < 2.0:
                        L4[i] += 1.0
l = 0
sd = 0
for j in SD:
    sd += j
for i in L:
    l += i
xm = 100.0
print('SD=',sd / xm)
print('SDR=',l.item() / xm)
l2,l3,l4,l5 = 0, 0, 0, 0

for i in range(19):
    l2 += L1[i] / xm
    l3 += L2[i] / xm
    l4 += L3[i] / xm
    l5 += L4[i] / xm
    print('SRD<4.0的',i,'号点正确率为：',L1[i] / xm)
    print('SRD<3.0的',i,'号点正确率为：',L2[i] / xm)
    print('SRD<2.5的',i,'号点正确率为：',L3[i] / xm)
    print('SRD<2.0的',i,'号点正确率为：',L4[i] / xm)

print('SRD<2.0的的平均正确率为：',l2 / 19)
print('SRD<2.5的的平均正确率为：',l3 / 19)
print('SRD<3.0的的平均正确率为：',l4 / 19)
print('SRD<4.0的的平均正确率为：',l5 / 19)


