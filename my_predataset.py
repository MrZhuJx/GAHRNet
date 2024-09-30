import os
import json
import numpy as np
import math
import cv2
import PIL
import torch

from torchvision import transforms, datasets, models
from PIL import Image, ImageFont, ImageDraw


def json_to_numpy(dataset_path,dataset_path2,new_height,new_width,h,w):
    coordinates = []
    coordinates2 = []
    res = []
    res2 = []
    with open(dataset_path) as fp:
         for line in fp:
            coordinates.append(line.strip())
    for i in range(19):
        res.append(coordinates[i].split(','))
        res[i][0] = int(res[i][0])
        res[i][1] = int(res[i][1])
    with open(dataset_path2) as fp:
         for line in fp:
            coordinates2.append(line.strip())
    for i in range(19):
        res2.append(coordinates2[i].split(','))
        res2[i][0] = int(res2[i][0])
        res2[i][1] = int(res2[i][1])

    landmarks = []
    landmarks = (np.array(res) + np.array(res2)) // 2

    landmarks = landmarks.reshape(-1,2)   
    for points in landmarks:
        points[1] = int(new_height * points[1] / h)
        points[0] = int(new_width * points[0] / w)
            

    return landmarks

def json_to_numpy2(dataset_path,new_height,new_width):
    with open(dataset_path) as fp: # 使用`with`语句打开文件，这样可以确保文件在使用后会被正确关闭。`dataset_path`是传入的文件路径。
        json_data = json.load(fp) # 读取文件内容，并将JSON格式的字符串解析为Python字典。
        points = json_data['shapes'] # 从字典中获取键为`'shapes'`的值，通常这个值包含图像中的关键点信息。
        h, w = json_data['imageHeight'], json_data['imageWidth'] # 获取原始图像的高度和宽度。

    landmarks = [] # 初始化两个空列表和字典，用于存储关键点信息。
    landmarks_dic = {} 
    for point in points: # 循环遍历每个关键点信息。
        for p in point['points']: # 因为每个关键点可能包含多个坐标点，所以再次遍历。
            landmarks_dic[point['label']] = p # 将关键点的标签和坐标存储到字典中。

    for i in range(1,30): # 假设关键点的标签是从1到29，遍历这些标签。
        landmarks.append(landmarks_dic[str(i)])  # 将字典中的关键点坐标添加到`landmarks`列表中。

    landmarks = np.array(landmarks) # 将列表转换为NumPy数组。

    landmarks = landmarks.reshape(-1,2)   # 确保关键点的坐标是2维的（即每个关键点由x和y坐标组成）。
    for points in landmarks: # 遍历每个关键点坐标。
        points[1] = int(new_height * points[1] / h) # 根据新的图像尺寸调整关键点的坐标。
        points[0] = int(new_width * points[0] / w)
            

    return landmarks # 返回调整后的关键点坐标数组。

def generate_heatmaps(landmarks,height,width,sigma,new_height,new_width):
    heatmaps = []
    for points in landmarks:
        heatmap = np.zeros((new_height, new_width))
        ch = int(points[1])
        cw = int(points[0])
        heatmap[ch][cw] = 1

        heatmap = cv2.GaussianBlur(heatmap,sigma,0)
        am = np.amax(heatmap)
        heatmap /= am / 255
        heatmaps.append(heatmap)

    heatmaps = np.array(heatmaps)    
    return heatmaps