import torch
import os
import numpy as np
import torch.nn.functional as F
import torch.utils.data
# import cv2
import torchvision

from torch import nn
from torchvision import datasets,transforms,models
from PIL import Image,ImageFont,ImageDraw
from my_predataset import json_to_numpy, generate_heatmaps,json_to_numpy2

    
class DataKeyPoint2(torch.utils.data.Dataset):

    def __init__(self,datasets_path,typee,fixed_size):
        super().__init__()
        self.datasets_path = datasets_path
        self.files_name_list = os.listdir(os.path.join(datasets_path,typee))
        self.img_list = []
        self.json_list = []
        self.json_list2 = []
        self.target = fixed_size

        for i in range(len(self.files_name_list)):
            if '.jpg' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                self.json_list.append(self.files_name_list[i].replace('.jpg','.txt'))
                self.json_list2.append(self.files_name_list[i].replace('.jpg','1.txt'))
            if '.bmp' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                self.json_list.append(self.files_name_list[i].replace('.bmp','.txt'))
                self.json_list2.append(self.files_name_list[i].replace('.bmp','1.txt'))

        for i in range(len(self.img_list)):
            self.img_list[i] = os.path.join(os.path.join(datasets_path,typee),self.img_list[i])

        for i in range(len(self.json_list)):
            self.json_list[i] = os.path.join(os.path.join(datasets_path,typee),self.json_list[i])

        for i in range(len(self.json_list2)):
            self.json_list2[i] = os.path.join(os.path.join(datasets_path,typee),self.json_list2[i])


    def __len__(self):
        return len(self.img_list)        


    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_name = self.img_list[index][:-4]
        save_name = self.img_list[index].split('/')[-1]
        img = Image.open(img_path)
        height,width = np.shape(img)[0], np.shape(img)[1]
        

        json_path = self.json_list[index]
        json_path2 = self.json_list2[index]
        landmarks = json_to_numpy(json_path,json_path2,new_height=168,new_width=168,h=height,w=width)
        points = torch.tensor(landmarks)

        heatmaps = generate_heatmaps(landmarks,height,width,(21,21),
                                     new_height=168,new_width=168)
        
        heatmaps = torch.tensor(heatmaps,dtype=torch.float32)


        

        transform = transforms.Compose([
            transforms.Resize((672,672)),
            transforms.ToTensor()
        ])
        img = transform(img)

       

        return img, heatmaps, points

class DataKeyPoint4(torch.utils.data.Dataset):

    def __init__(self,datasets_path,typee,fixed_size):
        super().__init__()
        self.datasets_path = datasets_path
        self.files_name_list = os.listdir(os.path.join(datasets_path,typee))
        self.img_list = []
        self.json_list = []
        self.json_list2 = []
        self.target = fixed_size

        for i in range(len(self.files_name_list)):
            if '.jpg' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                self.json_list.append(self.files_name_list[i].replace('.jpg','.txt'))
                self.json_list2.append(self.files_name_list[i].replace('.jpg','1.txt'))
            if '.bmp' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                self.json_list.append(self.files_name_list[i].replace('.bmp','.txt'))
                self.json_list2.append(self.files_name_list[i].replace('.bmp','1.txt'))

        for i in range(len(self.img_list)):
            self.img_list[i] = os.path.join(os.path.join(datasets_path,typee),self.img_list[i])

        for i in range(len(self.json_list)):
            self.json_list[i] = os.path.join(os.path.join(datasets_path,typee),self.json_list[i])

        for i in range(len(self.json_list2)):
            self.json_list2[i] = os.path.join(os.path.join(datasets_path,typee),self.json_list2[i])


    def __len__(self):
        return len(self.img_list)        


    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_name = self.img_list[index][:-4]
        save_name = self.img_list[index].split('/')[-1]
        img = Image.open(img_path)
        height,width = np.shape(img)[0], np.shape(img)[1]
        

        json_path = self.json_list[index]
        json_path2 = self.json_list2[index]
        landmarks = json_to_numpy(json_path,json_path2,new_height=256,new_width=256,h=height,w=width)
        points = torch.tensor(landmarks)

        heatmaps = generate_heatmaps(landmarks,height,width,(21,21),
                                     new_height=256,new_width=256)
        
        heatmaps = torch.tensor(heatmaps,dtype=torch.float32)


        

        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
        img = transform(img)

       

        return img, heatmaps, points

class DataKeyPoint5(torch.utils.data.Dataset):

    def __init__(self,datasets_path,typee,fixed_size):
        super().__init__()
        self.datasets_path = datasets_path
        self.files_name_list = os.listdir(os.path.join(datasets_path,typee))
        self.img_list = []
        self.json_list = []
        self.target = fixed_size

        for i in range(len(self.files_name_list)):
            if '.jpg' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                self.json_list.append(self.files_name_list[i].replace('.jpg','.json'))
            if '.bmp' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                self.json_list.append(self.files_name_list[i].replace('.bmp','.json'))

        for i in range(len(self.img_list)):
            self.img_list[i] = os.path.join(os.path.join(datasets_path,typee),self.img_list[i])

        for i in range(len(self.json_list)):
            self.json_list[i] = os.path.join(os.path.join(datasets_path,typee),self.json_list[i])


    def __len__(self):
        return len(self.img_list)        


    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_name = self.img_list[index][:-4]
        save_name = self.img_list[index].split('/')[-1]
        img = Image.open(img_path)
        height,width = np.shape(img)[0], np.shape(img)[1]
        

        json_path = self.json_list[index]
        landmarks = json_to_numpy2(json_path,new_height=336,new_width=336)
        points = torch.tensor(landmarks)

        heatmaps = generate_heatmaps(landmarks,height,width,(17,17),
                                     new_height=336,new_width=336)
        
        heatmaps = torch.tensor(heatmaps,dtype=torch.float32)


        

        transform = transforms.Compose([
            transforms.Resize((672,672)),
            transforms.ToTensor()
        ])
        img = transform(img)

       

        return img, heatmaps, points, height, width
    
class DataKeyPoint55(torch.utils.data.Dataset):

    def __init__(self,datasets_path,typee,fixed_size):
        super().__init__()
        self.datasets_path = datasets_path
        self.files_name_list = os.listdir(os.path.join(datasets_path,typee))
        self.img_list = []
        self.json_list = []
        self.target = fixed_size

        for i in range(len(self.files_name_list)):
            if '.jpg' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                self.json_list.append(self.files_name_list[i].replace('.jpg','.json'))
            if '.bmp' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                self.json_list.append(self.files_name_list[i].replace('.bmp','.json'))

        for i in range(len(self.img_list)):
            self.img_list[i] = os.path.join(os.path.join(datasets_path,typee),self.img_list[i])

        for i in range(len(self.json_list)):
            self.json_list[i] = os.path.join(os.path.join(datasets_path,typee),self.json_list[i])


    def __len__(self):
        return len(self.img_list)        


    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_name = self.img_list[index][:-4]
        save_name = self.img_list[index].split('/')[-1]
        img = Image.open(img_path)
        height,width = np.shape(img)[0], np.shape(img)[1]
        

        json_path = self.json_list[index]
        landmarks = json_to_numpy2(json_path,new_height=672,new_width=672)
        points = torch.tensor(landmarks)

        heatmaps = generate_heatmaps(landmarks,height,width,(41,41),
                                     new_height=672,new_width=672)
        
        heatmaps = torch.tensor(heatmaps,dtype=torch.float32)


        

        transform = transforms.Compose([
            transforms.Resize((672,672)),
            transforms.ToTensor()
        ])
        img = transform(img)

       

        return img, heatmaps, points, height, width

def hflip_landmarks(landmarks_ori,width):
    landmarks = []
    for points in landmarks_ori:
        ph = points[1]
        pw = width - points[0]
        landmarks.append([pw, ph])
    landmarks = np.array(landmarks)
    landmarks = landmarks.reshape(-1, 2)
    return landmarks

class DataKeyPoint1(torch.utils.data.Dataset):

    def __init__(self,datasets_path,typee,fixed_size):
        super().__init__()
        self.datasets_path = datasets_path
        self.files_name_list = os.listdir(os.path.join(datasets_path,typee))
        self.img_list = []
        self.json_list = []
        self.json_list1 = []
        self.target = fixed_size

        for i in range(len(self.files_name_list)):
            if '.jpg' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
            if '.bmp' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                # self.json_list.append(self.files_name_list[i].replace('.bmp','.json'))

        for i in range(len(self.img_list)):
            self.img_list[i] = os.path.join(os.path.join(datasets_path,typee),self.img_list[i])

        # for i in range(len(self.json_list)):
        #     self.json_list[i] = os.path.join(os.path.join(datasets_path,typee),self.json_list[i])


    def __len__(self):
        return len(self.img_list)        


    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_path2 = self.img_list[index]
        transform = self.img_list[index].split('/')[-1].split('.')[-2]
        img = Image.open(img_path)
        height,width = np.shape(img)[0], np.shape(img)[1]
        
        if not transform.find('hflip'):
            json = img_path
            json_path = json.replace('.hflip.bmp', '.txt')
            json_path2 = json.replace('.hflip.bmp', '1.txt')
            
            landmarks_ori = json_to_numpy(json_path,json_path2,new_height=168,new_width=168,h=height,w=width)
            landmarks = hflip_landmarks(landmarks_ori,168)
            points = torch.tensor(landmarks)
        elif not transform.find('gaussin'):
            json = img_path
            kernel = transform.split('_')[-2]
            sigma = transform.split('_')[-1]
            json_path = json.replace('.gaussin_{}_{}.bmp'.format(kernel, sigma), '.txt')
            json_path2 = json.replace('.gaussin_{}_{}.bmp'.format(kernel, sigma), '1.txt')
            landmarks = json_to_numpy(json_path,json_path2,new_height=168,new_width=168,h=height,w=width)
            points = torch.tensor(landmarks)
        else:
            json = img_path
            if '.jpg' in json:
                json_path = json.replace('.jpg', '.txt')
                json_path2 = json.replace('.jpg', '1.txt')
            if '.bmp' in json:
                json_path = json.replace('.bmp', '.txt')
                json_path2 = json.replace('.bmp', '1.txt')
             
            landmarks = json_to_numpy(json_path,json_path2,new_height=168,new_width=168,h=height,w=width)
            points = torch.tensor(landmarks)

        heatmaps = generate_heatmaps(landmarks,height,width,(21,21),
                                     new_height=168,new_width=168)
        
        heatmaps = torch.tensor(heatmaps,dtype=torch.float32)


        transform = transforms.Compose([
                transforms.Resize((672,672)),
                transforms.ToTensor()
            ])
        img = transform(img)

        

        return img, heatmaps,points

class DataKeyPoint22(torch.utils.data.Dataset):

    def __init__(self,datasets_path,typee,fixed_size):
        super().__init__()
        self.datasets_path = datasets_path
        self.files_name_list = os.listdir(os.path.join(datasets_path,typee))
        self.img_list = []
        self.json_list = []
        self.json_list2 = []
        self.target = fixed_size

        for i in range(len(self.files_name_list)):
            if '.jpg' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                self.json_list.append(self.files_name_list[i].replace('.jpg','.txt'))
                self.json_list2.append(self.files_name_list[i].replace('.jpg','1.txt'))
            if '.bmp' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                self.json_list.append(self.files_name_list[i].replace('.bmp','.txt'))
                self.json_list2.append(self.files_name_list[i].replace('.bmp','1.txt'))

        for i in range(len(self.img_list)):
            self.img_list[i] = os.path.join(os.path.join(datasets_path,typee),self.img_list[i])

        for i in range(len(self.json_list)):
            self.json_list[i] = os.path.join(os.path.join(datasets_path,typee),self.json_list[i])

        for i in range(len(self.json_list2)):
            self.json_list2[i] = os.path.join(os.path.join(datasets_path,typee),self.json_list2[i])


    def __len__(self):
        return len(self.img_list)        


    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_name = self.img_list[index][:-4]
        save_name = self.img_list[index].split('/')[-1]
        img = Image.open(img_path)
        height,width = np.shape(img)[0], np.shape(img)[1]
        

        json_path = self.json_list[index]
        json_path2 = self.json_list2[index]
        landmarks = json_to_numpy(json_path,json_path2,new_height=336,new_width=336,h=height,w=width)
        points = torch.tensor(landmarks)

        heatmaps = generate_heatmaps(landmarks,height,width,(41,41),
                                     new_height=336,new_width=336)
        
        heatmaps = torch.tensor(heatmaps,dtype=torch.float32)
        

        transform = transforms.Compose([
            transforms.Resize((672,672)),
            transforms.ToTensor()
        ])
        img = transform(img)

       

        return img, heatmaps, points, height, width

class DataKeyPoint222(torch.utils.data.Dataset):

    def __init__(self,datasets_path,typee,fixed_size):
        super().__init__()
        self.datasets_path = datasets_path
        self.files_name_list = os.listdir(os.path.join(datasets_path,typee))
        self.img_list = []
        self.json_list = []
        self.json_list2 = []
        self.target = fixed_size

        for i in range(len(self.files_name_list)):
            if '.jpg' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                self.json_list.append(self.files_name_list[i].replace('.jpg','.txt'))
                self.json_list2.append(self.files_name_list[i].replace('.jpg','1.txt'))
            if '.bmp' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                self.json_list.append(self.files_name_list[i].replace('.bmp','.txt'))
                self.json_list2.append(self.files_name_list[i].replace('.bmp','1.txt'))

        for i in range(len(self.img_list)):
            self.img_list[i] = os.path.join(os.path.join(datasets_path,typee),self.img_list[i])

        for i in range(len(self.json_list)):
            self.json_list[i] = os.path.join(os.path.join(datasets_path,typee),self.json_list[i])

        for i in range(len(self.json_list2)):
            self.json_list2[i] = os.path.join(os.path.join(datasets_path,typee),self.json_list2[i])


    def __len__(self):
        return len(self.img_list)        


    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_name = self.img_list[index][:-4]
        save_name = self.img_list[index].split('/')[-1]
        img = Image.open(img_path)
        height,width = np.shape(img)[0], np.shape(img)[1]
        

        json_path = self.json_list[index]
        json_path2 = self.json_list2[index]
        landmarks = json_to_numpy(json_path,json_path2,new_height=512,new_width=512,h=height,w=width)
        points = torch.tensor(landmarks)

        heatmaps = generate_heatmaps(landmarks,height,width,(41,41),
                                     new_height=512,new_width=512)
        
        heatmaps = torch.tensor(heatmaps,dtype=torch.float32)
        

        transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])
        img = transform(img)

       

        return img, heatmaps, points, height, width

class DataKeyPoint2222(torch.utils.data.Dataset):

    def __init__(self,datasets_path,typee,fixed_size):
        super().__init__()
        self.datasets_path = datasets_path
        self.files_name_list = os.listdir(os.path.join(datasets_path,typee))
        self.img_list = []
        self.json_list = []
        self.json_list2 = []
        self.target = fixed_size

        for i in range(len(self.files_name_list)):
            if '.jpg' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                self.json_list.append(self.files_name_list[i].replace('.jpg','.txt'))
                self.json_list2.append(self.files_name_list[i].replace('.jpg','1.txt'))
            if '.bmp' in self.files_name_list[i]:
                self.img_list.append(self.files_name_list[i])
                self.json_list.append(self.files_name_list[i].replace('.bmp','.txt'))
                self.json_list2.append(self.files_name_list[i].replace('.bmp','1.txt'))

        for i in range(len(self.img_list)):
            self.img_list[i] = os.path.join(os.path.join(datasets_path,typee),self.img_list[i])

        for i in range(len(self.json_list)):
            self.json_list[i] = os.path.join(os.path.join(datasets_path,typee),self.json_list[i])

        for i in range(len(self.json_list2)):
            self.json_list2[i] = os.path.join(os.path.join(datasets_path,typee),self.json_list2[i])


    def __len__(self):
        return len(self.img_list)        


    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_name = self.img_list[index][:-4]
        save_name = self.img_list[index].split('/')[-1]
        img = Image.open(img_path)
        height,width = np.shape(img)[0], np.shape(img)[1]
        

        json_path = self.json_list[index]
        json_path2 = self.json_list2[index]
        landmarks = json_to_numpy(json_path,json_path2,new_height=256,new_width=256,h=height,w=width)
        points = torch.tensor(landmarks)

        heatmaps = generate_heatmaps(landmarks,height,width,(17,17),
                                     new_height=256,new_width=256)
        
        heatmaps = torch.tensor(heatmaps,dtype=torch.float32)
        

        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
        img = transform(img)

       

        return img, heatmaps, points, height, width