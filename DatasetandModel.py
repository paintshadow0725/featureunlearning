import os
import random
import shutil
import glob
import pandas as pd
import torch
from torch import nn
import torch.nn.init as init
from torch.utils.data import DataLoader,TensorDataset,Dataset
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
import numpy as np
import timm
import torch.nn.functional as F

class UTKFace(Dataset):
    def __init__(self, image_paths):
        self.transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.image_paths = image_paths
        self.images = []
        self.ages = []
        self.genders = []
        self.races = []
        image_path = glob.glob(os.path.join(image_paths, '*.jpg'))
        for path in image_path:
            filename = path[8:].split("_")
             
            if len(filename)==4:
                self.images.append(path)
                self.ages.append(int(filename[0]))
                self.genders.append(int(filename[1]))
                self.races.append(int(filename[2]))
 
    def __len__(self):
          return len(self.images)
 
    def __getitem__(self, index):
            img = Image.open(self.images[index]).convert('RGB')
            img = self.transform(img)
           
            age = self.ages[index]
            gender = self.genders[index]
            eth = self.races[index]
             
            sample = {'image':img, 'age': age, 'gender': gender, 'ethnicity':eth}
             
            return sample
class Automobile(Dataset):
    def __init__(self, image_paths):
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.image_paths = image_paths
        self.images = []
        self.color = []
        self.car = []
        image_path = glob.glob(os.path.join(image_paths, '*.jpg'))
        for path in image_path:
            filename = path[8:].split("_")
            if len(filename)==3:
                self.images.append(path)
                self.color.append(int(filename[0].split('/')[-1]))
                self.car.append(int(filename[1]))
 
    def __len__(self):
          return len(self.images)
 
    def __getitem__(self, index):
            img = Image.open(self.images[index]).convert('RGB')
            img = self.transform(img)
            color = self.color[index]
            car = self.car[index]
            sample = {'image':img, 'color': color, 'car': car}
            return sample

class UTKFacesisa(Dataset):
    def __init__(self, image_paths, num_shards, shard_index):
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.image_paths = image_paths
        self.images = []
        self.ages = []
        self.genders = []
        self.races = []
        image_path = glob.glob(os.path.join(image_paths, '*.jpg'))
        for path in image_path:
            filename = path[8:].split("_")
            if len(filename) == 4:
                self.images.append(path)
                self.ages.append(int(filename[0]))
                self.genders.append(int(filename[1]))
                self.races.append(int(filename[2]))

        # 根据分片索引加载对应的数据
        shard_size = len(self.images) // num_shards
        start_index = shard_index * shard_size
        end_index = start_index + shard_size
        self.images = self.images[start_index:end_index]
        self.ages = self.ages[start_index:end_index]
        self.genders = self.genders[start_index:end_index]
        self.races = self.races[start_index:end_index]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = self.transform(img)

        age = self.ages[index]
        gender = self.genders[index]
        eth = self.races[index]

        sample = {'image': img, 'age': age, 'gender': gender, 'ethnicity': eth}

        return sample

class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(pretrained=False)
        # for param in self.net.parameters():
        #     init.zeros_(param)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
 
        self.net.fc1 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 4))]))
 
        self.net.fc2 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 5))]))
 
        # 保存要冻结的层的参数
        self.frozen_params_1 = list(self.net.fc1.parameters())
        self.frozen_params_2 = list(self.net.fc2.parameters())
    
    def freeze_1(self):
        with torch.no_grad():
            for param in self.frozen_params_1:
                offset = torch.tensor(random.uniform(-1000, -100000))
                param.add_(offset)
                # param.requires_grad = False
    def finetune_1(self):
        with torch.no_grad():
            for param in self.frozen_params_1:
                offset = torch.tensor(random.uniform(-0.1, 0.1))
                param.add_(offset)
                # param.requires_grad = False
    def freeze_2(self):
        with torch.no_grad():
            for param in self.frozen_params_2:
                offset = torch.tensor(random.uniform(-1000, -100000))
                param.add_(offset)
                # param.requires_grad = False
    
    def forward(self, x):
        age_head = self.net.fc1(self.net(x))
        gender_head = self.net.fc2(self.net(x))
        return age_head, gender_head

class HydraNet_mobilenet(nn.Module):
    def __init__(self):
        super().__init__()
        # 将ResNet101替换为MobileNet
        self.net = models.mobilenet_v2(pretrained=False)

        # 调整特征提取部分，MobileNetV2的最后一个通道是输出特征的数量
        self.n_features = self.net.classifier[1].in_features

        # 用一个Identity层替换classifier层，用于提取特征
        self.net.classifier = nn.Identity()

        self.fc1 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 4))]))
 
        self.fc2 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 5))]))
        # 保存要冻结的层的参数
        self.frozen_params_1 = list(self.fc1.parameters())
        self.frozen_params_2 = list(self.fc2.parameters())
    
    def freeze_1(self):
        with torch.no_grad():
            for param in self.frozen_params_1:
                offset = torch.tensor(random.uniform(-1000, -100000))
                param.add_(offset)
                # param.requires_grad = False
    def finetune_1(self):
        with torch.no_grad():
            for param in self.frozen_params_1:
                offset = torch.tensor(random.uniform(-0.1, 0.1))
                param.add_(offset)
                param.requires_grad = False
    def freeze_2(self):
        with torch.no_grad():
            for param in self.frozen_params_2:
                offset = torch.tensor(random.uniform(-1000, -100000))
                param.add_(offset)
                # param.requires_grad = False
    
    def forward(self, x):
        age_head = self.fc1(self.net(x))
        gender_head = self.fc2(self.net(x))
        return age_head, gender_head


class HydraNetattack(nn.Module):
    def __init__(self):
        super().__init__()
        # 将ResNet101替换为MobileNet
        self.net = models.resnet101(pretrained=False)

        # 调整特征提取部分，MobileNetV2的最后一个通道是输出特征的数量
        self.n_features = self.net.classifier[1].in_features

        # 用一个Identity层替换classifier层，用于提取特征
        self.net.classifier = nn.Identity()

 
        # 定义三个全连接层
        self.fc1 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features, self.n_features)),
             ('relu1', nn.ReLU()),
             ('final', nn.Linear(self.n_features, 1))]))

        self.fc2 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features, self.n_features)),
             ('relu1', nn.ReLU()),
             ('final', nn.Linear(self.n_features, 1))]))

        self.fc3 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features, self.n_features)),
             ('relu1', nn.ReLU()),
             ('final', nn.Linear(self.n_features, 5))]))

        # 保存要冻结的层的参数
        self.frozen_params_1 = list(self.fc1.parameters())
        self.frozen_params_2 = list(self.fc2.parameters())
        self.frozen_params_3 = list(self.fc3.parameters())

    # 冻结和微调函数保持不变
    def freeze_fc1(self):
        with torch.no_grad():
            for param in self.frozen_params_1:
                offset = torch.tensor(random.uniform(-1000, -100000))
                param.add_(offset)
                # param.requires_grad = False
    
    def freeze_fc2(self):
        with torch.no_grad():
            for param in self.frozen_params_2:
                offset = torch.tensor(random.uniform(-1000, -100000))
                param.add_(offset)
                # param.requires_grad = False
    
    
    def freeze_fc3(self):
        with torch.no_grad():
            for param in self.frozen_params_3:
                offset = torch.tensor(random.uniform(-1000, -100000))
                param.add_(offset)
                param.requires_grad = False
    def finetune_fc3(self):
        with torch.no_grad():
            for param in self.frozen_params_3:
                offset = torch.tensor(random.uniform(-0.1, 0.1))
                param.add_(offset)
                param.requires_grad = False

         
    def forward(self, x):
        x = self.net(x)  # 共享特征提取
        age_head = self.fc1(x)
        gender_head = self.fc2(x)
        ethnicity_head = self.fc3(x)
        return age_head, gender_head, ethnicity_head

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3_task1 = nn.Linear(84, 4)  
        self.fc3_task2 = nn.Linear(84, 5)   

        # 保存要冻结的层的参数
        self.frozen_params_1 = list(self.fc3_task1.parameters())
        self.frozen_params_2 = list(self.fc3_task2.parameters())
    
    def freeze_1(self):
        with torch.no_grad():
            for param in self.frozen_params_1:
                offset = torch.tensor(random.uniform(-1000, -100000))
                param.add_(offset)
                # param.requires_grad = False
    
    def freeze_2(self):
        with torch.no_grad():
            for param in self.frozen_params_2:
                offset = torch.tensor(random.uniform(-1000, -100000))
                param.add_(offset)
                # param.requires_grad = False
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        output_task1 = self.fc3_task1(x) 
        output_task2 = self.fc3_task2(x) 
        return output_task1, output_task2

class UTKFaceViT(Dataset):
    def __init__(self, image_paths):
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.image_paths = image_paths
        self.images = []
        self.ages = []
        self.genders = []
        self.races = []
        image_path = glob.glob(os.path.join(image_paths, '*.jpg'))
        for path in image_path:
            filename = path[8:].split("_")
             
            if len(filename)==4:
                self.images.append(path)
                self.ages.append(int(filename[0]))
                self.genders.append(int(filename[1]))
                self.races.append(int(filename[2]))
 
    def __len__(self):
          return len(self.images)
 
    def __getitem__(self, index):
            img = Image.open(self.images[index]).convert('RGB')
            img = self.transform(img)
           
            age = self.ages[index]
            gender = self.genders[index]
            eth = self.races[index]
             
            sample = {'image':img, 'age': age, 'gender': gender, 'ethnicity':eth}
             
            return sample

class ViT_MT_Model(nn.Module):
    def __init__(self):
        super(ViT_MT_Model, self).__init__()
        self.vit_model = timm.create_model('vit_base_patch16_224', pretrained=False) # 4
        num_features = self.vit_model.head.in_features
        self.vit_model = nn.Sequential(*list(self.vit_model.children())[:-1])   # 3
        
        # 共享的特征提取层
        self.features_extractor = nn.Sequential(
            self.vit_model,
            #torch.nn.Linear(150528,10240), #这里不是1024吗，咋来的呢
            #torch.nn.Linear(10240,1024),
            torch.nn.Linear(768,128),
            torch.nn.Linear(128,80)
            # nn.AdaptiveAvgPool2d((1, 1))  # 将特征图的大小调整为 (1, 1)
        )  
        
        self.head_person = nn.Sequential(
            torch.nn.Linear(15680,1024),
            torch.nn.Linear(1024,128),
            torch.nn.Linear(128, 4)
        )
        self.head_vehicle = nn.Sequential(
            torch.nn.Linear(15680,1024),
            torch.nn.Linear(1024,128),
            torch.nn.Linear(128, 5)
        )

        # 保存要冻结的层的参数
        self.frozen_params_1 = list(self.head_person.parameters())
        self.frozen_params_2 = list(self.head_vehicle.parameters())
    
    def freeze_1(self):
        with torch.no_grad():
            for param in self.frozen_params_1:
                param.zero_()
                param.requires_grad = False
    def finetune_1(self):
        with torch.no_grad():
            for param in self.frozen_params_1:
                offset = torch.tensor(random.uniform(-0.1, 0.1))
                param.add_(offset)
                param.requires_grad = False
    def freeze_2(self):
        with torch.no_grad():
            for param in self.frozen_params_2:
                param.zero_()
                param.requires_grad = False

    def forward(self, x):
        features = self.features_extractor(x)   #这里是128 * 80
        features_flat = torch.flatten(features, start_dim=1)    

        person_output = self.head_person(features_flat)
        person_output = F.softmax(person_output, dim=1)
        # _, person = torch.max(person_output, dim=1)
        # print("this is person", person_output)

        vehicle_output = self.head_vehicle(features_flat)
        vehicle_output = F.softmax(vehicle_output, dim=1)

        return person_output, vehicle_output