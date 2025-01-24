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
    
class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(pretrained=True)
        # for param in self.net.parameters():
        #     init.zeros_(param)
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
 
        self.net.fc1 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 1))]))
 
        self.net.fc2 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 1))]))
 
        self.net.fc3 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 5))]))
        
        # 保存要冻结的层的参数
        self.frozen_params_1 = list(self.net.fc1.parameters())
        self.frozen_params_2 = list(self.net.fc2.parameters())
        self.frozen_params_3 = list(self.net.fc3.parameters())
    
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
        age_head = self.net.fc1(self.net(x))
        gender_head = self.net.fc2(self.net(x))
        ethnicity_head = self.net.fc3(self.net(x))
        return age_head, gender_head, ethnicity_head


class HydraNetattack(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(pretrained=True)
        #self.net.load_state_dict(torch.load('E:\\experiments\\new\\Rapid-Retraining\\Rapid-Retraining\\model_params.pth'))
        self.n_features = self.net.fc.in_features
        self.net.fc = nn.Identity()
 
        self.net.fc1 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 1))]))
 
        self.net.fc2 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 1))]))
 
        self.net.fc3 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 5))]))
        
        # 保存要冻结的层的参数
        self.frozen_params_1 = list(self.net.fc1.parameters())
        self.frozen_params_2 = list(self.net.fc2.parameters())
        self.frozen_params_3 = list(self.net.fc3.parameters())
    
    def freeze_fc1(self):
        with torch.no_grad():
            for param in self.frozen_params_1:
                param.zero_()
                param.requires_grad = False
    
    def freeze_fc2(self):
        with torch.no_grad():
            for param in self.frozen_params_2:
                param.zero_()
                param.requires_grad = False
    
    
    def freeze_fc3(self):
        with torch.no_grad():
            for param in self.frozen_params_3:
                param.zero_()
                param.requires_grad = False
         
    def forward(self, x):
        age_head = self.net.fc1(self.net(x))
        gender_head = self.net.fc2(self.net(x))
        ethnicity_head = self.net.fc3(self.net(x))
        return age_head, gender_head, ethnicity_head
