import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models
class MultiTaskLeNet(nn.Module):
    def __init__(self, num_tasks):
        super(MultiTaskLeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        
        self.task_fc_layers = nn.ModuleList()
        for _ in range(num_tasks):
            num_classes = 10
            if num_tasks == 1:
                num_classes = 2
            self.task_fc_layers.append(nn.Linear(84, num_classes))
        '''
        # # for i in range(num_tasks):
        # #     layer_name = f"task_fc_layer_{i}"
        # #     num_classes = 10
        # #     if num_tasks == 1:
        # #         num_classes = 2
        # #     self.task_fc_layers.append(nn.Linear(84, num_classes, layer_name))'''
        # self.net.fc1 = nn.Sequential(OrderedDict(
        #     [('linear', nn.Linear(self.n_features,self.n_features)),
        #     ('relu1', nn.ReLU()),
        #     ('final', nn.Linear(self.n_features, 10))]))
 
        # self.net.fc2 = nn.Sequential(OrderedDict(
        #     [('linear', nn.Linear(self.n_features,self.n_features)),
        #     ('relu1', nn.ReLU()),
        #     ('final', nn.Linear(self.n_features, 2))]))

        self.frozen_params1 = list(self.fc1.parameters())
        self.frozen_params2 = list(self.fc2.parameters())

    def freeze_fc1(self):
        with torch.no_grad():
            for param in self.frozen_params1:
                param.zero_()
                param.requires_grad = False
    
    def freeze_fc2(self):
        with torch.no_grad():
            for param in self.frozen_params2:
                param.zero_()
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        output1 = self.task_fc_layers[0](x)
        output1 = F.log_softmax(output1, dim=1)

        output2 = self.task_fc_layers[1](x)
        output2 = F.log_softmax(output2, dim=1)
        
        return output1 , output2

'''class MultiTaskLeNet(nn.Module):
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
            ('final', nn.Linear(self.n_features, 10))]))
 
        self.net.fc2 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('final', nn.Linear(self.n_features, 2))]))
 
        
        
        # 保存要冻结的层的参数
        self.frozen_params_1 = list(self.net.fc1.parameters())
        self.frozen_params_2 = list(self.net.fc2.parameters())
        
    
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
    
    
         
    def forward(self, x):
        output1 = self.net.fc1(self.net(x))
        output2 = self.net.fc2(self.net(x))
        return output1, output2
'''