#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
from tqdm import tqdm
import torch
import time
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from utils import ada_hessain
import pandas as pd



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


"""
from ada_hessian import AdaHessian
...
model = YourModel()
optimizer = AdaHessian(model.parameters())
...
for input, output in data:
  optimizer.zero_grad()
  loss = loss_function(output, model(input))
  loss.backward(create_graph=True)  # this is the important line! 🧐
  optimizer.step()
...
"""


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
    
    def train(self, net, train_dataloader):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        #optimizer = ada_hessain.AdaHessian(net.parameters())

        loss_automobile = nn.CrossEntropyLoss()
        sig = nn.Sigmoid()

        total_training_loss = 0
        loss_color = 0
        loss_car = 0

        correct_color = 0
        total_color = 0
        correct_car = 0
        total_car = 0

        for i, data in enumerate(train_dataloader):
            inputs1 = [batch['image'] for batch in data]
            inputs = torch.stack(inputs1).to(self.args.device)

            # inputs = data['image'].to(device=self.args.device)
            color_label = [batch['color'] for batch in data]
            color_label = torch.tensor(color_label).to(self.args.device)
            # color_label = data["color"].to(device=self.args.device)
            car_label = [batch['car'] for batch in data]
            car_label = torch.tensor(car_label).to(self.args.device)
            # car_label = data["car"].to(device=self.args.device)
            
            optimizer.zero_grad()
            color_output, car_output = net(inputs)
            
            # 梯度上升
            
            if self.args.now_round > 100 and self.args.method == "pda":
                if color_output.grad is not None:
                    color_output.retain_grad()
                    color_output.grad *= -1
           
            ############################################################
            # 计算颜色的精度
            _, predicted_color = torch.max(color_output, 1)
            total_color += color_label.size(0)
            correct_color += (predicted_color == color_label).sum().item()

            # 计算car的精度
            _, predicted_car = torch.max(car_output, 1)
            total_car += car_label.size(0)
            correct_car += (predicted_car == car_label).sum().item()
            ############################################################
            
            loss_1 = loss_automobile(color_output, color_label)
            loss_2 = loss_automobile(car_output, car_label)

            loss_color += loss_1.item()
            loss_car += loss_2.item()
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()
        
            
            total_training_loss += loss
        
        # print("Epoch: {}, Training Loss: {}".format(iter, total_training_loss/len(train_dataloader)))
        print("the loss of color / car is : ", loss_color/len(train_dataloader), " / " , loss_car/len(train_dataloader))
        print("Accuracy on color / car is: " , correct_color/total_color ," / "  , correct_car / total_car)
        data = {
            "Metric": ["Loss of color", "Loss of Gender", "Accuracy on Age", "Accuracy on Gender"],
            "Value": [loss_color/len(train_dataloader), loss_car/len(train_dataloader),
                    correct_color/total_color, correct_car/total_car]
        }
        df = pd.DataFrame(data)
        df.to_csv('local_mine.csv', index=False)
        return net.state_dict(), loss_color/len(train_dataloader), loss_car/len(train_dataloader), total_training_loss, correct_color/total_color, correct_car / total_car


class LocalUpdate2(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, train_dataloader):
        net.train()

        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        #optimizer = ada_hessain.AdaHessian(net.parameters())

        ethnicity_loss = nn.CrossEntropyLoss()
        gender_loss = nn.BCELoss()
        age_loss = nn.L1Loss()
        sig = nn.Sigmoid()

        total_training_loss = 0
        loss_age = 0
        loss_gen = 0
        loss_eth = 0

        correct_age = 0
        total_age = 0
        correct_gender = 0
        total_gender = 0
        correct_eth = 0
        total_eth = 0

        for i, data in enumerate(train_dataloader):
            # print(correct_age , " / ", correct_gender)
            inputs = data["image"].to(device=self.args.device)
            
            age_label = data["age"].to(device=self.args.device)
            gender_label = data["gender"].to(device=self.args.device)
            eth_label = data["ethnicity"].to(device=self.args.device)
            
            
            age_output, gender_output, eth_output = net(inputs)

            ############################################################
            # 计算年龄任务的精度
            # predicted_age = torch.round(torch.abs(age_output))
            predicted_age = age_output.round().squeeze()  # 四舍五入并移除维度为1的维度
            total_age += age_label.size(0)
            # correct_age += (predicted_age == age_label).sum().item()
            within_range = torch.abs(predicted_age - age_label) <= 10
            correct_age += within_range.sum().item()

            # 计算性别任务的精度
            # predicted_gender = torch.round(torch.sigmoid(gender_output))
            predicted_gender = (torch.sigmoid(gender_output) >= 0.5).float().squeeze()  
            total_gender += gender_label.size(0)
            correct_gender += (predicted_gender == gender_label).sum().item()

            # 计算种族任务的精度
            _, predicted_eth = torch.max(eth_output, 1)
            total_eth += eth_label.size(0)
            correct_eth += (predicted_eth == eth_label).sum().item()
            ############################################################

            loss_1 = ethnicity_loss(eth_output, eth_label)
            #loss_2 = gender_loss(sig(gender_output), gender_label.unsqueeze(1).float())
            # loss_3 = age_loss(age_output, age_label.unsqueeze(1).float())
            # loss_age += loss_3.item()
            #loss_gen += loss_2.item()
            loss_eth += loss_1.item()
            #loss = loss_1 + loss_2 + loss_3
            optimizer.zero_grad()
            loss_1.backward()
            optimizer.step()
            
            total_training_loss += loss_age
        
        # print("Epoch: {}, Training Loss: {}".format(iter, total_training_loss/len(train_dataloader)))
        # print("the loss of age / gender / ethnicity label is : ", loss_age/len(train_dataloader), " / " , loss_gen/len(train_dataloader), 
            #  " / " , loss_eth/len(train_dataloader))
        print("Accuracy on age / gender / ethnicity label: " , correct_age/total_age ," / "  , correct_gender / total_gender , 
            " / " , correct_eth / total_eth)
        data = {
            "Metric": ["Loss of Age", "Loss of Gender", "Loss of Ethnicity", "Accuracy on Age", "Accuracy on Gender", "Accuracy on Ethnicity"],
            "Value": [loss_age/len(train_dataloader), loss_gen/len(train_dataloader), loss_eth/len(train_dataloader),
                    correct_age/total_age, correct_gender/total_gender, correct_eth/total_eth]
        }
        df = pd.DataFrame(data)
        df.to_csv('local_mine.csv', index=False)
        return net.state_dict(), loss_age/len(train_dataloader), loss_gen/len(train_dataloader), loss_eth/len(train_dataloader), total_training_loss, correct_age/total_age, correct_gender / total_gender, correct_eth / total_eth 

