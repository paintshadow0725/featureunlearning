#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from tqdm import tqdm
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from utils import ada_hessain
from utils.options import args_parser


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
        self.args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def mnisttrain(self, net, train_dataloader):
        net.train()
        args = args_parser()
        args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        #optimizer = ada_hessain.AdaHessian(net.parameters())

        mnistloss = nn.CrossEntropyLoss()

        total_training_loss = 0
        loss_1 = 0
        loss_2 = 0
        loss_task1 = 0
        loss_task2 = 0
        
        correct_1 = 0
        total_1 = 0
        correct_2 = 0
        total_2 = 0
        

        for i, data in enumerate(train_dataloader):
        
            inputs, labels_1, labels_2 = data
            
            
            optimizer.zero_grad()
            inputs, labels_1, labels_2 = inputs.to(args.device), labels_1.to(args.device), labels_2.to(args.device)
            output1, output2 = net(inputs)
            ############################################################
            # 计算1任务的精度
            _, predicted_1 = torch.max(output1, 1)
            total_1 += labels_1.size(0)
            correct_1 += (predicted_1 == labels_1).sum().item()

            # 计算2任务的精度
            _, predicted_2 = torch.max(output2, 1)
            total_2 += labels_2.size(0)
            correct_2 += (predicted_2 == labels_2).sum().item()
            ############################################################

            loss_1 = mnistloss(output1, labels_1)
            loss_2 = mnistloss(output2, labels_2)
            loss_task1 += loss_1.item()
            loss_task2 += loss_2.item()
            
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()
            
            total_training_loss += loss
        
        # print("Epoch: {}, Training Loss: {}".format(iter, total_training_loss/len(train_dataloader)))
        # print("the loss of age / gender / ethnicity label is : ", loss_age/len(train_dataloader), " / " , loss_gen/len(train_dataloader), 
            #  " / " , loss_eth/len(train_dataloader))
        return net.state_dict(), loss_1/len(train_dataloader), loss_2/len(train_dataloader), total_training_loss, correct_1/total_1, correct_2 / total_2

            
