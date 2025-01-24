#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt

def mnisttest(net_g, datatest, args):

    net_g.eval()

    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)

    # 保存精度和损失的变化
    accuracy1_list = []
    accuracy2_list = []

    
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

    for i, data in enumerate(data_loader):
        # print(correct_age , " / ", correct_gender)
        inputs, labels_1, labels_2 = data
        inputs, labels_1, labels_2 = inputs.to(args.device), labels_1.to(args.device), labels_2.to(args.device)  
        
        output1, output2 = net_g(inputs)

       
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
        
        total_training_loss = loss_1 + loss_2
        
        # 计算当前批次的精度和损失
        batch_accuracy_1 =  correct_1 / total_1
        
        batch_accuracy_2 =  correct_2 / total_2
        
        
        # 保存当前批次的精度和损失值
        accuracy1_list.append(batch_accuracy_1)
        accuracy2_list.append(batch_accuracy_2)
    
    # accuracy_1 =  correct_1 / len(data_loader.dataset)

    # # test_loss_gen /= len(data_loader.dataset)
    # accuracy_2 =  correct_2 / len(data_loader.dataset)

    # 生成曲线图
    plt.figure()
    plt.plot(range(1, l+1), accuracy1_list, label='Digit Recognition')
    plt.plot(range(1, l+1), accuracy2_list, label='Odd-even Number Classification')
    plt.xlabel('Batch')
    plt.ylabel('test_acc')
    plt.legend()
    plt.savefig('./save/mnisttestacc.png')

    # fig2 = plt.figure()
    # plt.plot(range(1, l+1), loss_list_age, label='age')
    # plt.plot(range(1, l+1), loss_list_age, label='age')
    # plt.plot(range(1, l+1), loss_list_age, label='age')
    # plt.xlabel('Batch')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.savefig('./save/testloss.png')



