#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt

def test_img(net_g, datatest, args):

    net_g.eval()
    # testing
    test_loss_color = 0
    test_loss_car = 0

    correct_color = 0
    correct_car = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)

    # 保存精度和损失的变化
    accuracy_list_color = []
    loss_list_color = []

    accuracy_list_car = []
    loss_list_car = []

    total_color = 0
    total_car = 0

    for i, data in enumerate(data_loader):
        inputs = data['image'].to(args.device)

        color_label = data['color'].to(args.device)
        car_label = data['car'].to(args.device)

        color_output, car_output = net_g(inputs)

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

        # 计算当前批次的精度和损失
        batch_accuracy_color =  correct_color / total_color
        batch_accuracy_car =  correct_car / total_car
       
        # 保存当前批次的精度和损失值
        accuracy_list_color.append(batch_accuracy_color)
        accuracy_list_car.append(batch_accuracy_car)
    
    print("accuracy_color : ", accuracy_list_color)
    print("accuracy_car : ", accuracy_list_car)
    data = {'acc_color': accuracy_list_color, 'acc_car': accuracy_list_car}
    df = pd.DataFrame(data)
    df.to_csv('test_acc.csv', index=False)


