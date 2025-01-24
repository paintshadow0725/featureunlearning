#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

from DatasetandModel import HydraNet, HydraNetattack, UTKFace
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from scipy.cluster.hierarchy import linkage,dendrogram
import pandas as pd

import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from utils.sampling import UTKFace_iid
from utils.options import args_parser
# from models.Update import LocalUpdate
from train.facetrain import LocalUpdate
from models.Update_unlearning import Local_Update

from models.Fed import FedAvg
from models.test import test_img
import matplotlib.pyplot as plt

def train_test_split(dataset, train_ratio=0.8):
    torch.manual_seed(42)  
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = (dataset_size - train_size)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset

def test_img(net_g, datatest):

    net_g.eval()

    correct_age = 0
    correct_gender = 0
    correct_eth = 0
    data_loader = DataLoader(datatest, batch_size=64)
    accuracy_list_age = []

    accuracy_list_gen = []
    accuracy_list_eth = []
    total_age = 0
    total_gender = 0
    total_eth = 0

    for i, data in enumerate(data_loader):
        # print(correct_age , " / ", correct_gender)
        inputs = data["image"]
        age_label = data["age"]
        gender_label = data["gender"]
        eth_label = data["ethnicity"]
        age_output, gender_output, eth_output = net_g(inputs)
        predicted_age = age_output.round().squeeze() 
        total_age += age_label.size(0)
        within_range = torch.abs(predicted_age - age_label) <= 10
        correct_age += within_range.sum().item()

        predicted_gender = (torch.sigmoid(gender_output) >= 0.5).float().squeeze()  
        total_gender += gender_label.size(0)
        correct_gender += (predicted_gender == gender_label).sum().item()

        _, predicted_eth = torch.max(eth_output, 1)
        total_eth += eth_label.size(0)
        correct_eth += (predicted_eth == eth_label).sum().item()

        batch_accuracy_age =  correct_age / total_age     
        batch_accuracy_gen =  correct_gender / total_gender
        batch_accuracy_eth =  correct_eth / total_eth

        accuracy_list_age.append(batch_accuracy_age)
        accuracy_list_gen.append(batch_accuracy_gen)
        accuracy_list_eth.append(batch_accuracy_eth)


    return accuracy_list_age, accuracy_list_gen, accuracy_list_eth

   

def attack(test_data):
    
    net_glob = HydraNetattack()
    net_glob.load_state_dict(torch.load('/data/xia-group-12/hy/Rapid-Retraining/model_params_mine.pth'))
    accuracy_list_age, accuracy_list_gen, accuracy_list_eth = test_img(net_glob, test_data)
    net_glob_init = HydraNet()
    accuracy_list_age_init, accuracy_list_gen_init, accuracy_list_eth_init = test_img(net_glob_init, test_data)
    result_age = [0] * len(accuracy_list_age)
    result_gen = [0] * len(accuracy_list_gen)
    result_eth = [0] * len(accuracy_list_eth)
    for i in range(len(accuracy_list_age)):
        
        print("before is : ", accuracy_list_eth[i])
        print("before is : ", accuracy_list_age[i])
        print("before is : ", accuracy_list_eth[i])
        if accuracy_list_age[i] > accuracy_list_age_init[i]:
            result_age[i] = accuracy_list_age[i] - accuracy_list_age_init[i]
        else:
            result_age[i] = 0
        if accuracy_list_gen[i] > accuracy_list_gen_init[i]:
            result_gen[i] = accuracy_list_gen[i] - accuracy_list_gen_init[i]
        else:
            result_gen[i] = 0
        if accuracy_list_eth[i] > accuracy_list_eth_init[i]:
            print("after is : ", accuracy_list_eth[i])
            print("init is :", accuracy_list_eth_init[i])
            result_eth[i] = accuracy_list_eth[i] - accuracy_list_eth_init[i]
        else:
            print("after is : ", accuracy_list_eth[i])
            print("init is :", accuracy_list_eth_init[i])
            result_eth[i] = 0

    plt.figure()
    plt.plot(range(1, len(accuracy_list_age)+1), result_age, label='age')
    plt.plot(range(1, len(accuracy_list_age)+1), result_gen, label='gender')
    plt.plot(range(1, len(accuracy_list_age)+1), result_eth, label='ethnicity')
    plt.xlabel('Batch')
    plt.ylabel('attack2')
    plt.legend()
    plt.savefig('./save/attack.png')

def attack2(test_data):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_glob = HydraNet().to(device=args.device)
    net_glob.load_state_dict(torch.load('/data/xia-group-12/hy/Rapid-Retraining/modelpth/sisa50.pth'))
    data_loader = DataLoader(test_data, batch_size=args.bs)
    for i, data in enumerate(data_loader):
        # print(correct_age , " / ", correct_gender)
        inputs = data["image"].to(device=args.device) 
        age_label = data["age"].to(device=args.device)
        gender_label = data["gender"].to(device=args.device)
        y_true = data["ethnicity"].to(device=args.device)
        _, _, ethoutput = net_glob(inputs)
        _, y_pred = torch.max(ethoutput, 1)
        y_true = y_true.cpu() 
        y_pred = y_pred.cpu()  
        precision_macro = precision_score(y_true, y_pred, average='macro' ) 
        precision_micro = precision_score(y_true, y_pred, average='micro')  
        precision_weighted = precision_score(y_true, y_pred, average='weighted') 
        print(f'Macro-average accuracy:{precision_macro:.4f}')
        print(f'Micro-average accuracy:{precision_micro:.4f}')
        print(f'Weighted average precision:{precision_weighted:.4f}')
        recall_macro = recall_score(y_true, y_pred, average='macro') 
        recall_micro = recall_score(y_true, y_pred, average='micro')  
        recall_weighted = recall_score(y_true, y_pred, average='weighted') 
        print(f'macro-averaging-recall:{recall_macro:.4f}')
        print(f'Micro-Average Recall Rate:{recall_micro:.4f}')
        print(f'Weighted average recall:{recall_weighted:.4f}')
        f1_macro = f1_score(y_true, y_pred, average='macro')  
        f1_micro = f1_score(y_true, y_pred, average='micro')  
        f1_weighted = f1_score(y_true, y_pred, average='weighted')  
        print(f'{f1_macro:.4f}')
        print(f'micro_f1：{f1_micro:.4f}')
        print(f'weigth_f1：{f1_weighted:.4f}')
        mcc = matthews_corrcoef(y_true, y_pred)
        print(f'Matthews：{mcc:.4f}')
       
    
        


if __name__ == '__main__':
    train_dataset = UTKFace('UTKFace/')
    train_data, test_data = train_test_split(train_dataset, train_ratio=0.8)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=64)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=64)
    args = args_parser()
    attack2(test_data)
