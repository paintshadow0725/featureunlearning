import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader

def entropy(probabilities):
    epsilon = 1e-12
    probabilities = np.clip(probabilities, epsilon, 1.0)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt

def output_visual(net_g, datatest, args):

    net_g.eval()
    data_loader = DataLoader(datatest, batch_size=args.bs)
    age = []
    gender = []
    eth = []

    for i, data in enumerate(data_loader):
        # print(correct_age , " / ", correct_gender)
        inputs = data["image"].to(device=args.device)  
        age_output, gender_output, eth_output = net_g(inputs)
        # print("===========age_output: ", age_output)

        age_num = age_output.detach().cpu().numpy()
        # print("age_num: ", age_num)
        gender_num = gender_output.detach().cpu().numpy()
        eth_num = eth_output.detach().cpu().numpy()
        age.append(age_num)
        gender.append(gender_num)
        eth.append(eth_num)
    # print("age: ", age)
    print("gender: ", gender)
    # print("eth: ", eth)
    # 计算每个输出的熵值
    age_entropies = [entropy(age_dist) for age_dist in age]
    gender_entropies = [entropy(gender_dist) for gender_dist in gender]
    eth_entropies = [entropy(eth_dist) for eth_dist in eth]

    # 创建保存图片的文件夹（如果不存在）
    save_dir = './save'
    os.makedirs(save_dir, exist_ok=True)

    # 绘制直方图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(age_entropies, bins='auto')
    plt.xlabel('Age Entropy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Age Entropy')

    plt.subplot(1, 3, 2)
    plt.hist(gender_entropies, bins='auto')
    plt.xlabel('Gender Entropy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gender Entropy')

    plt.subplot(1, 3, 3)
    plt.hist(eth_entropies, bins='auto')
    plt.xlabel('Ethnicity Entropy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Ethnicity Entropy')

    plt.tight_layout()

    # 保存图片
    plt.savefig(os.path.join(save_dir, 'output_distribution.png'))
    plt.show()