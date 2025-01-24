#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import sys
import os

import matplotlib
from DatasetandModel import UTKFace

from datasetmnist import get_datasetmnist
from mnistmodel import MultiTaskLeNet


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
import torchvision
from utils.sampling import UTKFace_iid, mnist_iid
from utils.options import args_parser
# from models.Update import LocalUpdate
# from train.facetrain import LocalUpdate
from train.mnisttrain import LocalUpdate
from models.Update_unlearning import Local_Update

from models.Fed import FedAvg
from models.mnisttest import mnisttest

# 定义相关性系数矩阵计算函数
def compute_correlation_matrix(parameters):
    num_params = len(parameters)
    correlation_matrix = np.zeros((num_params, num_params))

    for i in range(num_params):
        for j in range(num_params):
            if i == j:
                param1 = parameters[i].detach().cpu().numpy().flatten()
                param2 = parameters[j].detach().cpu().numpy().flatten()
                correlation_matrix[i, j] = np.corrcoef(param1, param2)[0, 1]
            else:
                continue

    return correlation_matrix
# def compute_correlation_matrix(model, num_params):
    
#     correlation_matrix = np.zeros((num_params, num_params))

#     for i, (name1, param1) in enumerate(model.named_parameters()):
#         for j, (name2, param2) in enumerate(model.named_parameters()):
#             if name1 == name2:
#                 param1_values = param1.detach().cpu().numpy().flatten()
#                 param2_values = param2.detach().cpu().numpy().flatten()
#                 correlation_matrix[i, j] = np.corrcoef(param1_values, param2_values)[0, 1]
#             else:
#                 continue

#     return correlation_matrix

import torch

def train_test_split(dataset, train_ratio=0.8):
    torch.manual_seed(42)  # 设置随机种子为 42
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = (dataset_size - train_size)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def testmnist():
    net_glob.eval()
    mnisttest(net_glob, test_data, args)
    print("testing is over")


if __name__ == '__main__':
    # output_file = open("output.txt", "w")
    # sys.stdout = output_file
    fw = open('mtfl.txt', 'w')
    start_time = time.time()
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    configs = {
        "mnist": {
            "path": 'data/MultiMNIST',
            "all_tasks": ["L", "R"]
        }
    }
    # # load dataset and split users
    
    # train_dataset = UTKFace('UTKFace/')
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.bs)
    
    # 加载原始数据集
    if args.datasets == 'UTKFace':
        train_dataset = UTKFace('UTKFace/')

        # 划分训练集和测试集
        train_data, test_data = train_test_split(train_dataset, train_ratio=0.8)

        # 创建训练集数据加载器
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.bs)

        # 创建测试集数据加载器
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.bs)

        dict_users = UTKFace_iid(train_dataset, args.num_users)

        # build model
        net_glob = HydraNet().to(device=args.device)
    elif args.datasets == 'mnist':
        train_dataset, train_dataloader, test_data, test_dataloader = get_datasetmnist(args, configs)
        dict_users = mnist_iid(train_dataset, args.num_users)
        num_tasks=2
        net_glob = MultiTaskLeNet(num_tasks).to(args.device)
    
    
    time_is = 0
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    w_init = copy.deepcopy(net_glob.state_dict())
    # w_noise = copy.deepcopy(net_noise_glob.state_dict())
    
    

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    delete_round = args.delete_round
    delete_attr = args.delete_attrs

    loss_age_plt = []
    loss_gen_plt = []
    loss_eth_plt = []
    loss1_plt = []
    loss2_plt = []

    acc_age_plt = []
    acc_gen_plt = []
    acc_eth_plt = []
    acc_age_plt1 = []
    acc_gen_plt1 = []
    acc_eth_plt1 = []

    acc1_plt = []
    acc2_plt = []
    acc1_plt1 = []
    acc2_plt1 = []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    for iter in range(args.epochs):
        net_glob.train()

        loss_locals_age = []
        loss_locals_gen = []
        loss_locals_eth = []
        loss1_locals = []
        loss2_locals = []

        # local_total_training_loss = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_users[idx])
            if args.datasets == 'UTKFace':
                w, loss_age, loss_gen, loss_eth, total_training_loss, acc_age, acc_gen, acc_eth = local.train(net=copy.deepcopy(net_glob).to(args.device), train_dataloader=train_dataloader)
            elif args.datasets == 'mnist':
                w, loss_1, loss_2, total_training_loss, acc_1, acc_2 = local.mnisttrain(net=copy.deepcopy(net_glob).to(args.device), train_dataloader=train_dataloader)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            
            if args.datasets == 'UTKFace':
                loss_locals_age.append(copy.deepcopy(loss_age))
                loss_locals_gen.append(copy.deepcopy(loss_gen))
                loss_locals_eth.append(copy.deepcopy(loss_eth))

                acc_age_plt1.append(acc_age)
                acc_gen_plt1.append(acc_gen)
                acc_eth_plt1.append(acc_eth)
            elif args.datasets == 'mnist':
                loss1_locals.append(loss_1)
                loss2_locals.append(loss_2)

                acc1_plt1.append(acc_1)
                acc2_plt1.append(acc_2)


        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        # print("the sum of age : ", sum(loss_locals_age), "the lenth of loss_locals_age : ", len(loss_locals_age))
        if args.datasets == 'UTKFace':
            # print loss
            loss_avg_age = sum(loss_locals_age) / len(loss_locals_age)
            fw.write('Round {:3d}, Average loss of age : {:.3f}\n"'.format(iter, loss_avg_age))
            loss_age_plt.append(loss_avg_age)
            # exit()
            loss_avg_gen = sum(loss_locals_gen) / len(loss_locals_gen)
            fw.write('Round {:3d}, Average loss of gender : {:.3f}\n'.format(iter, loss_avg_gen))
            loss_gen_plt.append(loss_avg_gen)

            loss_avg_eth = sum(loss_locals_eth) / len(loss_locals_eth)
            fw.write('Round {:3d}, Average loss of ethnicity : {:.3f}\n'.format(iter, loss_avg_eth))
            loss_eth_plt.append(loss_avg_eth)

            acc_avg_age = sum(acc_age_plt1) / len(acc_age_plt1)
            acc_age_plt.append(acc_avg_age)

            acc_avg_gen = sum(acc_gen_plt1) / len(acc_gen_plt1)
            acc_gen_plt.append(acc_avg_gen)

            acc_avg_eth = sum(acc_eth_plt1) / len(acc_eth_plt1)
            acc_eth_plt.append(acc_avg_eth)
        elif args.datasets == 'mnist':
            loss1_avg = sum(loss1_locals) / len(loss1_locals)
            loss1_plt.append(loss1_avg)

            loss2_avg = sum(loss2_locals) / len(loss2_locals)
            loss2_plt.append(loss2_avg)

            acc1_avg = sum(acc1_plt1) / len(acc1_plt1)
            acc1_plt.append(acc1_avg)

            acc2_avg = sum(acc2_plt1) / len(acc2_plt1)
            acc2_plt.append(acc2_avg)


        fw.write("round time is : {:.2f}s\n".format(time.time() - start_time))
        print("=========================================")
        # print("Epoch: {}, Training Loss: {}".format(iter, total_training_loss/len(train_dataloader)))
        # print("the loss of age / gender / ethnicity label is : ", loss_locals_age/len(train_dataloader), " / " , loss_locals_gen/len(train_dataloader), 
        #     " / " , loss_locals_eth/len(train_dataloader))
        # # print("Accuracy on age / gender / ethnicity label: " , correct_age/total_age ," / "  , correct_gender / total_gender , 
        # #     " / " , correct_eth / total_eth)
    
    
        if iter == delete_round:
            print("Delete Round is : ", iter, "the unlearning start")
            if delete_attr == 0:
                net_glob.freeze_fc1()
                fw.write("Delete Age\n")
            elif delete_attr == 1:
                net_glob.freeze_fc2()
                fw.write("Delete gender\n")
            elif delete_attr == 2:
                if args.datasets == 'UTKFace':
                    net_glob.freeze_fc3()
                fw.write("Delete ethnicity\n")

            if args.method == "retrain":
                fw.write("the unlerning method is retrain\n")
                fw.write("round unlearning time is : {:.2f}s\n".format(time.time() - start_time))
                #重新加载模型参数
                net_glob.load_state_dict(w_init)
                w_glob = net_glob.state_dict()
            elif args.method == "unlearning":
                fw.write("the unlerning method is unlearning\n")
                fw.write("round unlearning time is : {:.2f}s\n".format(time.time() - start_time))
                
                parameters = list(net_glob.parameters())
                corr_matrix = compute_correlation_matrix(parameters)
                # corr_matrix = compute_correlation_matrix(net_glob.parameters(), len(parameters))
                print("successful build model_corr_matrix")

                # 设置微调阈值，用于确定哪些参数之间的相关性高
                threshold = 15

                # 遍历相关性系数矩阵和模型参数，进行微调相关性高的参数
                for correlation, (name, param) in zip(ccm, net_glob.named_parameters()):
                    if abs(correlation) > threshold and 'fc' not in name:
                        adjustment = torch.tensor(random.uniform(-0.5, 0.5))
                        param.data += adjustment
                        param.requires_grad = True
                print("successful set resnet parameters")


    print("the sum of time : {:.2f}s".format(time.time() - start_time))
    # print("loss_age_plt\n", loss_age_plt)
    # print("loss_gen_plt\n", loss_gen_plt)
    # print("loss_eth_plt\n", loss_eth_plt)
    # print("acc_age_plt\n", acc_age_plt)
    # print("acc_gen_plt\n", acc_gen_plt)
    # print("acc_eth_plt\n", acc_eth_plt)
        
    
    if args.datasets == 'UTKFace':
        # plot loss curve
        plt.figure()
        plt.plot(range(args.epochs), loss_age_plt, color='blue', label='Age')
        plt.plot(range(args.epochs), loss_gen_plt, color='green', label='Gender')
        plt.plot(range(args.epochs), loss_eth_plt, color='red', label='Ethnicity')
        plt.xlabel('Epoch')
        plt.ylabel('train_loss')
        plt.legend()
        plt.xticks(range(0, args.epochs, 5))
        plt.savefig('./save/trainloss.png')

        plt.figure()
        plt.plot(range(args.epochs), acc_age_plt, color='blue', label='Age')
        plt.plot(range(args.epochs), acc_gen_plt, color='green', label='Gender')
        plt.plot(range(args.epochs), acc_eth_plt, color='red', label='Ethnicity')
        plt.xlabel('Epoch')
        plt.ylabel('train_acc')
        plt.legend()
        plt.xticks(range(0, args.epochs, 5))
        plt.savefig('./save/trainacc.png')

        test()
    
    elif args.datasets == 'mnist':
        plt.figure()
        plt.plot(range(args.epochs), torch.tensor(loss1_plt).cpu().numpy(), color='blue', label='Task1')
        plt.plot(range(args.epochs), torch.tensor(loss2_plt).cpu().numpy(), color='green', label='Task2')
        plt.xlabel('Epoch')
        plt.ylabel('train_loss')
        plt.legend()
        plt.xticks(range(0, args.epochs, 5))
        plt.savefig('./save/mnisttrainloss.png')

        plt.figure()
        plt.plot(range(args.epochs), torch.tensor(acc1_plt).cpu().numpy(), color='blue', label='Task1')
        plt.plot(range(args.epochs), torch.tensor(acc2_plt).cpu().numpy(), color='green', label='Task2')
        plt.xlabel('Epoch')
        plt.ylabel('train_acc')
        plt.legend()
        plt.xticks(range(0, args.epochs, 5))
        plt.savefig('./save/mnisttrainacc.png')

        w_save = copy.deepcopy(net_glob.state_dict())
        torch.save(w_save, 'model_params_mnist.pth')
        testmnist()
