#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import sys
import os
from DatasetandModel import Automobile, HydraNet, HydraNet_mobilenet, LeNet, UTKFacesisa, ViT_MT_Model

import matplotlib
import torch


def delete_shard(data, shard_index):
    shard_size = len(data) // args.num_shards 
    start_index = shard_index * shard_size  
    end_index = start_index + shard_size  

    del data[start_index:end_index]
    
    return data

def compute_model_para_diff(model1_para_list, model2_para_list):
    diff = 0
    norm1 = 0
    norm2 = 0
    all_dot = 0
    for i in model1_para_list.keys():
        param1 = model1_para_list[i]
        param2 = model2_para_list[i]
        curr_diff = torch.norm(param1 - param2, p='fro')
        norm1 += torch.pow(torch.norm(param1, p='fro'), 2)
        norm2 += torch.pow(torch.norm(param2, p='fro'), 2)
        all_dot += torch.sum(param1 * param2)
        diff += curr_diff * curr_diff
    print('Diff Weight:{:.6f}'.format(torch.sqrt(diff)))
    return (all_dot / torch.sqrt(norm1 * norm2))


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from utils.sampling import UTKFace_iid
from utils.options import args_parser
# from models.Update import LocalUpdate
from train.facetrain import LocalUpdate, LocalUpdate2
from models.Update_unlearning import Local_Update

from models.Fed import FedAvg
from models.test import test_img



def compute_correlation_matrix2(parameters):

    layer_params = net_glob.net.fc3.parameters()
    fc_params = [param.detach().cpu().numpy().flatten() for param in layer_params]
    params = [param for name, param in net_glob.named_parameters() if 'fc' not in name]
    shared_params = [param.detach().cpu().numpy() for param in params]
    num_params_fc = len(fc_params)
    num_params_shared = len(shared_params)
    correlation_matrix = np.zeros((num_params_fc, num_params_shared))
    for i in range(num_params_fc):
        for j in range(num_params_shared):
            param1 = fc_params[i]
            param2 = shared_params[j]
            param1_reshaped = np.reshape(param1, (64, 3, 7, 7))
            correlation_coeff = np.corrcoef(param1_reshaped, param2.flatten())[0, 1]
            correlation_matrix[i, j] = correlation_coeff    

    return correlation_matrix
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
def compute_correlation_matrix_1(parameters1, parameters2):
    num_params = len(parameters1)
    correlation_matrix = np.zeros(num_params)
    for i in range(num_params):
        param1 = parameters1[i].detach().cpu().numpy().flatten()
        param2 = parameters2[i].detach().cpu().numpy().flatten()
        correlation_matrix[i] = np.mean(param1 - param2) * 100000
    return correlation_matrix

def train_test_split(dataset, train_ratio=0.8):
    torch.manual_seed(42) 
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = (dataset_size - train_size)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def test():
    net_glob.eval()
    test_img(net_glob, test_data, args)
    #attack(net_glob, test_data, args)
    print("testing is over")


if __name__ == '__main__':
    fw = open('loss.txt', 'w')
    fwacc = open('acc.txt', 'w')
    start_time = time.time()
    # parse args
    args = args_parser()
    ccm = np.zeros((72, 72))
    args.device = torch.device('cuda:0')

    train_dataset = Automobile('')
    train_data, test_data = train_test_split(train_dataset, train_ratio=0.8)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.bs, collate_fn=lambda x:x)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.bs, collate_fn=lambda x:x)
    dict_users = UTKFace_iid(train_dataset, args.num_users)
    net_glob = HydraNet().to(args.device)

    time_is = 0
    net_glob.train
    w_glob = net_glob.state_dict()
    w_init = copy.deepcopy(net_glob.state_dict())
    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    delete_round = args.delete_round
    delete_attr = args.delete_attrs
    loss_1_plt = []
    loss_2_plt = []
    acc_1_plt = []
    acc_2_plt = []
    
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    
    shard_index = args.shard_index 
    

    
    for iter in range(args.epochs):

        slice_index = iter + 1
        shard_index += 1

        acc_1_plt1 = []
        acc_2_plt1 = []

        net_glob.train()

        loss_locals_1 = []
        loss_locals_2 = []
        
        if iter < delete_round or iter > delete_round:
            iter_mat = args.delete_round - 2
            if iter==iter_mat and args.method == "unlearning":
                para1 = list(net_glob.parameters())
                parameters1 = copy.deepcopy(para1)
                for e in range(2):
                    for idx in idxs_users:
                        if idx == args.delete_client:
                            local = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_users[idx])
                            w, loss_1, loss_2, total_training_loss, acc_1, acc_2 = local.train(net=copy.deepcopy(net_glob).to(args.device), train_dataloader=train_dataloader)
                            if args.all_clients:
                                w_locals[idx] = copy.deepcopy(w)
                            else:
                                w_locals.append(copy.deepcopy(w))
                            loss_locals_1.append(copy.deepcopy(loss_1))
                            loss_locals_2.append(copy.deepcopy(loss_2))

                            acc_1_plt1.append(acc_1)
                            acc_2_plt1.append(acc_2)
                    w_glob = FedAvg(w_locals)
                    net_glob.load_state_dict(w_glob)
                    parameters2 = list(net_glob.parameters())
                    ccm = compute_correlation_matrix_1(parameters1, parameters2)
                    fw.write("ccm is: {}".format(ccm))
                        
            # local_total_training_loss = []
            if not args.all_clients:
                w_locals = []
            m = max(args.num_users, 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                if idx == 0:
                    args.now_round += 1
                local = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_users[idx])
                w, loss_1, loss_2, total_training_loss, acc_1, acc_2 = local.train(net=copy.deepcopy(net_glob).to(args.device), train_dataloader=train_dataloader)
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals_1.append(copy.deepcopy(loss_1))
                loss_locals_2.append(copy.deepcopy(loss_2))

                acc_1_plt1.append(acc_1)
                acc_2_plt1.append(acc_2)
            w_glob = FedAvg(w_locals)
            net_glob.load_state_dict(w_glob)
            loss_avg_1 = sum(loss_locals_1) / len(loss_locals_1)
            fw.write('Round {:3d}, Average loss of 1 : {:.3f}\n"'.format(iter, loss_avg_1))
            loss_1_plt.append(loss_avg_1)
            loss_avg_2 = sum(loss_locals_2) / len(loss_locals_2)
            fw.write('Round {:3d}, Average loss of 2 : {:.3f}\n'.format(iter, loss_avg_2))
            loss_2_plt.append(loss_avg_2)
            acc_avg_1 = sum(acc_1_plt1) / len(acc_1_plt1)
            acc_1_plt.append(acc_avg_1)
            acc_avg_2 = sum(acc_2_plt1) / len(acc_2_plt1)
            acc_2_plt.append(acc_avg_2)

            fw.write("round time is : {:.2f}s\n".format(time.time() - start_time))

            print("**now_round** is : ", args.now_round)
            print("=========================================", iter)
            torch.save(net_glob.state_dict(), f'/data/xia-group-12/hy/Rapid-Retraining/shard/shard{shard_index}_model.pth')

        elif iter == delete_round and (args.method == "unlearning" or args.method == "retrain"):
            parameters = list(net_glob.parameters())
            print("Delete Round is : ", iter, "the unlearning start")
            if delete_attr == 0:
                net_glob.freeze_1()
                # net_glob.finetune_1()
                fw.write("Delete Person\n")
            elif delete_attr == 1:
                net_glob.freeze_2()
                fw.write("Delete Vehicle\n")
            
            if args.method == "retrain":
                fw.write("the unlerning method is retrain\n")
                fw.write("round unlearning time is : {:.2f}s\n".format(time.time() - start_time))
                net_glob.load_state_dict(w_init)
                w_glob = net_glob.state_dict()
            elif args.method == "unlearning":
                fw.write("the unlerning method is unlearning\n")
                fw.write("round unlearning time is : {:.2f}s\n".format(time.time() - start_time))
                
                parameters = list(net_glob.parameters())
                threshold = 5
                for correlation, (name, param) in zip(ccm, net_glob.named_parameters()):
                    if abs(correlation) > threshold and 'fc' not in name:
                        adjustment = torch.tensor(random.uniform(-0.3, 0.3))
                        param.data += adjustment
                        param.requires_grad = False
                print("successful set resnet parameters")

    print("loss_age_plt is : ", loss_1_plt)
    print("loss_gen_plt is : ", loss_2_plt)
    print("acc_1_plt is : ", acc_1_plt)
    print("acc_2_plt is : ", acc_2_plt)

    w_save = copy.deepcopy(net_glob.state_dict())
    torch.save(w_save, 'model_params.pth')
    test()
