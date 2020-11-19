# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:38:58 2020

@author: Qian Sihan
"""

import os
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import LSTM_clf,LSTM_relation_block
from config import opt
from evaluate import Batch_metrics
from data import Dataget

device = torch.device("cuda:{}".format(opt.device_num) if (opt.use_cuda and torch.cuda.is_available()) else "cpu")

#训练模型，过程中会进行验证


def relation_loss(output, target, target_label, alpha_relation):
    '''
    output,target:(batch, seq_len, relation_size)
    '''
    output=output.double()
    log_num=torch.where(output.double()<(1e-45),torch.tensor(1e-45).double().to(device),output.double())
    log_num2=torch.where(output.double()>(1-(1e-45)),torch.tensor(1-(1e-45)).to(device).double(),output.double())
    Loss=torch.log(log_num)*target + torch.log(1-log_num2)*(1-target)
    Loss1=-torch.sum(Loss[target_label.int()==0])
    Loss2=-torch.sum(Loss[target_label.int()!=0])
    return (Loss1+alpha_relation*Loss2)/(output.size(0)*output.size(1))
    #return (alpha_relation*Loss2)/(output.size(0)*output.size(1))


def train():
    opt._print_config()
    
    #数据集加载
    data = Dataget()
    data._print_config()
    
    #dataloader加载
    train_dataloader = DataLoader(
        data.train_data,
        opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers
    )
    val_dataloader = DataLoader(
        data.valid_data,
        opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )
    
    #模型设置
    model = LSTM_clf(
        vocab_size=data.vocab_size,
        label_size=data.num_classes,
        emb_size=opt.emb_size,
        lstm_hidden_size=opt.lstm_hidden_size,
        lstm_hidden_layer=opt.lstm_hidden_layer,
        bidirectional=opt.bidirectional,
        attention=opt.attention,
        dropout=opt.dropout
    )
    model.to(device)
    model2 = LSTM_relation_block(
        vocab_size=data.vocab_size,
        label_size=data.num_classes,
        emb_size=opt.emb_size,
        lstm_hidden_size=opt.lstm_hidden_size,
        lstm_hidden_size2=opt.lstm_hidden_size2,
        lstm_hidden_layer=opt.lstm_hidden_layer,
        lstm_hidden_layer2=opt.lstm_hidden_layer2,
        bidirectional=opt.bidirectional,
        bidirectional2=opt.bidirectional2,
        attention=opt.attention,
        dropout=opt.dropout,
        ####################还没创建
        relation_size=data.num_relation_classes
        #####################还没创建
    )
    model2.to(device)
    
    #目标函数和优化器设置
    criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion2 = relation_loss

    optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr = opt.learning_rate,
                    weight_decay = opt.weight_decay
                )

    optimizer2 = torch.optim.Adam(
                model2.parameters(),
                lr = opt.learning_rate,
                weight_decay = opt.weight_decay
            )


    #统计指标设置
    best_model_dic = {
        'best_epoch':0,
        'best_valid_acc':0
        }
    
    #开始训练
    print('开始训练---------------------------------')
    for epoch in range(opt.max_epoch):
        te_start = time.time()
        train_loss_list = []
        
        #训练模型
        B_metrics1 = Batch_metrics()
        B_metrics2 = Batch_metrics()

        B_metrics1_all = Batch_metrics()
        B_metrics2_all = Batch_metrics()

        t = tqdm(train_dataloader)
        for (input, target, target2) in t:
            input = input.to(device)
            target = target.to(device)
            target2 = target2.to(device)
            optimizer.zero_grad() # 每个batch清空梯度，必做
            optimizer2.zero_grad()
            Attention_tensor, output = model(input)
            output2 = model2(input, Attention_tensor)
            predicted = torch.argmax(output, dim=2)
            #predicted: (batch_size, seq_len)

            predicted2 = torch.where(output2>0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
            #predicted2: (batch_size, seq_len, num_relation)

            predicted3 = torch.where(predicted2.int()==target2.int(), torch.tensor(1).to(device), torch.tensor(0).to(device))
            predicted3 = torch.sum(predicted3, dim=2)
            predicted3 = torch.where(predicted3.int()==data.num_relation_classes, torch.tensor(1).to(device), torch.tensor(0).to(device))
            target3 = torch.where(predicted3==predicted3, torch.tensor(1).to(device), torch.tensor(0).to(device))
            target4 = torch.sum(target2,dim=2)!=0 #relation是否为空
            target44 = target.int() != 0 #entity是否为空
            predicted5=predicted3[target44]
            target5=target3[target44]

            predicted11=predicted[target44]
            target11=target[target44]


            B_metrics1.add_batch(predicted11.view(-1).cpu(), target11.view(-1).cpu())
            
            B_metrics2.add_batch(predicted5.view(-1).cpu(), target5.view(-1).cpu())

            B_metrics1_all.add_batch(predicted.view(-1).cpu(), target.view(-1).cpu())
            B_metrics2_all.add_batch(predicted3.view(-1).cpu(), target3.view(-1).cpu())
            #output: (batch_size, seq_len, label_size)
            #target: (batch_size, seq_len)
            output_entity1 = output[target.int()==0] #(batch_size * seq_len, label_size)
            target_entity1 = target[target.int()==0] #(batch_size * seq_len)
            output_entity2 = output[target.int()!=0]
            target_entity2 = target[target.int()!=0]
            loss_entity= (criterion(output_entity1,target_entity1) + opt.alpha_entity * criterion(output_entity2, target_entity2))/(output.size(0)*output.size(1))


            loss_train = loss_entity + criterion2(output2,target2, target, opt.alpha_relation) #计算loss
            train_loss_list.append(float(loss_train))
            t.set_postfix(loss=np.mean(train_loss_list))
            
            loss_train.backward() #计算梯度
            optimizer.step() #参数更新
            optimizer2.step()
            


        #计算验证集上的指标
        train_metrics1=B_metrics1.cal_metrics()
        train_metrics2=B_metrics2.cal_metrics()
        train_metrics1_all=B_metrics1_all.cal_metrics()
        train_metrics2_all=B_metrics2_all.cal_metrics()

        valid_metrics1,valid_metrics2 = val(model,model2, val_dataloader, data.num_relation_classes)

        valid_acc = (valid_metrics1[0]+9 * valid_metrics2[0])/10
        if valid_acc >= best_model_dic['best_valid_acc']:
            best_model_dic['best_valid_acc'] = valid_acc
            best_model_dic['best_epoch'] = epoch

            #当需要测试与线上使用时的时候需要在合适位置存储最优的模型
            model.save(opt.load_model_path)
            model2.save(opt.load_model2_path)

        train_metrics1_print="{:.4f} ".format(train_metrics1[0])
        train_metrics2_print="{:.4f} ".format(train_metrics2[0])
        train_metrics1_all_print="{:.4f} ".format(train_metrics1_all[0])
        train_metrics2_all_print="{:.4f} ".format(train_metrics2_all[0])


        valid_metrics1_print="{:.4f}  ".format(valid_metrics1[0])
        valid_metrics2_print="{:.4f} ".format(valid_metrics2[0])
        print("epoch: {}, loss: {:.4f}, time: {:.2f}".format(epoch, np.mean(train_loss_list), time.time()-te_start))
        print("                      acc")
        print("train_entity:       {}".format(train_metrics1_print))
        print("train_relation:     {}".format(train_metrics2_print))
        print("train_entity_all:   {}".format(train_metrics1_all_print))
        print("train_relation_all: {}".format(train_metrics2_all_print))
        print("valid_entity:       {}".format(valid_metrics1_print))
        print("valid_relation:     {}\n".format(valid_metrics2_print))

    print("best_epoch: {}, best_valid_acc: {:.4f}".format(
            best_model_dic['best_epoch'],
            best_model_dic['best_valid_acc']
        ))


#验证模型
def val(model,model2,dataloader,num_relation_classes):
    #把模型设为验证模式
    model.eval()
    model2.eval()
    B_metrics1 = Batch_metrics()
    B_metrics2 = Batch_metrics()
    t = tqdm(dataloader)
    for(input, target, target2) in t:
        input = input.to(device)
        target = target.to(device)
        target2 = target2.to(device)
        Attention_tensor, output = model(input)
        output2 = model2(input, Attention_tensor)
        predicted = torch.argmax(output, dim=2)
        predicted2 = torch.where(output2>0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))

        predicted3 = torch.where(predicted2.int()==target2.int(), torch.tensor(1).to(device), torch.tensor(0).to(device))
        predicted3 = torch.sum(predicted3, dim=2)
        predicted3 = torch.where(predicted3.int()==num_relation_classes, torch.tensor(1).to(device), torch.tensor(0).to(device))
        target3 = torch.where(predicted3==predicted3, torch.tensor(1).to(device), torch.tensor(0).to(device))

        target4 = torch.sum(target2,dim=2)!=0
        target44 = target.int() != 0
        predicted5=predicted3[target44]
        target5=target3[target44]

        predicted11=predicted[target44]
        target11=target[target44]

        B_metrics1.add_batch(predicted11.view(-1).cpu(), target11.view(-1).cpu())
        B_metrics2.add_batch(predicted5.view(-1).cpu(), target5.view(-1).cpu())
    model.train()
    model2.train()

    return (B_metrics1.cal_metrics(),B_metrics2.cal_metrics())


def test():
    opt._print_config()

    data=Dataget()
    data._print_config()

    test_dataloader = DataLoader(
        data.test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )

    model = LSTM_clf(
        vocab_size=data.vocab_size,
        label_size=data.num_classes,
        emb_size=opt.emb_size,
        lstm_hidden_size=opt.lstm_hidden_size,
        lstm_hidden_layer=opt.lstm_hidden_layer,
        bidirectional=opt.bidirectional,
        attention=opt.attention,
        dropout=opt.dropout
    )
    model2 = LSTM_relation_block(
        vocab_size=data.vocab_size,
        label_size=data.num_classes,
        emb_size=opt.emb_size,
        lstm_hidden_size=opt.lstm_hidden_size,
        lstm_hidden_size2=opt.lstm_hidden_size2,
        lstm_hidden_layer=opt.lstm_hidden_layer,
        lstm_hidden_layer2=opt.lstm_hidden_layer2,
        bidirectional=opt.bidirectional,
        bidirectional2=opt.bidirectional2,
        attention=opt.attention,
        dropout=opt.dropout,
        relation_size=data.num_relation_classes
    )
    model.eval()
    model2.eval()
    model.load(opt.load_model_path) #此时需要指定存储好的模型位置
    model2.load(opt.load_model2_path) 
    model.to(device)
    model2.to(device)
        
    B_acc1 = Batch_metrics()
    B_acc2 = Batch_metrics()
    B_acc1_all = Batch_metrics()
    B_acc2_all = Batch_metrics()

    t = tqdm(test_dataloader)
    for (input, target, target2) in t:
        input=input.to(device)
        target=target.to(device)
        target2=target2.to(device)
        Attention_tensor, output = model(input)
        output2 = model2(input, Attention_tensor)
        predicted = torch.argmax(output, dim=2)
        predicted2 = torch.where(output2>0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))

        predicted3 = torch.where(predicted2.int()==target2.int(), torch.tensor(1).to(device), torch.tensor(0).to(device))
        predicted3 = torch.sum(predicted3, dim=2)
        predicted3 = torch.where(predicted3.int()==data.num_relation_classes, torch.tensor(1).to(device), torch.tensor(0).to(device))
        target3 = torch.where(predicted3==predicted3, torch.tensor(1).to(device), torch.tensor(0).to(device))
        target44=target.int()!=0
        predicted5=predicted3[target44]
        target5=target3[target44]

        predicted11=predicted[target44]
        target11=target[target44]



        B_acc1.add_batch(predicted11.view(-1).cpu(), target11.view(-1).cpu())
        B_acc2.add_batch(predicted5.view(-1).cpu(), target5.view(-1).cpu())

        B_acc1_all.add_batch(predicted.view(-1).cpu(), target.view(-1).cpu())
        B_acc2_all.add_batch(predicted3.view(-1).cpu(), target3.view(-1).cpu())

    test_metrics1 = B_acc1.cal_metrics()
    test_metrics2 = B_acc2.cal_metrics()

    test_metrics1_all = B_acc1_all.cal_metrics()
    test_metrics2_all = B_acc2_all.cal_metrics()


    test_metrics1_print="{:.4f} ".format(test_metrics1[0])
    test_metrics2_print="{:.4f}".format(test_metrics2[0])
    test_metrics1_all_print="{:.4f}".format(test_metrics1_all[0])
    test_metrics2_all_print="{:.4f}".format(test_metrics2_all[0])


    print("                     acc")
    print("test_entity:       {}".format(test_metrics1_print))
    print("test_relation:     {}".format(test_metrics2_print))
    print("test_entity_all:   {}".format(test_metrics1_all_print))
    print("test_relation_all: {}\n".format(test_metrics2_all_print))


if __name__=='__main__':
    train()
    # test()
        
        
        
        
        
        
        
    
    
    