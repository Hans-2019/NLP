# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:27:18 2020

@author: Qian Sihan
"""
class DefaultConfig():
    load_model_path = "best_model.pth"
    load_model2_path = "best_model2.pth"

    alpha_relation = 1000
    alpha_entity = 100


    emb_size = 200           # 嵌入维度
    lstm_hidden_size = 200   # LSTM隐层维度
    lstm_hidden_size2 = 200  # 关系块LSTM层数
    lstm_hidden_layer = 2    # LSTM层数
    lstm_hidden_layer2 = 2   # 关系块LSTM层数
    bidirectional = True     # 是否为双向
    bidirectional2 = True    # 关系块是否为双向
    attention = True         # 是否使用注意力（不使用可能会不收敛）
    dropout = 0.5            # dropout概率

    batch_size = 256         # 批次大小 256
    num_workers = 4          # 加载数据的进程
    max_epoch = 30           # 训练轮数
    learning_rate = 0.001    # 学习率 0.001

    weight_decay = 0.0001    # L2正则
    lr_decay = 0.95          # 当验证损失上升，学习率下降比例

    use_cuda = True          # 是否使用GPU
    device_num = 1           # GPU号

    def _print_config(self):
        # 打印配置信息  
        print('Use config:')
        for k, v in self.__class__.__dict__.items():
            if not (k.startswith('__') or k.startswith('_')):
                print("\t" + k + "\t" + str(getattr(self, k)))

opt = DefaultConfig()
