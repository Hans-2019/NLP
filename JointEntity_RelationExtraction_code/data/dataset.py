import os
import sys
import torch
from torch.utils import data
import json
import copy

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

# 对特定形式的数据集进行加载，并根据数据需求限制序列的长度
def load_text_data():
    with open(os.path.join(ABS_PATH, 'data/vocab_dic.json')) as f:
        vocab_dic=json.load(f)

    seq_len = 100
    vocab_size = len(vocab_dic)
    num_classes = 5
    num_relation_classes = 48
    data = {
        "X_train": [],
        "entity_train": [],
        "relation_train": [],
        "X_test": [],
        "entity_test": [],
        "relation_test": [],
        "X_valid": [],
        "entity_valid": [],
        "relaion_valid": [],
        "num_classes": num_classes,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "num_relation_classes": num_relation_classes
    }
    
    with open(os.path.join(ABS_PATH, "data/test_text_data.json")) as f:
        test_X = json.load(f)
    with open(os.path.join(ABS_PATH, "data/test_label.json")) as f:
        test_Y = json.load(f)
    with open(os.path.join(ABS_PATH, "data/test_relation.json")) as f:
        test_Z = json.load(f)
    
    with open(os.path.join(ABS_PATH, "data/train_text_data.json")) as f:
        train_X = json.load(f)
    with open(os.path.join(ABS_PATH, "data/train_label.json")) as f:
        train_Y = json.load(f)
    with open(os.path.join(ABS_PATH, "data/train_relation.json")) as f:
        train_Z = json.load(f)

    with open(os.path.join(ABS_PATH, "data/valid_text_data.json")) as f:
        valid_X = json.load(f)
    with open(os.path.join(ABS_PATH, "data/valid_label.json")) as f:
        valid_Y = json.load(f)
    with open(os.path.join(ABS_PATH, "data/valid_relation.json")) as f:
        valid_Z = json.load(f)
    
    def convert_len(lis):
        result_lis=[]
        for i in lis:
            if len(i) >= seq_len:
                i=i[:seq_len]
            else:
                i.extend([0] * (seq_len - len(line)))
            result_lis.append(i)
        return torch.tensor(result_lis)



    data['X_train']=convert_len(train_X)
    data['entity_train']=convert_len(train_Y)
    data['relation_train']=convert_len(train_Z)
    data['X_test']=convert_len(test_X)
    data['entity_test']=convert_len(test_Y)
    data['relation_test']=convert_len(test_Z)
    data['X_valid']=convert_len(valid_X)
    data['entity_valid']=convert_len(valid_Y)
    data['relation_valid']=convert_len(valid_Z)

    data["vocab_size"] = vocab_size +1

    return data

# 用dataloader加载数据
class Dataget():
   
    def __init__(self):
        data = load_text_data()
        self.train_data = Dataget_sub(data['X_train'], data['entity_train'], data['relation_train'])
        self.test_data = Dataget_sub(data['X_test'], data['entity_test'], data['relation_test'])
        self.valid_data = Dataget_sub(data['X_valid'], data['entity_valid'], data['relation_valid'])
        self.num_classes = data['num_classes']
        self.vocab_size = data['vocab_size']
        self.seq_len = data['seq_len']
        self.num_relation_classes = data['num_relation_classes']
    def _print_config(self):
        # 打印数据集信息  
        print('Dataset config:')
        print("\tnum_entity_classes\t" + str(self.num_classes))
        print("\tvocab_size\t" + str(self.vocab_size))
        print("\tseq_len \t" + str(self.seq_len))
        print("\tnum_relation_classes\t" + str(self.num_relation_classes))

# 组织为特定的输出形式
class Dataget_sub(data.Dataset):

    def __init__(self, X, target, target2):
        self.X = X
        self.target = target
        self.target2 = target2

    def __getitem__(self, index):
       '''
       返回一个sample的数据
       '''
       data = self.X[index]
       label = self.target[index]
       relation_classes =self.target2[index] 
       return data, label, relation_classes

    def __len__(self):
       '''
       返回数据集中所有样本的个数
       '''
       return len(self.X)