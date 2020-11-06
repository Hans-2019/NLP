# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 21:20:38 2020

@author: Qian Sihan
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import copy


#对batch求精度，predicted为列表的softmax值， target为整数值的标签
#先将所有的batch结果加入，再对整体计算精度

class Batch_metrics():
    def __init__(self):
        self.flag=0
    #predicted_list = np.array([], dtype=int)
    #target_list = np.array([], dtype=int)
    
    def add_batch(self, predicted, target):
        predicted = np.array(predicted, dtype=int)
        target = np.array(target,dtype=int)

        if self.flag==0:
            self.predicted_list=copy.deepcopy(predicted)
            self.target_list=copy.deepcopy(target)
            self.flag=1
        else:
            self.predicted_list = np.concatenate((self.predicted_list,predicted))
            self.target_list = np.concatenate((self.target_list, target))
        
    def cal_metrics(self):
        accuracy = accuracy_score(self.predicted_list,self.target_list)
        #macro_precision = precision_score(self.predicted_list, self.target_list, average='macro')
        #macro_recall = recall_score(self.predicted_list, self.target_list, average='macro')
        #macro_f1 = f1_score(self.predicted_list, self.target_list, average='macro')
        #kappa = cohen_kappa_score(self.predicted_list, self.target_list)
        #return (accuracy, macro_precision, macro_recall, macro_f1,kappa)
        return [accuracy]
    
    
    