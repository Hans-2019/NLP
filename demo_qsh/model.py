# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:11:57 2020

@author: Qian Sihan
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModule(nn.Module):
    '''
    封装了nn.Module，主要提供save和load两个方法
    '''

    def __init__(self,opt=None):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self)) # 模型的默认名字

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名，
        如AlexNet_0710_23:57:29.pth
        '''
        if name is None:
            prefix = self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name
    
class Attention_Layer(nn.Module):
    
    def __init__(self, lstm_hidden_size, bias=True):
        super(Attention_Layer, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, hidden_vector_sequence, tag_sequence):
        '''
        hidden_vector_sequence: (batch, seq_len, lstm_hidden_size)
        tag_sequence: (batch, seq_len)
        '''
        A=torch.Tensor(1,hidden_vector_sequence.size(1),hidden_vector_sequence.size(2))
        for i in torch.arange(hidden_vector_sequence.size(0)):
            H = hidden_vector_sequence[i]
            He = H[tag_sequence[i].int()!=0]
            Ho = H[tag_sequence[i].int()==0]
            Ae = self.softmax(torch.matmul(H,He.t()))
            Ao = self.softmax(torch.matmul(H,Ho.t()))
            A = torch.cat([A,(torch.matmul(Ae,He)+torch.matmul(Ao,Ho)+H).unsqueeze(0)],dim=0)
        A = A[torch.arange(A.size(0))!=0]
        return A
    #A: (batch, seq_len, lstm_hidden_size)
    
class LSTM_clf(BasicModule):
    def __init__(self,
        vocab_size,
        label_size,
        emb_size,
        lstm_hidden_size,
        lstm_hidden_layer,
        bidirectional,
        attention,
        dropout,
    ):
        super(LSTM_clf, self).__init__()
        
        self.bidirectional = bidirectional
        self.attention = attention
        
        #嵌入层
        self.Embedding = nn.Embedding(vocab_size, emb_size)
        
        #LSTM层
        self.LSTM = nn.LSTM(
            emb_size,
            lstm_hidden_size,
            lstm_hidden_layer,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True
        )
        
        #全连接层与注意力层
        if self.bidirectional:
            self.out = nn.Linear(lstm_hidden_size * 2, label_size)
            if self.attention:
                self.Attention_Layer = Attention_Layer(lstm_hidden_size *2)
        else:
            self.out = nn.Linear(lstm_hidden_size, label_size)
            if self.attention:
                self.Attention_Layer = Attention_Layer(lstm_hidden_size)

        #激活层
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):
        embeddings = self.Embedding(input)
        #(batch, seq_len, emb_size)
        
        x, (hn, cn) = self.LSTM(embeddings)
        #x: (batch, seq_len, lstm_hidden_size*num_directions)
        result=self.out(x)
        tag_sequence = torch.argmax(result, dim=2)
        if self.attention:
            A = self.Attention_Layer(x,tag_sequence)
        
        return (A,result)
        #{'Attention':A,'label_sequence':result}
        #A: (batch, seq_len, lstm_hidden_size*num_directions)
        #result: (batch, seq_len, label_size)
        
class LSTM_relation_block(LSTM_clf):
    def __init__(self,
        vocab_size,
        label_size,
        emb_size,
        lstm_hidden_size,
        lstm_hidden_size2,
        lstm_hidden_layer,
        lstm_hidden_layer2,
        bidirectional,
        bidirectional2,
        attention,
        dropout,
        relation_size
    ):
        super(LSTM_relation_block,self).__init__(vocab_size,label_size,emb_size,lstm_hidden_size,lstm_hidden_layer,bidirectional,attention,dropout)
        self.bidirectional2=bidirectional2
        self.lstm_hidden_size2=lstm_hidden_size2
        self.lstm_hidden_layer2=lstm_hidden_layer2
        self.relation_size=relation_size
        #LSTM层
        if self.bidirectional:
            self.LSTM = nn.LSTM(
                emb_size+lstm_hidden_size*2,
                lstm_hidden_size2,
                lstm_hidden_layer2,
                bidirectional=bidirectional2,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.LSTM = nn.LSTM(
                emb_size+lstm_hidden_size,
                lstm_hidden_size2,
                lstm_hidden_layer2,
                bidirectional=bidirectional2,
                dropout=dropout,
                batch_first=True)
        #全连接层
        if self.bidirectional2:
            self.out=nn.Linear(2*lstm_hidden_size2, relation_size)
        else:
            self.out=nn.Linear(lstm_hidden_size2, relation_size)
            
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input, A):
        embeddings = self.Embedding(input)
        #(batch, seq_len, emb_size)
        
        x=torch.cat([embeddings, A], dim=2)
        
        h, (hn, cn) = self.LSTM(x)
        # h: (batch, seq_len, lstm_hidden_size2*num_directions2)
        return self.sigmoid(self.out(h))
        #(batch, seq_len, relation_size)
































