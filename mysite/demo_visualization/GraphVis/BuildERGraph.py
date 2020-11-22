#-*- coding=utf-8 -*-
#@Time:  
#@Author: zjh
#@File: RelationNetwork.py
#@Software: PyCharm

# 有向图的顺序可能有点问题,但貌似是最初的文件就有问题
# 对于没有关系的实体暂时无法可视化

import json
import torch
import numpy as np
import networkx as nx
from GraphVis import NetworkVis
import matplotlib.pyplot as plt

from tqdm import tqdm

class RelationNetwork:
    def __init__(self,path_data_r,path_data_e,text):
        self.__rpath = path_data_r
        self.__epath = path_data_e
        self.__text = text
        self.__rel = "demo_visualization/data/relation_label_tag.txt"
        #self.__savepth = ""


    def __show_edge_info(self,edges):
        '''

        :param edges:  [['Frank\nG.\nZarb', 'NASD', 19, '/business/person/company']]
        :return: [['Frank G. Zarb', 'NASD', 19, '/business/person/company']]
        '''
        info = []
        for rel in edges:
            rel[0] = rel[0].replace("\n"," ")
            info.append(rel)
        return info
        # def save_path(self,save_path):
        #     self.__savepth = save_path

    def __append_dict(self,dict,key,value):
        if value not in dict.values():
            dict[key] = value

    def __search_rel_name(self,num):
        for r in self.__rel:
            if int(r.split()[1]) == num:
                break
        return r.split()[0][:-2]

    def __search_another_part(self,array,num):
        rtn = []
        for word in array:
            if word[1] == num:
                #print(word)
                rtn.append(word)

        return rtn

    def __add_one_edge(self,e1,e2):
        '''

        :param e1: the first entity-rel
        :param e2: the second entity-rel
        :return: edges consist of e1 and e2
        '''
        c1 = e1[0]        # 第一个实体
        c2 = e2[0]        # 第二个实体
        c3 = e1[1]        # 关系代号
        c4 = self.__search_rel_name(c3)   # 关系名称
        return list([c1,c2,c3,c4])


    def __node_process(self,triple,text):
        '''
        将单词进行合并
        合并说明：只有两种结构会被作为正确实体: 1 或者 (24*3)
        将关系三元组按照relation的顺序排序,从上往下遍历
        - 若1开头,则是单独的实体,放入array,游标word_id推进1
        - 若2开头,则是复合实体,游标word_id推进1 [2]
            - 循环到不是4结束,游标不断推进,此时应做边界处理,如果游标推导头后,两次break跳出两重循环 [4*]
            - 判断是不是3,如果是3的话组成正确的结构,将整体放入array[3],游标word_id推进1
        - 若0 3 4开头,则不是合理的实体结构,直接跳过本次循环
        注: 01234分别对应 O S B E I

        :param d_tuple: 原始的二维关系标签组
        :param text: 原文列表
        :return: (字符串标签,关系代号)二维数组,标签列表
        '''
        idex = np.lexsort([triple[:,1]])  # 按照关系代号排序
        #print(idex)
        triple = triple[idex, :]
        print("triple_sort*****\n",triple)
        array = []


        word_id = 0
        while word_id < len(triple):
            #print(triple[word_id])
            if triple[word_id][2] == 0 or \
            triple[word_id][2] == 3 or \
            triple[word_id][2] == 4:
                # 非实体和不合理标记跳过(predict中出现)
                word_id += 1
                continue

            elif triple[word_id][2] == 1:
                if [ text[triple[word_id][0]],triple[word_id][1] ] not in array:  # 防止一个实体重复出现
                    array.append([
                        text[triple[word_id][0]],
                        triple[word_id][1]
                    ])
                word_id += 1

            elif triple[word_id][2] == 2:
                print("here",triple[word_id])
                #print("here",text)
                tempcat = [text[triple[word_id][0]]]
                word_id += 1
                while triple[word_id][2] == 4: # 中间词
                    tempcat.append(text[triple[word_id][0]])
                    word_id += 1
                    if (word_id >= len(triple)): break  # 当识别末尾有问题时退出
                if (word_id >= len(triple)): break
                if triple[word_id][2] == 3: # 末尾词
                    print("here",triple[word_id])
                    tempcat.append(text[triple[word_id][0]])

                    # append代码缩进到if中,可以保证当上述过程中有任何一处情况不符合时不保存该实体
                    if [ "\n".join(tempcat),triple[word_id][1] ] not in array:
                        array.append([
                            "\n".join(tempcat),   # 换行是为了展示好看,vis.js不会给实体换行
                            triple[word_id][1]
                        ])
                    word_id += 1


            # 自己训练的结果234不一定按照这个结构,当不符合结构时应当设置抛出机制（等待设置）

        return array

    def __relation_process(self,array):
        '''
        抽取关系,转化为绘图所需的结构
        :param d_tuple: (字符串标签,关系代号)二维数组[按照标签序号排好序]
        :param rel 完整的关系,不同关系列表,一个关系一个字符串
        :return: (字符串1,字符串2,关系代号)
        '''
        edges = []

        for i in range(len(array)):  # 遍历array数组由奇找偶
            #print("now",array[i],end='')
            temp = array[i][1]
            #print("temp",temp)
            if temp % 2 == 1:
                another_part = self.__search_another_part(array, temp + 1)
                #print(another_part)
                for word in another_part:
                    edges.append(self.__add_one_edge(array[i],word))

        return edges


    def __eredges_process(self,edges,sen): #解开耦合,用不同句子区分关系
        er_edges = []
        for edge in edges:
            edge[3] = edge[3][1:].replace('/','/\n') + ':' + str(sen)
            #er_edges.append(edge)
            er_edges.append([edge[0],edge[3]])
            er_edges.append([edge[3],edge[1]])
        return er_edges


    def __getkey(self,d,value):
        return [k for k, v in d.items() if v == value][0]
        # 此处可以保证一个单词只对应一个node


    #@profile
    def lazy_vis(self):
        with open(self.__rpath, 'r') as f:
            relation_data = json.load(f)
        with open(self.__epath, 'r') as f:
            entity_data = json.load(f)
        with open(self.__rel, 'r') as f:
            self.__rel = f.read().splitlines()

        def vis(*args,label_show = True,save_pth = "../images/1.png"):
            #G = nx.DiGraph()
            # plot the networkx
            G = nx.MultiDiGraph()
            bar = tqdm(list(args))
            events = []     # Vis边可视化
            #rel_id_dict = {}  # 用于的添加

            for sen in bar:
                bar.set_description("Now get sen " + str(sen))
                entity_label = entity_data[sen]
                d_tuple = torch.tensor(relation_data[sen]).nonzero()
                d_tuple = d_tuple.numpy()
                triple = np.empty((0,3))    # 创建0行空数组
                for i in range(len(d_tuple)):
                    #print(d_tuple[i][0],d_tuple[i][1],entity_label[d_tuple[i][0]])
                    temp_entity = np.array([
                        int(d_tuple[i][0]),
                        int(d_tuple[i][1]),
                        int(entity_label[d_tuple[i][0]])
                    ]).reshape(1,3)
                    #print(temp_entity.shape,triple.shape)
                    triple = np.insert(temp_entity,0,values=triple,axis=0)
                #print("triple*********:\n",triple)
                text = self.__text[sen].strip().split()   # 成功访问到self？
                #gh_11.19 19:  print("text:**********:\n",text)
                array= self.__node_process(triple,text)
                #gh_11.19  print("array****\n",array)
                ret=[]
                edges = self.__relation_process(array)
                ret.append(edges)
                #gh_11.19 19: print("edges****\n",edges)
                edges = self.__eredges_process(edges,sen)
                #gh_11.19 19: print("er_edges****\n", edges)

                for e in edges:
                    G.add_edge(e[0],e[1])
                    #if list(e[0:2]) not in events:   # 只添加一次
                    #    events.append(list(e[0:2]))    # 添加事件
                    events.append(list(e[0:2]))
                ret.append(events)
            return ret

            # #pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes
            # #pos = nx.spring_layout(G, scale=3)           #标准布局
            # pos = nx.circular_layout(G)
            # nx.draw(G,pos,node_color = '#31ECA8')
            # for p in pos:  # raise text positions
            #     pos[p][1] += 0.07
            # nx.draw_networkx_labels(G, pos,font_size=10)


            #edge_labels = {}   # 用来整合edge_dict信息
            # if label_show == True:
            #     edge_dict = nx.get_edge_attributes(G, 'name')
            # else:
            #     edge_dict = nx.get_edge_attributes(G, 'ind')
            #
            # for k,v in edge_dict.items():
            #     if k[-1] == 0:
            #         edge_labels[k[:2]] = str(v)
            #     else:
            #         edge_labels[k[:2]] = edge_labels[k[:2]] + '\n' + str(v)
            #print(edge_dict,edge_labels)


            # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            # plt.savefig(save_pth)
            #plt.show()
        return vis


# C = RelationNetwork('test_relation_part.json',
#                     'test_label_part.json',
#                     'test_text.json',
#                     'relation_label_tag.txt')
# f = C.lazy_vis()
# # temp = list(range(10))
# # events = f(*temp)
# events = f(267)
# print("events****\n",events)
#
# G = NetworkVis.GraphShow()
# G.create_page(events)
