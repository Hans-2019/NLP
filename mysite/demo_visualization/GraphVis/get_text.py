# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 21:07:16 2020

@author: zjh
"""
import json

with open('E:\\科研小助手\\Joint Entity and Relation Extraction\\EntityRelation\\NYT\\raw_valid.json', 'r') as f:
    datas = []
    for line in f.readlines():
        data = json.loads(line)['sentText'] 
        datas.append(data)

with open('E:\\大三上2020秋\\1 现代程序设计技术\\Homework\\Week9\\data\\valid_text.json','w') as f:
    json.dump(datas, f)