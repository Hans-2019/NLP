from django.shortcuts import HttpResponse, render, redirect
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from demo_visualization.Test import test_er_single, test_er_batch


def show_edge_info(edges):
    '''
    :param edges:  [['Frank\nG.\nZarb', 'NASD', 19, '/business/person/company']]
    :return: [['Frank G. Zarb', 'NASD', 19, '/business/person/company']]
    '''
    
    info = []
    if len(edges)==0:
        return info
    
    for rel in edges:
        rel[0] = rel[0].replace("\n", " ")
        rel[-1] = rel[-1].replace("\n", " ")
        info.append(rel)
    return info


def apply(request):
    if request.method == "GET":
        return render(request, 'app02/apply.html')

    else:
        sentence = request.POST.get('sentence')
        if sentence == '' or sentence == '\r\n' or sentence == None:
            return render(request, "app02/apply.html", {'warn': "pls input a sentence"})
        else:

            s_list = list(sentence.split("\n"))

            events = test_er_single.test(s_list[0])
            info = show_edge_info(events)
            return render(request, "app02/graph_show.html", {'aft': info, 'bfr': sentence})

            '''
            a_sentence = sentence+"-----"
            # 上面是测试用代码
            return render(request, "app02/apply.html", {'aft': a_sentence, 'bfr': sentence})
            '''


def home(request):
    return render(request, 'app02/2home_page.html')


def graph(request):
    return render(request, 'app02/graph_show2.html')