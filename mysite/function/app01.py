from django.shortcuts import HttpResponse, render, redirect

from demo_visualization.Test import test_er_single,test_er_batch


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
        
    info=" ".join(str(id) for id in info)
    return info




def home(request):
    if request.method=="GET":
        return render(request, 'app01/home.html')
    
    else:
        sentence = request.POST.get('sentence')
        print(sentence)
        if sentence == '' or sentence == '\r\n' or sentence==None:
            return render(request, "app01/home.html", {'warn': "pls input a sentence"})
        else:
            s_list = list(sentence.split("\n"))
            
            events=test_er_single.test(sentence)
            info = show_edge_info(events)
            return render(request, "app01/home_graph_show.html",{'aft': info, 'bfr': sentence})



def intro(request):
    
    return render(request, 'app01/intro.html')
    


def define(request):
    return render(request, 'app01/define.html')



def lossf(request):

    return render(request, 'app01/lossf.html')
