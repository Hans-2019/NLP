#-*- coding=utf-8 -*-
#@Time:  
#@Author: zjh
#@File: NetworkVis.py
#@Software: PyCharm
import sys
sys.path.append(r'mysite')


'''创建展示页面'''
class GraphShow():
    def __init__(self):
        self.base = '''
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>home</title>
    <link rel="styleSheet" href="/static/common.css" />
    <link rel="styleSheet" href="/static/btn1.css" />
    <link rel="styleSheet" href="/static/text1.css" />
    <link rel="styleSheet" href="/static/text2.css" />

    <link rel="styleSheet" href="/static/btn2.css" />
    
       <script type="text/javascript" src="/static/dist/vis.js"></script>
    <link href="/static/dist/vis.css" rel="stylesheet" type="text/css">
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
</head>



<body>

    <div style="float: right;padding-right: 2rem;">
        <a class="btn2_colored" href="/app01/home"> Home </a><a class="btn2_blank" href="/app01/intro">Intro</a><a class="btn2_blank" href="/app01/define">Definition</a><a class="btn2_blank" href="/app01/loss_func">LossFunc</a>
        <br>
    </div>
    </div>
    <div style="padding: 30px;background-color: #fff;">
    </div>

    <div class="center" style="padding-bottom: 20px;background: #4abdac;text-align: center;">
        <br>
        <div class="container  top_gray"></div>
        <h2 style="color: #fff">
            Joint Entity and Relation Extraction:
            <br> A Hierarchical Sequence Labeling Approach
        </h2>
        <h3 style="font-size:26px;font-family: 'Microsoft JhengHei';color: #fff;">Autumn 2020</h3>
        <hr>
        <p class="text_introduction center " style="padding-left:25%;padding-right: 25%;">a two-layer sequence model to identify entity and relation respectively, which decreases the number of tags<br></p>
    </div>
    <div class="center" style="padding-bottom: 10px;padding-top:8px;background-color: #f0f0f0; ">
        <p class=" text_introduction center">This is an application to show our researchs. You can input a sentence or a paragraph, then a triplet containing two entities along with their semantic relation will be given to show the relation in the sentence or the paragraph.<br></p>
    </div>


    <div style="width:100%;padding-bottom:30px">
        <div style="background-color: #f0f0f0;float: left ;width:50%;height:350px;">
            <form method='POST' action="/app01/home" class="tex_input22 center">

                <div style="font-size: 20px;padding-left:15%;padding-top:5%;width:80%;height:300px">
                    <textarea type="text " placeholder="Input a sentence or a paragraph " class="btn-border1 tex_input3 txa_tran" name="sentence">{{bfr}}</textarea>
                    <p style="font-size:15px">{{aft}}</p>
                </div>

        </div>


        <div id="VIS_draw" style="float: left;width:50%;height:350px;background-color:#f0f0f0"></div>
        
 <script type="text/javascript">
      var nodes = data_nodes;
      var edges = data_edges;

      var container = document.getElementById("VIS_draw");

      var data = {
        nodes: nodes,
        edges: edges
      };

      var options = {
          nodes: {
              shape: 'circle',
              size: 15,
              font: {
                  size: 15
              }
          },
          edges: {
              font: {
                  size: 100,
                  align: 'center'
              },
              color: 'red',
              arrows: {
                  to: {enabled: true, scaleFactor: 0.8}   /*原1.2*/
              },
              smooth: {enabled: true}
          },
          physics: {
              enabled: true
          }
      };

      var network = new vis.Network(container, data, options);

    </script>
    </div>

    <div style="padding-top: 20px;padding-bottom:20px;float:none;background-color:#f0f0f0" class="centered">

        <input type="submit" class="btn1 btn-border3" value="RUN" style="float: none;">
        <div style="padding-bottom: 30px">
            <p class="warning centered"> {{warn}} </p>
        </div>
    </div>
    </form>

    </div>

</body>

</html>

    '''
    '''读取文件数据'''
    def create_page(self, events):
        nodes = []
        for event in events:
            nodes.append(event[0])
            nodes.append(event[1])
        node_dict = {node: index for index, node in enumerate(nodes)}

        data_nodes = []
        data_edges = []
        for node, id in node_dict.items():
            data = {}
            data["group"] = 'Event'
            data["id"] = id
            data["label"] = node
            data_nodes.append(data)

        for edge in events:
            data = {}
            data['from'] = node_dict.get(edge[0])
            data['label'] = ''
            data['to'] = node_dict.get(edge[1])
            data_edges.append(data)

        self.create_html(data_nodes, data_edges)
        #print(data_nodes,data_edges)
        return

    '''生成html文件'''
    def create_html(self, data_nodes, data_edges):
        f = open('HTML/app01/home_graph_show.html', 'w+')
        #print(self.base)
        html = self.base.replace('data_nodes', str(data_nodes)).replace('data_edges', str(data_edges))
        #print(html)
        f.write(html)
        f.close()


# events = [['Iceland', 'Reykjavik'], ['Iceland', 'Reykjavik']]
# G = GraphShow()
# G.create_page(events)