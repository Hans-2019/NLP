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
    <html>
    <head>
    <script type="text/javascript" src="/static/dist/vis.js"></script>
    <link href="/static/dist/vis.css" rel="stylesheet" type="text/css">
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    </head>
    <body>

    <div id="VIS_draw"></div>

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
        f = open('HTML/app02/graph_show2.html', 'w+')
        #print(self.base)
        html = self.base.replace('data_nodes', str(data_nodes)).replace('data_edges', str(data_edges))
        #print(html)
        f.write(html)
        f.close()


# events = [['Iceland', 'Reykjavik'], ['Iceland', 'Reykjavik']]
# G = GraphShow()
# G.create_page(events)