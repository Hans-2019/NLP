import json
import torch
import sys
sys.path.append(r'demo_visualization')


from Test.model import LSTM_clf, LSTM_relation_block
from GraphVis import BuildERGraph, NetworkVis,networkvis2

with open('demo_visualization/data/vocab.txt', 'r') as f:
    gg = f.readlines()

vocab = {}
vocab[' '] = 0
num = 0
for i in gg:
    num += 1
    i.strip()
    a, b = i.split()
    vocab[a.lower()] = num
vocab_set = set(vocab.keys())
vocab_num = len(vocab.keys())


def text_data_process(line):
    a = []
    line_words = line.strip().split()
    for j in line_words:
        j = j.lower()
        if j not in vocab_set:
            a.append(vocab_num)
        else:
            a.append(vocab[j])
    a.extend([0] * (100 - len(a)))
    return torch.tensor(a)


def model_forward(use_data):
    '''

    :param use_data:单个特定的句子
    :return: predicted - label;predicted2 - relation
    '''
    input = torch.unsqueeze(text_data_process(use_data), 0)
    model = LSTM_clf(
        vocab_size=47465,
        label_size=5,
        emb_size=200,
        lstm_hidden_size=200,
        lstm_hidden_layer=2,
        bidirectional=True,
        attention=True,
        dropout=0
    )
    model2 = LSTM_relation_block(
        vocab_size=47465,
        label_size=5,
        emb_size=200,
        lstm_hidden_size=200,
        lstm_hidden_size2=200,
        lstm_hidden_layer=2,
        lstm_hidden_layer2=2,
        bidirectional=True,
        bidirectional2=True,
        attention=True,
        dropout=0,
        relation_size=48
    )
    model.eval()
    model2.eval()
    # model.load('best_model.pth',map_location=torch.device('cpu')) #此时需要指定存储好的模型位置
    # model2.load('best_model2.pth',map_location=torch.device('cpu'))

    model.load_state_dict(torch.load('demo_visualization/data/best_model.pth', map_location=torch.device('cpu')))
    model2.load_state_dict(torch.load('demo_visualization/data/best_model2.pth', map_location=torch.device('cpu')))

    Attention_tensor, output = model(input)
    output2 = model2(input, Attention_tensor)
    predicted = torch.argmax(output, dim=2)
    predicted2 = torch.where(output2 > 0.5, torch.tensor(1), torch.tensor(0))
    # print(predicted,predicted2.numpy().tolist())

    predicted = predicted.numpy().tolist()
    predicted2 = predicted2.numpy().tolist()
    # insert 0 to each top ,change from 48 to 49
    for sen in range(len(predicted2)):
        for word in range(len(predicted2[sen])):
            predicted2[sen][word].insert(0, 0)
    return predicted, predicted2


def test(use_data2):
    '''
    text_data: 输入标准格式的文本,每一句话为list的一个元素
    :return: 可视化结果html
    '''
    use_data = "Crucial to the team 's success was the recruitment of an international roster of stars , led by the exuberant Pele -LRB- coaxed out of retirement with the help of Secretary of State Henry Kissinger and untold millions -RRB- and Germany 's smooth-as-silk Franz Beckenbauer ."
    use_data = "www.formula1.com August Aug. 1-5 National Corvette Restorers Society Annual Convention , Henry B. Gonzalez Convention Center , San Antonio ."
    use_data = "The final deal was brokered through the major assistance of Annette L. Nazareth , an S.E.C. commissioner who once led its market regulation office , and Frank G. Zarb , the former chairman of NASD and a major presence on Wall Street and in Washington for much of his career ."

    predicted, predicted2 = model_forward(use_data2)

    with open('predicted_label.json', 'w') as f:
        json.dump(predicted, f)
    with open('predicted_relation.json', 'w') as f:
        json.dump(predicted2, f)

    C = BuildERGraph.RelationNetwork(
        'predicted_relation.json',
        'predicted_label.json',
        [use_data2])
    f = C.lazy_vis()
    # temp = list(range(10))
    # events = f(*temp)
    ret = f(0)
    events=ret[1]
    #gh_11.19 15:09-- print(events)

    G = NetworkVis.GraphShow()
    G.create_page(events)
    
    G2=networkvis2.GraphShow()
    G2.create_page(events)
    return ret[0]










