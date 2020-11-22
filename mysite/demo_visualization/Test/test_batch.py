import json
import torch
from Test.model import LSTM_clf,LSTM_relation_block
from GraphVis import BuildNetwork,NetworkVis


with open('../data/vocab.txt','r') as f:
    gg=f.readlines()

vocab={}
vocab[' ']=0
num=0
for i in gg:
    num+=1
    i.strip()
    a,b=i.split()
    vocab[a.lower()]=num
vocab_set=set(vocab.keys())
vocab_num=len(vocab.keys())

def text_data_process(line):

    a=[]
    line_words=line.strip().split()
    for j in line_words:
        j=j.lower()
        if j not in vocab_set:
            a.append(vocab_num)
        else:
            a.append(vocab[j])
    a.extend([0]*(100-len(a)))
    return torch.tensor(a)

def model_forward(use_data):
    '''

    :param use_data:单个特定的句子
    :return: predicted - label;predicted2 - relation
    '''
    input=torch.unsqueeze(text_data_process(use_data),0)
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

    model.load_state_dict(torch.load('../data/best_model.pth', map_location=torch.device('cpu')))
    model2.load_state_dict(torch.load('../data/best_model2.pth', map_location=torch.device('cpu')))

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
    return predicted,predicted2


def test(text_data):
    '''
    text_data: 输入标准格式的文本,每一句话为list的一个元素
    :return: 可视化结果html
    '''
    label,relation = [],[]

    for use_data in text_data:
        predicted,predicted2 = model_forward(use_data)

        label.extend(predicted)
        relation.extend(predicted2)
        #print(f"finish{use_data}")

    with open('predicted_label.json', 'w') as f:
    	json.dump(label,f)
    with open('predicted_relation.json', 'w') as f:
    	json.dump(relation,f)


    C = BuildNetwork.RelationNetwork(
        'predicted_relation.json',
        'predicted_label.json',
        text_data)
    f = C.lazy_vis()

    temp = list(range(6))
    events = f(*temp)

    #events = f(0)
    print(events)


    G = NetworkVis.GraphShow()
    G.create_page(events)


if __name__ == "__main__":
    text_data = [
        "But that spasm of irritation by a master intimidator was minor compared with what Bobby Fischer , the erratic former world chess champion , dished out in March at a news conference in Reykjavik , Iceland .",
        "TAR HEELS WIN WOMEN 'S TITLE -- Casey Nogueira , who graduated from high school a year early to join the storied North Carolina program this fall , had a goal and an assist yesterday to lead the Tar Heels past Notre Dame , 2-1 , in the Women 's College Cup title game in Cary , N.C. This was North Carolina 's 18th championship in the 25-year history of the Women 's College Cup .",
        "Kyoto 's cuisine -LRB- kyo-ryori -RRB- is the legacy of court and temple , aristocratic and understated ; unusual locations and exquisite food made the meals I ate there some of the most magical of my life , and made the traditional Japan I 'd first fallen in love with palpable .",
        "Mr. McClellan and other administration spokesmen said they had no concrete evidence of Syria 's involvement in the killing of Mr. Hariri , a prominent opposition leader and critic of Syria 's role in Lebanon , who died along with at least 11 others when a car bomb blew up next to his motorcade in Beirut .",
        "But Schaap seems as comfortable in that role as Joe Buck , the Fox baseball and football sportscaster who so clearly benefited from learning beside his father , Jack Buck , the late voice of the St. Louis Cardinals . ''",
        "www.formula1.com August Aug. 1-5 National Corvette Restorers Society Annual Convention , Henry B. Gonzalez Convention Center , San Antonio ."]
    test(text_data)

        
        
        
        
    
    
    