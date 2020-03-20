"""句法分析"""
from pyltp import Segmentor, Postagger, Parser

# 实例初始化
segmentor = Segmentor()
postagger = Postagger()
parser = Parser()

# 模型路径
PATH = 'ltp_data_v3.4.0/'
CWS_MODEL = PATH + 'cws.model'  # 分词
POS_MODEL = PATH + 'pos.model'  # 词性标注
PARSER_MODEL = PATH + 'parser.model'  # 依存句法分析

# 读取模型
segmentor.load(CWS_MODEL)
postagger.load(POS_MODEL)
parser.load(PARSER_MODEL)

# 分词、标注、句法分析
text = '会议宣布了资深院士名单'
words = list(segmentor.segment(text))  # 分词
postags = list(postagger.postag(words))  # 词性标注
arcs = parser.parse(words, postags)  # 句法分析

# 释放模型
segmentor.release()
postagger.release()
parser.release()

"""networkx可视化"""
import networkx as nx, matplotlib.pyplot as mp

# 无多重边有向图
G = nx.DiGraph()

# [2, 0, 2, 5, 6, 2]
ah = [a.head for a in arcs]

# ['ROOT', '会议', '宣布', '了', '资深', '院士', '名单']
tree = ['ROOT'] + words

# 添加节点和边
for w in tree:
    G.add_node(w)
for i in range(len(ah)):
    j = ah[i]
    G.add_edge(words[i], tree[j])

# 可视化
mp.rcParams['font.sans-serif']=['SimHei']  # 用黑体显示中文
nx.draw(G, with_labels=True, node_color='lightgreen', font_size=20, node_size=2000, width=3, alpha=0.8)
mp.show()