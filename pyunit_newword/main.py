import re
import sys
import os
path = os.path.join(os.getcwd(), "..")
sys.path.append(path)
# from pyunit_newword import NewWords
from words import NewWords
from dao import Data, session


def get_data():
    data = session.query(Data).all()
    return data


if __name__ == '__main__':
    a = re.split('[^\u4e00-\u9fa50-9a-zA-Z]', "你的3大幅度wo4rds说#的ad是6多，少的sd")
    b = re.findall(r'[\u4e00-\u9fa50-9]', "你的3大幅度wo4rds说#的ad是6多，少的sd")
    datas = get_data()
    datas = [data.recognize_text for data in datas]
    with open("./datas.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(datas))
    nw = NewWords(filter_cond=10, filter_free=2)
    nw.add_text2(datas)
    # nw.add_text(r'E:\PycharmProjects\NLPExample\训练词向量\斗破苍穹.txt', encoding="GBK")
    nw.analysis_data()
    with open('分析结果1.txt', 'w', encoding='utf-8') as f:
        for word in nw.get_words():
            print(word)
            f.write(word[0] + '\n')