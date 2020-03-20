# -*- coding: utf-8 -*- 
# @Time 2020/3/20 13:47
# @Author wcy
import os
import time
import numpy as np
import gensim
import jieba
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
from 句向量可视化.fse import IndexedList
from 句向量可视化.fse.models import SIF


class Generate(object):
    def __init__(self):
        t1 = time.time()
        if not os.path.exists("E:\\DATA\\tencent\\ChineseEmbedding.bin"):
            # wv_from_text = gensim.models.KeyedVectors.load_word2vec_format('E:\\DATA\\tencent\\Tencent_AILab_ChineseEmbedding.txt', limit=10000, binary=False)
            wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(
                'E:\\DATA\\tencent\\Tencent_AILab_ChineseEmbedding.txt', binary=False)
            # 使用init_sims会比较省内存
            print("文件载入耗费时间：", (time.time() - t1), "s")
            wv_from_text.init_sims(replace=True)
            # 重新保存加载变量为二进制形式
            wv_from_text.save(r"E:\\DATA\\tencent\\ChineseEmbedding.bin")
        else:
            wv_from_text = gensim.models.KeyedVectors.load(r'E:\\DATA\\tencent\\ChineseEmbedding.bin', mmap='r')

        self.wv_from_text = wv_from_text
        self.vector_size = wv_from_text.vector_size
        print("文件载入耗费时间：", (time.time() - t1) / 60, "minutes")

    def generate_mean_vector(self, sentences):
        """生成平均句向量"""
        vectors = []
        for sentence in tqdm(sentences):
            if isinstance(sentence, str):
                sentence = jieba.lcut(sentence)
            vector = np.mean(np.array(
                [np.array(self.wv_from_text[word]) for word in sentence if word in self.wv_from_text.index2word]),
                axis=0)
            vectors.append(vector)
        return vectors


class Dataset(object):
    """数据集生成器，包括蚂蚁金服数据、ChineseSTS数据、自己的数据"""
    dataset = {}

    def __init__(self):
        # self.init_atec_nlp()
        # self.init_ChineseSTS()
        self.init_self_library()

    def init_atec_nlp(self, path=None):
        if path is None:
            path = ["datas/atec_nlp/atec_nlp_sim_train.csv", "datas/atec_nlp/atec_nlp_sim_train_add.csv"]
        sentences = []
        datas = []
        labels = []
        lines = []
        for p in path:
            with open(p, "r", encoding="utf-8") as f:
                lines.extend(f.readlines())
        for i, line in enumerate(tqdm(lines[:8000], desc="加载蚂蚁金服数据：", mininterval=1)):
            _, seq1, seq2, score = line.strip("\n").split("\t")
            seq1 = seq1.encode('utf-8').decode('utf-8-sig')
            seq2 = seq2.encode('utf-8').decode('utf-8-sig')
            if not seq1 in sentences:
                sentences.append(seq1)
            if not seq2 in sentences:
                sentences.append(seq2)
            datas.append([sentences.index(seq1), sentences.index(seq2)])
            labels.append(int(score))
        datas_same = [data for data, label in zip(datas, labels) if label == 1]
        sentences = [jieba.lcut(sentence) for sentence in sentences]
        self.dataset["atec"] = {"datas": datas, "labels": labels, "sentences": sentences, "datas_same": datas_same}

    def init_ChineseSTS(self, path="datas/ChineseSTS/simtrain_to05sts.txt"):
        sentences = []
        datas = []
        labels = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f.readlines()[:5000], desc="加载ChineseSTS数据：", mininterval=1)):
                _, seq1, _, seq2, score = line.strip("\n").split("\t")
                if not seq1 in sentences:
                    sentences.append(seq1)
                if not seq2 in sentences:
                    sentences.append(seq2)
                datas.append([sentences.index(seq1), sentences.index(seq2)])
                labels.append(float(score)/5)
        datas_same = [data for data, label in zip(datas, labels) if label == 1]
        sentences = [jieba.lcut(sentence) for sentence in sentences]
        self.dataset["sts"] = {"datas": datas, "labels": labels, "sentences": sentences, "datas_same": datas_same}

    def init_self_library(self):
        path1 = "datas/self_library/贵阳市政务服务中心（知识点及问法).xlsx"
        path2 = "datas/self_library/省政府办公厅（知识点及问法）.xlsx"
        datas1 = pd.read_excel(path1)
        print()


    def get_dataset(self, name="atec"):
        return self.dataset[name]


if __name__ == '__main__':
    dataset = Dataset()
    data = dataset.get_dataset()

    generate = Generate()
    for k, datas in dataset.dataset.items():
        sentences = datas["sentences"]
        datas_same = datas["datas_same"]
        inputs = datas["datas"]
        labels = datas["labels"]
        model = SIF(generate.wv_from_text)
        model.train(IndexedList(sentences))
        predicts = []
        labels2 = []
        for [i, j], label in zip(inputs, labels):
            predict = model.sv.similarity(i, j)
            predict = 1 if predict>0.5 else 0
            label = 1 if label>0.5 else 0
            predicts.append(predict)
            labels2.append(label)
        cm = confusion_matrix(labels2, predicts)
        TP = cm[0, 0]
        FN = cm[0, 1]
        FP = cm[1, 0]
        TN = cm[1, 1]
        accuracy = (TP + TN) / (TP + TN + FP + FN)  # 准确率
        error = (FP + FN) / (TP + TN + FP + FN)  # 错误率

        specificity = TN / (TN + FP)  # 特效度 表示的是所有负例中被分对的比例，衡量了分类器对负例的识别能力。
        precision = TP / (TP + FP)  # 精确率、精度  表示被分为正例的示例中实际为正例的比例。

        recall = confusion_sensitive = TP / (TP + FN)  # 召回率,灵敏度 表示的是所有正例中被分对的比例，衡量了分类器对正例的识别能力。

        print(cm)
        print(f"准确率: {accuracy}")
        print(f"错误率: {error}")
        print(f"特效度: {specificity}")
        print(f"精确率: {precision}")
        print(f"召回率: {recall}")
