# -*- coding: utf-8 -*- 
# @Time 2020/3/20 13:47
# @Author wcy
import csv
import os
import pickle
import time
import numpy as np
import gensim
import jieba
from tqdm import tqdm
import pandas as pd
from random import choice, sample, random, randint, shuffle
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
        # self.init_self_library()
        self.init_LCQMC()

    def init_atec_nlp(self, path=None):
        filename = "datas/atec_nlp/atec_nlp.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb')as f:
                atec = pickle.load(f)
                self.dataset["atec"] = atec
                return
        if path is None:
            path = ["datas/atec_nlp/atec_nlp_sim_train.csv", "datas/atec_nlp/atec_nlp_sim_train_add.csv"]
        sentences = []
        datas = []
        labels = []
        lines = []
        for p in path:
            with open(p, "r", encoding="utf-8") as f:
                lines.extend(f.readlines())
        # csv_data = csv.reader(open(path[0], "r", encoding="utf-8"), delimiter='\t')
        for i, line in enumerate(tqdm(lines[:], desc="加载蚂蚁金服数据：", mininterval=1)):
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
        atec = {"datas": datas, "labels": labels, "sentences": sentences, "datas_same": datas_same}
        with open(filename, 'wb') as f:
            pickle.dump(atec, f)
        self.dataset["atec"] = atec

    def init_ChineseSTS(self, path="datas/ChineseSTS/simtrain_to05sts.txt"):
        filename = "datas/ChineseSTS/ChineseSTS.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb')as f:
                sts = pickle.load(f)
                self.dataset["sts"] = sts
                return
        sentences = []
        datas = []
        labels = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f.readlines()[:], desc="加载ChineseSTS数据：", mininterval=1)):
                _, seq1, _, seq2, score = line.strip("\n").split("\t")
                if not seq1 in sentences:
                    sentences.append(seq1)
                if not seq2 in sentences:
                    sentences.append(seq2)
                datas.append([sentences.index(seq1), sentences.index(seq2)])
                labels.append(float(score) / 5)
        datas_same = [data for data, label in zip(datas, labels) if label == 1]
        sentences = [jieba.lcut(sentence) for sentence in sentences]
        sts = {"datas": datas, "labels": labels, "sentences": sentences, "datas_same": datas_same}
        with open(filename, 'wb') as f:
            pickle.dump(sts, f)
        self.dataset["sts"] = sts

    def init_self_library(self, index=1):
        filename = f"datas/self_library/self_library{index}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb')as f:
                data = pickle.load(f)
                self.dataset["self_library"] = data
                return
        path = ["datas/self_library/贵阳市政务服务中心（知识点及问法).xlsx", "datas/self_library/省政府办公厅（知识点及问法）.xlsx"]
        datas1 = pd.read_excel(path[index], sheet_name=1).head(n=60000)
        if index == 0:
            question_dict = datas1.groupby("定位标准问").动态测试样例.apply(list).to_dict()
            sentences = list(set.union(set(datas1["定位标准问"]), set(datas1["动态测试样例"])))
        else:
            question_dict = datas1.groupby("标准问题").测试样例.apply(list).to_dict()
            sentences = list(set.union(set(datas1["标准问题"]), set(datas1["测试样例"])))
        datas_same = [[sentences.index(key), sentences.index(value)] for key, values in question_dict.items() for value
                      in values]
        labels_same = [1 for i in range(len(datas_same))]
        question_keys = list(question_dict.keys())
        datas_not_same = []
        labels_not_same = []
        for i in range(len(datas_same)):
            key1, key2 = sample(question_keys, 2)
            values1 = question_dict[key1]
            values2 = question_dict[key2]
            value1 = values1[randint(0, len(values1) - 1)]
            value2 = values2[randint(0, len(values2) - 1)]
            questions1 = [key1, value1]
            questions2 = [key2, value2]
            datas_not_same.append([sentences.index(choice(questions1)), sentences.index(choice(questions2))])
            labels_not_same.append(0)
        datas = datas_same + datas_not_same
        labels = labels_same + labels_not_same

        # 打乱顺序
        # dataset = list(zip(datas, labels))
        # shuffle(dataset)
        # datas[:], labels[:] = zip(*dataset)

        sentences = [jieba.lcut(sentence) for sentence in sentences]
        self_library = {"datas": datas, "labels": labels, "sentences": sentences, "datas_same": datas_same}
        with open(filename, 'wb') as f:
            pickle.dump(self_library, f)
        self.dataset["self_library"] = self_library

    def init_LCQMC(self, path="datas/CCKS/processed/test.tsv"):
        # filename = "datas/LCQMC/test.pkl"
        # if os.path.exists(filename):
        #     with open(filename, 'rb')as f:
        #         LCQMC = pickle.load(f)
        #         self.dataset["LCQMC"] = LCQMC
        #         return
        sentences = []
        datas = []
        labels = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f.readlines()[:], desc="加载LCQMC数据：", mininterval=1)):
                seq1, seq2, score = line.strip("\n").split("\t")
                if not seq1 in sentences:
                    sentences.append(seq1)
                if not seq2 in sentences:
                    sentences.append(seq2)
                datas.append([sentences.index(seq1), sentences.index(seq2)])
                labels.append(float(score))
        datas_same = [data for data, label in zip(datas, labels) if label == 1]
        sentences = [jieba.lcut(sentence) for sentence in sentences]
        LCQMC = {"datas": datas, "labels": labels, "sentences": sentences, "datas_same": datas_same}
        # with open(filename, 'wb') as f:
        #     pickle.dump(LCQMC, f)
        self.dataset["LCQMC"] = LCQMC

    def get_dataset(self, name="self_library"):
        return self.dataset[name]


if __name__ == '__main__':
    dataset = Dataset()
    # data = dataset.get_dataset()

    generate = Generate()
    for k, datas in dataset.dataset.items():
        print(f"数据集：{k}")
        sentences = datas["sentences"]
        datas_same = datas["datas_same"]
        inputs = datas["datas"]
        labels = datas["labels"]

        indexs = [i for i, [data, label] in enumerate(zip(inputs, labels)) if label == 0 or label == 1]
        inputs = [inputs[i] for i in indexs]
        labels = [labels[i] for i in indexs]

        model = SIF(generate.wv_from_text)
        model.train(IndexedList(sentences))
        predicts = []
        labels2 = []
        s = time.time()
        p = sum(labels)/len(labels)
        print(p)
        for [i, j], label in zip(inputs, labels):
            s1 = time.time()
            label1 = 1 if label > 0.5 else 0
            # if random() > p and label == 0:
            #     continue
            predict = model.sv.similarity(i, j)
            predict1 = 1 if predict > 0.5 else 0
            if abs(predict-label)>0.7:
                print("".join(sentences[i]), "".join(sentences[j]), predict, label)
            predicts.append(predict1)
            labels2.append(label1)
            e1 = time.time()
        e = time.time()
        print(f"耗时： {e - s}, count: {len(inputs)} 单个: {e1 - s1}")
        cm = confusion_matrix(predicts, labels2)
        print(cm)
        cm = cm[::-1, ::-1]
        TP = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TN = cm[1, 1]
        accuracy = (TP + TN) / (TP + TN + FP + FN)  # 准确率
        error = (FP + FN) / (TP + TN + FP + FN)  # 错误率

        specificity = TN / (TN + FP)  # 特效度 表示的是所有负例中被分对的比例，衡量了分类器对负例的识别能力。
        precision = TP / (TP + FP)  # 精确率、精度  表示被分为正例的示例中实际为正例的比例。

        recall = confusion_sensitive = TP / (TP + FN)  # 召回率,灵敏度 表示的是所有正例中被分对的比例，衡量了分类器对正例的识别能力。

        a = 1
        F1 = ((a ** 2 + 1) * precision * recall) / (
                (precision + recall) * a ** 2)

        print(cm)
        print(f"准确率: {accuracy}")
        print(f"错误率: {error}")
        print(f"特效率: {specificity}")
        print(f"精确率: {precision}")
        print(f"召回率: {recall}")
        print(f"F1: {F1}")
