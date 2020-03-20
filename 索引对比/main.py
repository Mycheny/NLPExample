# -*- coding: utf-8 -*- 
# @Time 2020/3/19 13:36
# @Author wcy

import time, random
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn import preprocessing
from annoy import AnnoyIndex

import gensim
import warnings

warnings.filterwarnings("ignore")

path = "E:\\DATA\\tencent\\ChineseEmbedding.bin"

# model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors='ignore')
model = gensim.models.KeyedVectors.load(path, mmap='r')
# model.init_sims(replace=True)  # l2-normalized <=> euclidean :  (a-b)^2 = a^2 + b^2 - 2ab = 2-2ab  <==>2-2*cos
words = ["王者"] * 1000


class ANNSearch:
    word2idx = {}
    idx2word = {}
    data = []

    def __init__(self, model):
        for counter, key in enumerate(model.vocab.keys()):
            self.data.append(model[key])
            self.word2idx[key] = counter
            self.idx2word[counter] = key

        # leaf_size is a hyperparameter

        # 这里加L2正则化，使得余弦相似度就是跟欧式距离等价
        # self.data=preprocessing.normalize(np.array(self.data), norm='l2')

        self.data = np.array(self.data)

    def search_by_annoy(self, query, annoymodel, k=10):
        index = self.word2idx[query]
        result = annoymodel.get_nns_by_item(index, k)
        word_result = [self.idx2word[idx] for idx in result[1:]]

        return word_result


def time_test():
    # Linear Search
    start = time.time()
    for word in words:
        model.most_similar(word, topn=10)
    stop = time.time()
    print("time/query by (gensim's) Linear Search = %.2f s" % (float(stop - start)))

    search_model = ANNSearch(model)

    ###annoy  serarch
    annoy_model = AnnoyIndex(200)
    annoy_model.load('/Users/zhoumeixu/Documents/python/word2vec/bin/annoy.model')
    start = time.time()
    for word in words:
        search_model.search_by_annoy(word, annoy_model, k=10)
    stop = time.time()
    print("time/query by annoy Search = %.2f s" % (float(stop - start)))


def result_test():
    # print("gensim:", model.most_similar("王者", topn=2))

    search_model = ANNSearch(model)

    annoy_model = AnnoyIndex(200)
    annoy_model.save()
    annoy_model.load('/Users/zhoumeixu/Documents/python/word2vec/bin/annoy.model')

    print("annoy:", list(search_model.search_by_annoy("王者", annoy_model, k=6)))


if __name__ == "__main__":
    # time_test()

    result_test()
