# -*- coding: utf-8 -*- 
# @Time 2020/4/19 21:11
# @Author wcy
import json
import os

import gensim
from keras import Input, Model
from keras import backend as K
from keras.layers import Lambda, Dense
from keras.layers import concatenate, multiply, Dense
from keras import layers
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def shard_cnn_encoder(input_shape, cnn_filters=128):
    # Input Layer
    _input = Input(shape=input_shape)

    max_sent_len, dim_size = input_shape
    # Conv Layer
    convs = []
    filter_sizes = [2, 3, 4, 5, 6, 7]

    for size in filter_sizes:
        l_conv = layers.Conv1D(filters=cnn_filters, kernel_size=size, activation='relu')(_input)
        l_pool = layers.MaxPooling1D(max_sent_len - size + 1)(l_conv)
        l_pool = layers.Flatten()(l_pool)
        convs.append(l_pool)

    sentence_rep = layers.concatenate(convs, axis=1)

    shared_model = Model(_input, sentence_rep, name='cnn_shared_model')

    return shared_model


def build_siamese_cnn_model(dim_size=200, max_sent_len=16):
    left_input = Input(shape=(max_sent_len, dim_size), dtype='float32', name="left_x")
    right_input = Input(shape=(max_sent_len, dim_size), dtype='float32', name='right_x')

    shard_model = shard_cnn_encoder((max_sent_len, dim_size))

    u_input = shard_model(left_input)
    v_input = shard_model(right_input)

    u_sub_v = Lambda(lambda x: K.abs(x[0] - x[1]))([u_input, v_input])
    u_mul_v = multiply([u_input, v_input])

    u_concat_v = concatenate([u_input, v_input, u_sub_v, u_mul_v])

    # dense = Dense(dense_unit, activation='relu')(u_concat_v)
    similarity = Dense(1, activation='sigmoid')(u_concat_v)

    model = Model([left_input, right_input], similarity)
    print(model.summary())
    return model


model = build_siamese_cnn_model()

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report


def train_model(model, train_x, train_y, dev_x, dev_y, checkpointpath, lr=1e-3, batch_size=128, epochs=16):
    adam = Adam(lr=lr)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    checkpoint_callback = ModelCheckpoint(checkpointpath,
                                          monitor='val_acc',
                                          save_best_only=True,
                                          save_weights_only=True,
                                          mode='auto',
                                          period=1)
    model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(dev_x, dev_y),
              shuffle=True,
              callbacks=[checkpoint_callback])


def model_predict(model, test_x, test_y=None, predict_batchsize=128):
    predict_y = model.predict(test_x, batch_size=predict_batchsize)[:, 0]

    # for i in range(10):
    #     s = time.time()
    #     predict_y = model.predict([test_x[0][0:1], test_x[1][0:1]], batch_size=predict_batchsize)[:, 0]
    #     e = time.time()
    #     print(f"time: {e-s}")

    if test_y.any():
        predict_y[predict_y >= 0.5] = 1
        predict_y[predict_y < 0.5] = 0
        print(classification_report(test_y, predict_y))


import jieba

# dictfile='tc_min.dict'
# jieba 加载自定义词典
# jieba.load_userdict(dictfile)

# from annoy import AnnoyIndex
#
#
# def init_index(annoy_indexfile='tc_index_build10.ann.index', word2indexfile='tc_word_index.json'):
#     # 我们用保存好的索引文件重新创建一个Annoy索引, 单独进行加载
#     annoy_index = AnnoyIndex(200)
#     annoy_index.load(annoy_indexfile)
#
#     with open(word2indexfile, 'r') as fp:
#         word2index = json.load(fp)
#
#     # 准备一个反向id==>word映射词表
#     index2word = dict([(value, key) for (key, value) in word2index.items()])
#
#     return annoy_index, word2index, index2word
#
# annoy_index,word2index,index2word = init_index()

path = "E:\\DATA\\tencent\\ChineseEmbedding.bin"
vector_model = gensim.models.KeyedVectors.load(path, mmap='r')
index2word = vector_model.index2word
word2index = {word: index for index, word in enumerate(index2word)}
print()


# word2index['你好']

def load_sentence_data(file_path):
    sentences1 = []
    sentences2 = []
    labels = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            s1, s2, label = line.split('\t')
            if not label or not label.strip("\n").isdigit():
                continue
            labels.append(int(label))
            sentences1.append(s1)
            sentences2.append(s2)
    return sentences1, sentences2, labels


from pathlib import Path

dataset_path = ''

LCQMC = [
    '../data/LCQMC/processed/train.tsv',
    '../data/LCQMC/processed/dev.tsv',
    '../data/LCQMC/processed/test.tsv',
]

CCKS = [
    '../data/CCKS/processed/train.tsv',
    '../data/CCKS/processed/dev.tsv',
    '../data/CCKS/processed/test.tsv',
]

ATEC = [
    '../data/ATEC/processed/train.tsv',
    '../data/ATEC/processed/dev.tsv',
    '../data/ATEC/processed/test.tsv',
]

CORPUS = [
    (LCQMC, 'LCQMC'),
    (CCKS, 'CCKS'),
    (ATEC, 'ATEC')
]

import numpy as np


def texts_to_sequences(sentences):
    seq_len = len(sentences)
    sequences = []

    for sentence in sentences:
        seq_list = [word2index[token] for token in list(jieba.cut(sentence))
                    if token in word2index.keys()]
        sequences.append(seq_list)

    return np.array(sequences)


def sequences_to_embeddings(sequences, embed_index=vector_model, dim_size=200):
    seq_num, seq_len = sequences.shape
    embeddings = np.zeros((seq_num, seq_len, dim_size))
    items = embed_index.vectors.shape[0]
    i = 0
    for seq in sequences:
        new_sequence = np.array(
            [embed_index[index2word[wordindex]] if wordindex < items else embed_index[index2word[wordindex]]
             for wordindex in seq])
        embeddings[i] = new_sequence
        i = i + 1

    return embeddings


from keras_preprocessing.sequence import pad_sequences


def build_dataset(dataset, max_sent_len=16):
    s1, s2, labels = load_sentence_data(dataset)

    left_X = pad_sequences(texts_to_sequences(s1), max_sent_len)
    right_X = pad_sequences(texts_to_sequences(s2), max_sent_len)

    left_X = sequences_to_embeddings(left_X)
    right_X = sequences_to_embeddings(right_X)

    Y = np.array(labels)

    return [left_X, right_X], Y


import time


def evaluate(dataset, checkpoint):
    dataset_dir = Path(dataset_path)
    train_file = dataset_dir / dataset[0]
    dev_file = dataset_dir / dataset[1]
    test_file = dataset_dir / dataset[2]

    # train_x, train_y = build_dataset(train_file)
    # dev_x, dev_y = build_dataset(dev_file)
    test_x, test_y = build_dataset(test_file)

    start = time.time()
    model = build_siamese_cnn_model()

    # train_model(model, train_x, train_y, dev_x, dev_y, checkpointpath=checkpoint)

    stop = time.time()

    print("time for train model = %.2f s" % (float(stop - start)))

    start = time.time()

    model.load_weights(checkpoint)
    # model.save_weights(checkpoint)
    # model.save(f"{checkpoint}_save.h5")

    model_predict(model, test_x, test_y)

    stop = time.time()

    print("time for predict = %.2f s" % (float(stop - start)))


for data in CORPUS:
    print('----- DataSet:{0}---------'.format(data[1]))
    checkpoint = "siamese_cnn_" + data[1] + "_best.h5"
    evaluate(data[0], checkpoint)
