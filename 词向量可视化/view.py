import time

import gensim
from numpy.linalg import linalg
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import codecs

words, embeddings = [], []
log_path = 'model'

t1 = time.time()
if not os.path.exists("E:\\DATA\\tencent\\ChineseEmbedding.bin"):
    # wv_from_text = gensim.models.KeyedVectors.load_word2vec_format('E:\\DATA\\tencent\\Tencent_AILab_ChineseEmbedding.txt', limit=10000, binary=False)
    wv_from_text = gensim.models.KeyedVectors.load_word2vec_format('E:\\DATA\\tencent\\Tencent_AILab_ChineseEmbedding.txt', binary=False)
    # 使用init_sims会比较省内存
    print("文件载入耗费时间：", (time.time() - t1), "s")
    wv_from_text.init_sims(replace=True)
    # 重新保存加载变量为二进制形式
    wv_from_text.save(r"E:\\DATA\\tencent\\ChineseEmbedding.bin")
else:
    wv_from_text = gensim.models.KeyedVectors.load(r'E:\\DATA\\tencent\\ChineseEmbedding.bin', mmap='r')
print("文件载入耗费时间：", (time.time() - t1) / 60, "minutes")
words = wv_from_text.index2word[0:300000]
embeddings = wv_from_text.vectors[0:300000, :]
embeddings = np.array(embeddings)
vector_size = wv_from_text.vector_size
with tf.Session() as sess:
    X = tf.Variable([0.0], name='embedding')
    place = tf.placeholder(tf.float32, shape=[len(words), vector_size])
    set_x = tf.assign(X, place, validate_shape=False)
    sess.run(tf.global_variables_initializer())
    sess.run(set_x, feed_dict={place: embeddings})
    with codecs.open(log_path + '/metadata.tsv', 'w', encoding="utf-8") as f:
        f.write("Index\tLabel\n")
        root = "下"
        vector0 = wv_from_text[root]
        for index, word in enumerate(tqdm(words)):
            vector1 = wv_from_text[word]
            cos = np.sum(vector0.T * vector1)/(np.sqrt(np.sum(np.square(vector0)))*np.sqrt(np.sum(np.square(vector1))))
            type = int(cos*10000)
            f.write("%d\t%s\n" % (type, word))

    # with summary
    summary_writer = tf.summary.FileWriter(log_path, sess.graph)
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = 'embedding:0'
    embedding_conf.metadata_path = os.path.join('metadata.tsv')
    projector.visualize_embeddings(summary_writer, config)

    # save
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(log_path, "model.ckpt"))
