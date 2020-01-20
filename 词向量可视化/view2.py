import time
import jieba
import gensim
from numpy.linalg import linalg
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import codecs


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
        self.wv_from_text=wv_from_text
        self.vector_size = wv_from_text.vector_size
        print("文件载入耗费时间：", (time.time() - t1) / 60, "minutes")

    def generate_vector(self, sentences):
        vectors = []
        for sentence in sentences:
            sentence = jieba.lcut(sentence)
            vector = np.sum(np.array([np.array(self.wv_from_text[word]) for word in sentence]), axis=0)
            vectors.append(vector)
        return vectors


if __name__ == '__main__':
    log_path = 'model'
    generate = Generate()
    words = ["猫站在地毯上", "人站在地毯上", "地毯上有猫", "地毯上有人",
             "我好喜欢你", "我喜欢你", "你喜欢我吗？", "我好恨你", "我恨你", "你个憨货",
             "谢霆锋的儿子是谁", "谢霆锋是谁的儿子"]
    embeddings = generate.generate_vector(words)

    with tf.Session() as sess:
        X = tf.Variable([0.0], name='embedding')
        place = tf.placeholder(tf.float32, shape=[len(words), generate.vector_size])
        set_x = tf.assign(X, place, validate_shape=False)
        sess.run(tf.global_variables_initializer())
        sess.run(set_x, feed_dict={place: embeddings})
        with codecs.open(log_path + '/metadata.tsv', 'w', encoding="utf-8") as f:
            f.write("Index\tLabel\n")
            for index, word in enumerate(tqdm(words)):
                f.write("%d\t%s\n" % (index, word))

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
