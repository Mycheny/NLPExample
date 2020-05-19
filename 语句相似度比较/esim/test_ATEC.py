import os
import sys
import time

import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from 语句相似度比较.esim.graph import Graph
import tensorflow as tf
from 语句相似度比较.utils.load_data import load_char_data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# p, h, y = load_char_data('data/LCQMC/processed/test.tsv', data_start=0, data_size=12500)

model = Graph()
saver = tf.train.Saver()

with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './model_ATEC/esim_49.ckpt')
    CM = np.zeros((2, 2), dtype=np.int)
    ps, hs, ys = load_char_data('data/ATEC/processed/test.tsv')
    ts = []
    for i in range(5):
        s = time.time()
        p = ps[0:1, :]
        h = hs[0:1, :]
        y = ys[0:1]
        # p = ps
        # h = hs
        # y = ys
        loss, acc, cm = sess.run([model.loss, model.acc, model.confusion_matrix],
                             feed_dict={model.p: p,
                                        model.h: h,
                                        model.y: y,
                                        model.keep_prob: 1})
        e = time.time()
        t = e - s
        ts.append(t)
        print(f"time: {t}")
        CM+=cm
        print(cm, f"\n time: {e-s}", end="\n\n")
    print(f"{np.array(ts).mean()}")
    print(f"{np.array(ts)}")
    print('loss: ', loss, ' acc:', acc, '\n', 'cm', CM)
