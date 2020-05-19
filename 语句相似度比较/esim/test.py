import os
import sys
import time

import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from 语句相似度比较.esim.graph import Graph
import tensorflow as tf
from 语句相似度比较.utils.load_data import load_char_data

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# p, h, y = load_char_data('data/LCQMC/processed/test.tsv', data_start=0, data_size=12500)

model = Graph()
saver = tf.train.Saver()

with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, '../../model/esim_16.ckpt')
    # saver.restore(sess, './model/esim_38.ckpt')
    CM = np.zeros((2, 2), dtype=np.int)
    ts = []
    for i in range(50):
        s = time.time()
        p, h, y = load_char_data('data/LCQMC/processed/test.tsv', data_start=1*i, data_size=1*(i+1))
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
        print(cm, end="\n\n")
    print(f"{np.array(ts).mean()}")
    print('loss: ', loss, ' acc:', acc, '\n', 'cm', CM)
