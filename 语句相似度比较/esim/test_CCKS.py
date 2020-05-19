import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from 语句相似度比较.esim.graph import Graph
import tensorflow as tf
from 语句相似度比较.utils.load_data import load_char_data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# p, h, y = load_char_data('data/LCQMC/processed/test.tsv', data_start=0, data_size=12500)

model = Graph()
saver = tf.train.Saver()

with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './model_CCKS/esim_39.ckpt')
    CM = np.zeros((2, 2), dtype=np.int)
    for i in range(10):
        p, h, y = load_char_data('data/CCKS/processed/test.tsv', data_start=1000*i, data_size=1000*(i+1))
        loss, acc, cm = sess.run([model.loss, model.acc, model.confusion_matrix],
                             feed_dict={model.p: p,
                                        model.h: h,
                                        model.y: y,
                                        model.keep_prob: 1})
        CM+=cm
        print(cm, end="\n\n")

    print('loss: ', loss, ' acc:', acc, '\n', 'cm', CM)
