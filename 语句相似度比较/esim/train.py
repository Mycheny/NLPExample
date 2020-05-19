import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from 语句相似度比较.esim.graph import Graph
import tensorflow as tf
from 语句相似度比较.utils.load_data import load_char_data
from 语句相似度比较.esim import args
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


p, h, y = load_char_data('./data/LCQMC/processed/train.tsv', data_size=None)
p_eval, h_eval, y_eval = load_char_data('./data/LCQMC/processed/dev.tsv', data_size=1000)

p_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='p')
h_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='h')
y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

dataset = tf.data.Dataset.from_tensor_slices((p_holder, h_holder, y_holder))
dataset = dataset.batch(args.batch_size).repeat(args.epochs)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

model = Graph()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={p_holder: p, h_holder: h, y_holder: y})
    steps = int(len(y) / args.batch_size)
    for epoch in range(args.epochs):
        for step in range(steps):
            p_batch, h_batch, y_batch = sess.run(next_element)
            _, loss, acc, cm = sess.run([model.train_op, model.loss, model.acc, model.confusion_matrix],
                                    feed_dict={model.p: p_batch,
                                               model.h: h_batch,
                                               model.y: y_batch,
                                               model.keep_prob: args.keep_prob})
            print('epoch:', epoch, ' step:', step, ' loss: ', loss, ' acc:', acc, 'cm', cm)

        loss_eval, acc_eval, cm = sess.run([model.loss, model.acc, model.confusion_matrix],
                                       feed_dict={model.p: p_eval,
                                                  model.h: h_eval,
                                                  model.y: y_eval,
                                                  model.keep_prob: 1})
        print('loss_eval: ', loss_eval, ' acc_eval:', acc_eval, 'cm', cm)
        print('\n')
        saver.save(sess, f'./model/esim_{epoch}.ckpt')
