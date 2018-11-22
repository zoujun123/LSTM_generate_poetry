# coding:utf-8

import argparse
import sys
import os
import time
import numpy as np
import collections
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.legacy_seq2seq as seq2seq
import data_process
from utils import get_logger

class Poetry_LSTM(object):
    """
    build_graph()搭建网络结构，设置运算关系；
    train()中根据每个run_one_epoch()函数对每个变量进行了计算
    """
    def __init__(self, args, embeddings, char2id_dict, id2char_dict,words_size, model_path, infer = False):

        self.infer = infer
        if self.infer:
            self.batch_size = 1
        else:
            self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings_dim = args.embedding_dim
        self.embeddings = embeddings
        self.n_layers = args.layers
        self.char2id_dict = char2id_dict
        self.id2char_dict = id2char_dict
        # self.update_embedding = args.update_embedding
        # self.dropout_keep_prob = args.dropout
        # 优化器的类型
        self.optimizer = args.optimizer
        self.lr_pl = args.lr
        self.model_path = model_path
        self.logger = get_logger(model_path + '_record.log')
        self.layers = args.layers
        self.words_size = words_size
        self.unknown_char = self.char2id_dict.get('*')
        # self.unknown_char = self.char2id_dict.get(UNKNOWN_CHAR)
        self.BEGIN_CHAR = '^'
        self.END_CHAR = '$'


    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.lstm_layer_op()
        self.loss_op()
        self.train_op()
        self.init_variables()

    def add_placeholders(self):

        # self.x_poetry_id存储每个句子id化后的结果 feedback后应该有：shape [诗句总量, 维度]
        self.x_poetry_id = tf.placeholder(tf.int32, shape=[None, None], name="x_poetry_id")
        self.y_tf = tf.placeholder(tf.int32, shape = [None, None], name = "y_labels_id")

    def lookup_layer_op(self):

        self.word_embeddings = tf.nn.embedding_lookup(params=self.embeddings,
                                                     ids=self.x_poetry_id,
                                                     name="word_embeddings")

    def lstm_layer_op(self):

        with tf.variable_scope("lstm"):
            # LSTM Cell会产生两个内部状态 c 和h;当state_is_tuple=True时，
            # 上面讲到的状态c和h就是分开记录，放在一个二元tuple中返回，如果这个参数没有设定或设置成False，
            # 两个状态就按列连接起来返回。
            # 作者原代码中有state_is_tuple = False
            cell = rnn.BasicLSTMCell(self.hidden_dim)
            # 多层循环神经网络
            self.cell = rnn.MultiRNNCell([cell] * self.n_layers)
            # self.initial_state = self.cell.zero_state(self.batch_size, tf.float64)
            # outputs, final_state = tf.nn.dynamic_rnn(
            #     self.cell, self.word_embeddings, initial_state=self.initial_state, scope='lstm')
            outputs, final_state = tf.nn.dynamic_rnn(
                self.cell, self.word_embeddings, scope='lstm', dtype=tf.float64)

            softmax_w = tf.get_variable(name = "softmax_w",
                                        shape = [self.hidden_dim, self.words_size],
                                        # initializer=tf.contrib.layers.xavier_initializer(),
                                        dtype=tf.float64
                                        )
            softmax_b = tf.get_variable(name = "softmax_b",
                                        shape = [self.words_size],
                                        # initializer=tf.zeros_initializer(),
                                        dtype=tf.float64
                                        )

            self.output = tf.reshape(outputs, [-1, self.hidden_dim])
            self.logits = tf.matmul(self.output, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits)
            self.final_state = final_state
            self.pred = tf.reshape(self.y_tf, [-1])

    def loss_op(self):

        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [self.pred],
                                                [tf.ones_like(self.pred, dtype=tf.float64)], )

        self.cost = tf.reduce_mean(loss)

    def train_op(self):

        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            # grads_and_vars = optim.compute_gradients(self.loss)
            # grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            # self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

            # 以上三步可直接用minimize()函数替代
            self.train_op = optim.minimize(self.cost)

    def init_variables(self):

        self.init_op = tf.global_variables_initializer()

    def train(self, x_train):

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            sess.run(self.init_op)

            # 调用run_one_epoch函数进行具体每个epoch的运算
            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, x_train, epoch, saver)

    def run_one_epoch(self, sess, x_train, epoch, saver):

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # num_batches = (len(x_train) + self.batch_size - 1) // self.batch_size
        num_batches = len(x_train) // self.batch_size
        batches = data_process.batch_yield(x_train, self.batch_size)
        # 一个epoch要把所有的batch跑一遍
        for step, batch_x in enumerate(batches):
            # step 表示当前是第几个batch, batch_x是当前batch下的所有内容
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            sys.stdout.flush()

            length = max(map(len, batch_x))
            print ("length: ", length)
            curr_batch_size = len(batch_x)
            for row in range(curr_batch_size):
                if len(batch_x[row]) < length:
                    r = length - len(batch_x[row])
                    # 用UNKNOW_CHAR进行padding
                    batch_x[row][len(batch_x[row]): length] = [self.unknown_char] * r

            xdata = np.array(batch_x)
            ydata = np.copy(xdata)
            ydata[:, :-1] = xdata[:, 1:]

            # step_num 标志是总的第几个batch
            step_num = epoch * num_batches + step + 1
            feed_dict = {self.x_poetry_id: xdata, self.y_tf: ydata}
            # print ("self.x_poetry_id: ", self.x_poetry_id.shape)
            train_loss, i, j, step_num_ = sess.run([self.cost, self.final_state,
                                                    self.train_op, self.global_step],
                                                    feed_dict=feed_dict)

            sys.stdout.write('\r')
            info = "{}/{} (epoch {}) | train_loss {:.3f}" \
                .format(epoch * self.batch_size + step, self.epoch_num * self.batch_size, epoch, train_loss)
            sys.stdout.write(info)
            sys.stdout.flush()

            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                # step 标识当前epoch下的第几个batch
                # step_num 标志是总的第几个batch
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                train_loss, step_num))
            # 当前Batch训练完成
            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

    def to_word(self, weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sa = int(np.searchsorted(t, np.random.rand(1) * s))
        return data_process.id2char(sa, self.id2char_dict)

    def sample(self):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, self.model_path)

            poem = ''
            head = self.BEGIN_CHAR

            temp = []
            for char in head:
                temp.append(data_process.char2id(char, self.char2id_dict))
            x = np.array([temp])
            # state = sess.run(self.cell.zero_state(self.batch_size, tf.float32))
            # feed_dict = {self.x_poetry_id: x, self.initial_state: state}
            feed_dict = {self.x_poetry_id: x}
            # 不用计算Loss,因此不用带入y;
            [probs, state] = sess.run([self.probs, self.final_state], feed_dict)
            # print("prob: ", probs)
            # print("probs[-1]: ", probs[-1])
            word = self.to_word(probs[-1])
            # print("word: ", word)
            # word是生成的句首的字
            while word != self.END_CHAR and len(poem) < 24:
                if word != '。' and word != ',' and word != '*' :
                    poem += word
                x = np.zeros((1, 1))
                # print("x: ", x)
                # 之前训练模型的原理是，根据前一个字符预测后一个字符，因此需要先有一个句首的字
                x[0, 0] = data_process.char2id(word, self.char2id_dict)
                [probs, state] = sess.run([self.probs, self.final_state],
                                          {self.x_poetry_id: x})
                # [probs, state] = sess.run([self.probs, self.final_state],
                #                           {self.x_poetry_id: x, self.initial_state: state})
                word = self.to_word(probs[-1])
                if len(poem) == 5 or len(poem)%6 == 5:
                    poem += '\n'
                # print ("word: ", word)
            return poem
