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

msg = """
            Usage:
            Training: 
                python poetry_gen.py --mode train
            Sampling:
                python poetry_gen.py --mode sample --head 明月别枝惊鹊
            """
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train',
                    help=u'usage: train or sample, sample is default')
parser.add_argument('--head', type=str, default='',
                    help='生成藏头诗')

args = parser.parse_args()

BEGIN_CHAR = '^'
END_CHAR = '$'
UNKNOWN_CHAR = '*'
MAX_LENGTH = 100
MIN_LENGTH = 10
max_words = 3000
epochs = 5
poetry_file = 'som-poetry.txt'
save_dir = 'log'

if args.mode == 'train':
    timestamp = str(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
    model_path = os.path.join('.', save_dir, timestamp)
    print (model_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
else:
    timestamp = input("input: ")
    model_path = os.path.join('.', save_dir, timestamp)


class Data:
    def __init__(self):
        self.batch_size = 5
        self.poetry_file = poetry_file
        self.load()
        self.create_batches()

    def load(self):
        def handle(line):
            if len(line) > MAX_LENGTH:
                index_end = line.rfind('。', 0, MAX_LENGTH)
                index_end = index_end if index_end > 0 else MAX_LENGTH
                line = line[:index_end + 1]
            return BEGIN_CHAR + line + END_CHAR

        # 获取了每首诗的正文内容
        self.poetrys = [line.strip().replace(' ', '').split(':')[1] for line in
                        open(self.poetry_file, encoding='utf-8')]
        # 添加上开头结尾符号，并通过MAX_LENGTH进行了长度约束 【尚未进行padding】
        self.poetrys = [handle(line) for line in self.poetrys if len(line) > MIN_LENGTH]
        # 所有字
        words = []
        for poetry in self.poetrys:
            words += [word for word in poetry]
        counter = collections.Counter(words)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        # [('，', 50), ('。', 50), ('^', 9), ('$', 9),...
        words, _ = zip(*count_pairs)
        # ('，', '。', '^', '$',...)  (50, 50, 9, 9,...)

        # 取出现频率最高的词的数量组成字典，不在字典中的字用'*'代替
        words_size = min(max_words, len(words))
        self.words = words[:words_size] + (UNKNOWN_CHAR,)
        # 字典容量：min(max_words, len(words)) + 1
        self.words_size = len(self.words)

        # 字映射成id
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}
        self.unknow_char = self.char2id_dict.get(UNKNOWN_CHAR)
        self.char2id = lambda char: self.char2id_dict.get(char, self.unknow_char)
        self.id2char = lambda num: self.id2char_dict.get(num)
        # 根据长度进行排序
        self.poetrys = sorted(self.poetrys, key=lambda line: len(line))
        # 将每首诗的字转化为id
        self.poetrys_vector = [list(map(self.char2id, poetry)) for poetry in self.poetrys]

    def create_batches(self):
        # bath的数量
        self.n_size = len(self.poetrys_vector) // self.batch_size
        # 这样获取Batch的方式会使最后不满足batch_size的部分被丢弃
        self.poetrys_vector = self.poetrys_vector[:self.n_size * self.batch_size]

        self.x_batches = []
        self.y_batches = []
        for i in range(self.n_size):
            batches = self.poetrys_vector[i * self.batch_size: (i + 1) * self.batch_size]
            # length是当前batch内的最大长度,用作padding的参考
            length = max(map(len, batches))
            for row in range(self.batch_size):
                if len(batches[row]) < length:
                    r = length - len(batches[row])
                    # 用UNKNOW_CHAR进行padding
                    batches[row][len(batches[row]): length] = [self.unknow_char] * r
            xdata = np.array(batches)
            ydata = np.copy(xdata)
            # y是从x的第一列开始，然后最后一列多重复一次？
            ydata[:, :-1] = xdata[:, 1:]
            # self.x_batched是所有的x_batch
            self.x_batches.append(xdata)
            # self.y_batched是所有的y_batch
            self.y_batches.append(ydata)


class Model:
    def __init__(self, data, model='lstm', infer=False):
        self.rnn_size = 128
        self.n_layers = 2

        if infer:
            self.batch_size = 1
        else:
            self.batch_size = data.batch_size

        if model == 'rnn':
            cell_rnn = rnn.BasicRNNCell
        elif model == 'gru':
            cell_rnn = rnn.GRUCell
        elif model == 'lstm':
            cell_rnn = rnn.BasicLSTMCell

        # rnn_size是hidden_dim
        cell = cell_rnn(self.rnn_size, state_is_tuple=False)
        self.cell = rnn.MultiRNNCell([cell] * self.n_layers, state_is_tuple=False)

        self.x_tf = tf.placeholder(tf.int32, [self.batch_size, None])
        self.y_tf = tf.placeholder(tf.int32, [self.batch_size, None])
        # # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size) 【hidden_dim】
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)


        with tf.variable_scope('rnnlm'):
            # 每次是预测下一个字，因此输出是层维度是所有汉字的数量
            softmax_w = tf.get_variable("softmax_w", [self.rnn_size, data.words_size])
            softmax_b = tf.get_variable("softmax_b", [data.words_size])
            with tf.device("/cpu:0"):
                # 【QUESTION: embedding的词向量内容呢？？？】
                embedding = tf.get_variable(
                    "embedding", [data.words_size, self.rnn_size])
                inputs = tf.nn.embedding_lookup(embedding, self.x_tf)
                outputs, final_state = tf.nn.dynamic_rnn(
                    self.cell, inputs, initial_state=self.initial_state, scope='rnnlm')

        self.output = tf.reshape(outputs, [-1, self.rnn_size])
        self.logits = tf.matmul(self.output, softmax_w) + softmax_b
        # logits:  Tensor("add:0", shape=(?, data.words_sieze), dtype=float32)  [batch_size, data.words_sieze] ?
        # 注意是softmax 不是 argmax
        self.probs = tf.nn.softmax(self.logits)
        # probs:  Tensor("Softmax:0", shape=(?, 372), dtype=float32) [batch_size, data.words_sieze] ?
        self.final_state = final_state
        # 【QUESTION】self.y_tf是Data中的data_y,为什么标签可以是这样？？
        # pred:  Tensor("Reshape_1:0", shape=(?,), dtype=int32)
        # 需要把self.y_tf展开成一列，[batch_size*num_steps]
        pred = tf.reshape(self.y_tf, [-1])
        # seq2seq
        # tf.ones_like 创建一个tensor，左右的元素都设置为1。
        # 返回每个example的交叉熵
        # targets 的shape = [batch_size*num_steps] 难道这里num_steps
        """
            https://blog.csdn.net/appleml/article/details/54017873
            sequence_loss_by_example的做法是，针对logits中的每一个num_step,即[batch_size, vocab_size], 
            对所有vocab_size个预测结果，得出预测值最大的那个类别，与target中的值相比较计算Loss值
        """
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [pred],
                                                [tf.ones_like(pred, dtype=tf.float32)],)

        self.cost = tf.reduce_mean(loss)
        self.learning_rate = tf.Variable(0.0, trainable=False)
        # 返回所有 当前计算图中 在获取变量时未标记 trainable=False 的变量集合
        # 【QUESTION：】为什么以前没有见到过tf.trainable_variables()获取变量？ 这里有什么特殊之处？
        tvars = tf.trainable_variables()
        print ("tvars: ", tvars)
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


def train(data, model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # model_file = tf.train.latest_checkpoint(save_dir)
        # saver.restore(sess, model_file)
        n = 0
        for epoch in range(epochs):
            # 【tf.assign 是一个新遇到的函数】
            sess.run(tf.assign(model.learning_rate, 0.002 * (0.97 ** epoch)))
            pointer = 0
            # data.n_size是batch的数量
            for batche in range(data.n_size):
                n += 1
                # 计算loss的时候要用y
                feed_dict = {model.x_tf: data.x_batches[pointer], model.y_tf: data.y_batches[pointer]}
                pointer += 1
                train_loss, _, _ = sess.run([model.cost, model.final_state, model.train_op], feed_dict=feed_dict)
                sys.stdout.write('\r')
                info = "{}/{} (epoch {}) | train_loss {:.3f}" \
                    .format(epoch * data.n_size + batche,
                            epochs * data.n_size, epoch, train_loss)
                sys.stdout.write(info)
                sys.stdout.flush()
                # save
                if (epoch * data.n_size + batche) % 1000 == 0 \
                        or (epoch == epochs-1 and batche == data.n_size-1):
                    checkpoint_path = os.path.join(model_path, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=n)
                    sys.stdout.write('\n')
                    print("model saved to {}".format(checkpoint_path))
            sys.stdout.write('\n')


def sample(data, model, head=u''):
    def to_word(weights):
        # 【QUESTION】 为什么？
        # (numpy数组,轴参数,输出数组的元素的数据类型，不会用到的参数)
        # >> > a = np.array([[1, 2, 3], [4, 5, 6]])
        # >> > np.cumsum(a)  # array([ 1,  3,  6, 10, 15, 21])
        t = np.cumsum(weights)
        s = np.sum(weights)
        # np.searchsorted(t,a) 在t中查找a的位置，输出下标
        sa = int(np.searchsorted(t, np.random.rand(1) * s))
        return data.id2char(sa)

    for word in head:
        if word not in data.words:
            return u'{} 不在字典中'.format(word)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint(model_path)
        # print(model_file)
        saver.restore(sess, model_file)

        if head:
            print('生成藏头诗 ---> ', head)
            poem = BEGIN_CHAR
            for head_word in head:
                poem += head_word
                # 【注意】这里是对poem进行char2id操作
                x = np.array([list(map(data.char2id, poem))])
                print (x)
                state = sess.run(model.cell.zero_state(1, tf.float32))
                feed_dict = {model.x_tf: x, model.initial_state: state}
                [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
                word = to_word(probs[-1])
                while word != u'，' and word != u'。':
                    poem += word
                    x = np.zeros((1, 1))
                    x[0, 0] = data.char2id(word)
                    [probs, state] = sess.run([model.probs, model.final_state],
                                              {model.x_tf: x, model.initial_state: state})
                    word = to_word(probs[-1])
                poem += word
            return poem[1:]
        else:
            poem = ''
            head = BEGIN_CHAR
            x = np.array([list(map(data.char2id, head))])
            print ("x input: ", x)
            print ("model_init_state: ", model.initial_state)
            state = sess.run(model.cell.zero_state(1, tf.float32))
            print ("state: ", state)
            print ("state: ", state.shape, type(state))
            feed_dict = {model.x_tf: x, model.initial_state: state}
            # 不用计算Loss,因此不用带入y; init_state为什么要生成？ 为什么第一个参数是1【因为每次是一个字？】
            [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
            # print ("prob: ", probs)
            # print ("probs[-1]: ", probs[-1])
            word = to_word(probs[-1])
            # print ("word: ", word)
            # word是生成的句首的字
            while word != END_CHAR:
                poem += word
                x = np.zeros((1, 1))
                # print ("x: ", x)
                # 之前训练模型的原理是，根据前一个字符预测后一个字符，因此需要先有一个句首的字
                x[0, 0] = data.char2id(word)
                [probs, state] = sess.run([model.probs, model.final_state],
                                          {model.x_tf: x, model.initial_state: state})
                word = to_word(probs[-1])
            return poem


def main():

    if args.mode == 'sample':
        # input: 2018-11-21 19-17-38
        infer = True  # True
        data = Data()
        model = Model(data=data, infer=infer)
        print(sample(data, model, head=args.head))
    elif args.mode == 'train':
        infer = False
        data = Data()
        model = Model(data=data, infer=infer)
        print(train(data, model))
    else:
        print(msg)


if __name__ == '__main__':
    main()