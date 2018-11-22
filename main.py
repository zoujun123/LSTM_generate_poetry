# coding:utf-8
import data_process
from mymodel import  Poetry_LSTM
import argparse
import numpy as np
import tensorflow as tf
import time
import os
from utils import get_logger

## hyperparameters
parser = argparse.ArgumentParser(description='Generate Poetry Task')
parser.add_argument('--batch_size', type=int, default=5, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=10, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=100, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--embedding_dim', type=int, default=100, help='random init char embedding_dim')
parser.add_argument('--layers', type=int, default=2, help='layers of rnn')
parser.add_argument('--mode', type=str, default='test', help='train/test')
# parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
# parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
# parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
# parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
# parser.add_argument('--demo_model', type=str, default='2018-11-11 15-55-22', help='model for test and demo')
args = parser.parse_args()


# x_train 是 id 化后的句子

poetry_vectors, words_size, char2id_dict, id2char_dict = data_process.construct_poetry_idvector()
embeddings = np.random.uniform(-0.25, 0.25, (words_size, args.embedding_dim)) # word_embeddings

paths = {}
# 每次进行训练时生成一个时间戳
if args.mode == 'train':
    timestamp = str(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
else:
    timestamp = input("请输入要restore的模型生成时间戳,例如2018-11-22 19-53-08:")
    # timestamp = "2018-11-22 19-53-08"
# 总体存储在./data_path_save目录下
model_path = os.path.join('.', "log", timestamp + "/")
if not os.path.exists(model_path): os.makedirs(model_path)

# (self, args, embeddings, tag2label, vocab, paths, layers):
if args.mode == 'train':
    infer = False
    model = Poetry_LSTM(args, embeddings, char2id_dict,id2char_dict, words_size, model_path, infer)
    model.build_graph()
    model.train(poetry_vectors)
elif args.mode == 'test':
    # 获取check_point file的路径
    model_path = tf.train.latest_checkpoint(model_path)
    infer = True
    # 2018-11-21 20-53-56
    model = Poetry_LSTM(args, embeddings, char2id_dict,id2char_dict, words_size, model_path, infer)
    model.build_graph()
    poem = model.sample()
    print (poem)