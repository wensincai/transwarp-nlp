# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from multi_label_classify.dataset import rawdata

class CNNModel(object):
    def __init__(self, config):
        # self._is_training = is_training
        self._config = config
        self._x_in = tf.placeholder(tf.int64, shape=[None, config.sentence_length])
        self._y_in = tf.placeholder(tf.float32, shape=[None, config.class_num])
        self._keep_prob = tf.placeholder(tf.float32)
        # 要学习的词向量矩阵

        self._embeddings = tf.Variable(tf.random_uniform([config.word_idx_map_szie, config.vector_size], -1.0, 1.0))

        self._loss, self._accuracy = self.build_model()

        self._global_step = tf.Variable(0)
        self._learning_rate = tf.train.exponential_decay(1e-4, self._global_step, config.num_epoch, 0.99, staircase=True)  # 学习率递减
        self._train_step = tf.train.AdagradOptimizer(self._learning_rate).minimize(self._loss, global_step=self._global_step)

    @property
    def loss(self):
        return self._loss

    @property
    def config(self):
        return self._config

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def x_in(self):
        return self._x_in

    @property
    def y_in(self):
        return self._y_in

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def global_step(self):
        return self._global_step

    @property
    def train_step(self):
      return self._train_step

    # 卷积图层 第一个卷积
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    # 定义pooling图层
    def max_pool(self, x, filter_h):
        return tf.nn.max_pool(x, ksize=[1, self._config.img_h - filter_h + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

    def build_model(self):
        config = self._config
        # Embedding layer===============================

        # 输入reshape
        x_image_tmp = tf.nn.embedding_lookup(self._embeddings, self._x_in)
        # 输入size: sentence_length*vector_size
        # x_image = tf.reshape(x_image_tmp, [-1,sentence_length,vector_size,1])======>>>>>
        # 将[None, sequence_length, embedding_size]转为[None, sequence_length, embedding_size, 1]
        x_image = tf.expand_dims(x_image_tmp, -1)  # 单通道

        # 定义卷积层，进行卷积操作===================================
        h_conv = []
        for filter_h in config.filter_hs:
            # 卷积的patch大小：vector_size*filter_h, 通道数量：1, 卷积数量：hidden_layer_input_size
            filter_shape = [filter_h, config.vector_size, 1, config.num_filters]
            W_conv1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[config.num_filters]), name="b")
            h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)  # 输出szie: (sentence_length-filter_h+1,1)
            h_conv.append(h_conv1)

        # pool层========================================
        h_pool_output = []
        for h_conv1, filter_h in zip(h_conv, config.filter_hs):
            h_pool1 = self.max_pool(h_conv1, filter_h)  # 输出szie:1
            h_pool_output.append(h_pool1)

        # 全连接层=========================================
        l2_reg_lambda = 0.001
        # 输入reshape
        num_filters_total = config.num_filters * len(config.filter_hs)
        h_pool = tf.concat(h_pool_output, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        h_drop = tf.nn.dropout(h_pool_flat, self._keep_prob)

        W = tf.Variable(tf.truncated_normal([num_filters_total, config.class_num], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[config.class_num]), name="b")
        l2_loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

        scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")  # wx+b
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=self._y_in)
        loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(scores)), tf.round(self._y_in))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return loss, accuracy

def run(sess, model, revs, n_train_batches, word_idx_map, class_num):
    for minibatch_index in np.random.permutation(range(n_train_batches)):  # 随机打散 每次输入的样本的顺序都不一样
        batch_x, batch_y = rawdata.generate_batch(revs, word_idx_map, minibatch_index, class_num)
        feed_dict = {model.x_in: batch_x, model.y_in: batch_y, model.keep_prob: 0.5}
        _, step = sess.run([model.train_step, model.global_step], feed_dict)
