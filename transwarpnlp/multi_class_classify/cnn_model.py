# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import model_cnn
import time, os

class CNNModel(object):
    def __init__(self, is_training, config):
        self._is_training = is_training
        self._batch_size = config.batch_size
        self._config = config
        self._loss = None
        self._accuracy = None
        self._embeddings = None
        self._x_in = None
        self._y_in = None
        self._keep_prob = None

    # 卷积图层 第一个卷积
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    # 定义pooling图层
    def max_pool(self, x, filter_h):
        return tf.nn.max_pool(x, ksize=[1, self._config.img_h - filter_h + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

    def build_model(self):
        config = self._config
        self._x_in = tf.placeholder(tf.int64, shape=[None, config.sentence_length], name="input_x")
        self._y_in = tf.placeholder(tf.int64, [None], name="input_y")
        self._keep_prob = tf.placeholder(tf.float32)

        # Embedding layer===============================
        # 要学习的词向量矩阵
        embeddings = tf.Variable(tf.random_uniform([config.word_idx_map_szie, config.vector_size], -1.0, 1.0))
        # 输入reshape
        x_image_tmp = tf.nn.embedding_lookup(embeddings, self._x_in)
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

        W = tf.Variable(tf.truncated_normal([num_filters_total, 2], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
        l2_loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

        scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")  # wx+b
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=self._y_in)
        loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        correct_prediction = tf.equal(tf.argmax(scores, 1), self._y_in)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self._loss = loss
        self._accuracy = accuracy
        self._embeddings = embeddings
        return loss, accuracy, embeddings

    def run(self, sess, train_step, train_path, revs, word_idx_map, global_step):

        n_batches = len(revs) / self._config.batch_size
        n_train_batches = int(np.round(n_batches * 0.9))

        # summaries,====================
        timestamp = str(int(time.time()))
        out_dir = os.path.join(train_path, "summary", timestamp)
        print("Writing to {}\n".format(out_dir))
        loss_summary = tf.summary.scalar("loss", self._loss)
        acc_summary = tf.summary.scalar("accuracy", self._accuracy)
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        batch_x_test, batch_y_test = model_cnn.get_test_batch(revs, word_idx_map)

        for minibatch_index in np.random.permutation(range(n_train_batches)):
            batch_x, batch_y = model_cnn.generate_batch(revs, word_idx_map, minibatch_index)
            # train_step.run(feed_dict={x_in: batch_x, y_in: batch_y, keep_prob: 0.5})
            feed_dict = {self._x_in: batch_x, self._y_in: batch_y, self._keep_prob: 0.5}
            _, step, summaries = sess.run([train_step, global_step, train_summary_op], feed_dict)
            train_summary_writer.add_summary(summaries, step)
        test_accuracy = self._accuracy.eval(feed_dict={self._x_in: batch_x_test, self._y_in: batch_y_test, self._keep_prob: 1.0})
        current_step = tf.train.global_step(sess, global_step)

        return current_step, test_accuracy


