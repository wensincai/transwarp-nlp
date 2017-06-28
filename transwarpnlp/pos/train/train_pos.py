# -*- coding:utf-8 -*-

from __future__ import unicode_literals
import os

import tensorflow as tf

from transwarpnlp.pos.dataset import rawdata, dataset
from transwarpnlp.pos import pos_model
from transwarpnlp.pos import pos_model_bilstm

from transwarpnlp.pos.config import LargeConfig

def train_lstm(data_dir, train_dir):
    config = LargeConfig()
    eval_config = LargeConfig()
    eval_config.batch_size = 1

    raw_data = rawdata.load_data(data_dir, config.num_steps)
    # train_data, valid_data, test_data, _ = raw_data
    train_word, train_tag, dev_word, dev_tag, test_word, test_tag, _ = raw_data

    train_dataset = dataset.Dataset(train_word, train_tag)
    valid_dataset = dataset.Dataset(dev_word, dev_tag)
    test_dataset = dataset.Dataset(test_word, test_tag)

    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_normal_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope('pos_var_scope', reuse=None, initializer=initializer):
            m = pos_model.POSTagger(is_training=True, config=config)
        with tf.variable_scope('pos_var_scope', reuse=True, initializer=initializer):
            valid_m = pos_model.POSTagger(is_training=False, config=config)
            test_m = pos_model.POSTagger(is_training=False, config=eval_config)

        ckpt = tf.train.get_checkpoint_state(os.path.join(train_dir, "lstm"))

        if ckpt and ckpt.model_checkpoint_path:
            print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
            m.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        for i in range(config.epoch):
            lr_decay = config.lr_decay ** max(float(i - config.max_epoch), 0.0)
            m.assign_lr(sess, config.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))

            train_perplexity = pos_model.run(sess, m, train_dataset, m.train_op, pos_train_dir=train_dir, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            valid_perplexity = pos_model.run(sess, valid_m, valid_dataset, tf.no_op(), pos_train_dir=train_dir)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            train_dataset.reset()
            valid_dataset.reset()

        test_perplexity = pos_model.run(sess, test_m, test_dataset, tf.no_op(), pos_train_dir=train_dir)
        print("Test Perplexity: %.3f" % test_perplexity)

def train_bilstm(data_path, train_dir):
    config = LargeConfig()
    eval_config = LargeConfig()
    eval_config.batch_size = 1

    raw_data = rawdata.load_data(data_path, config.num_steps)
    train_word, train_tag, dev_word, dev_tag, test_word, test_tag, vocabulary = raw_data

    train_dataset = dataset.Dataset(train_word, train_tag)
    valid_dataset = dataset.Dataset(dev_word, dev_tag)
    test_dataset = dataset.Dataset(test_word, test_tag)

    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_normal_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope('pos_var_scope', reuse=None, initializer=initializer):
            m = pos_model_bilstm.POSTagger(is_training=True, config=config)
        with tf.variable_scope('pos_var_scope', reuse=True, initializer=initializer):
            valid_m = pos_model_bilstm.POSTagger(is_training=False, config=config)
            test_m = pos_model_bilstm.POSTagger(is_training=False, config=eval_config)

        ckpt = tf.train.get_checkpoint_state(os.path.join(train_dir, "bilstm"))

        if ckpt:
            print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
            m.saver.restore(sess, tf.train.latest_checkpoint(os.path.join(train_dir, "bilstm")))
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        for i in range(config.epoch):
            lr_decay = config.lr_decay ** max(float(i - config.max_epoch), 0.0)
            m.assign_lr(sess, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))
            train_perplexity = pos_model_bilstm.run(sess, m, train_dataset, m.train_op,
                                         verbose=True, pos_train_dir=train_dir)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = pos_model_bilstm.run(sess, valid_m, valid_dataset, tf.no_op(),
                                                          pos_train_dir=train_dir)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            train_dataset.reset()
            valid_dataset.reset()

        test_perplexity = pos_model_bilstm.run(sess, test_m, test_dataset, tf.no_op(),
                                                     pos_train_dir=train_dir)
        print("Test Perplexity: %.3f" % test_perplexity)