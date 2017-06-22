# -*- coding:utf-8 -*-

from __future__ import unicode_literals
import os

import tensorflow as tf

from ner.dataset import dataset, rawdata
from ner import ner_model, ner_model_bilstm
from ner.config import LargeConfig

pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
train_dir = os.path.join(pkg_path, "data", "ner", "ckpt")

flags = tf.flags
flags.DEFINE_string("ner_train_dir", train_dir, "Training directory.")
flags.DEFINE_string("ner_scope_name", "ner_var_scope", "Variable scope of ner Model")

FLAGS = flags.FLAGS

def train_lstm(data_path):
    if not data_path:
        raise ValueError("No data files found in 'data_path' folder")

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
        with tf.variable_scope(FLAGS.ner_scope_name, reuse=None, initializer=initializer):
            m = ner_model.NERTagger(is_training=True, config=config)
        with tf.variable_scope(FLAGS.ner_scope_name, reuse=True, initializer=initializer):
            valid_m = ner_model.NERTagger(is_training=False, config=config)
            test_m = ner_model.NERTagger(is_training=False, config=eval_config)

        ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.ner_train_dir, "lstm"))

        if ckpt and ckpt.model_checkpoint_path:
            print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
            m.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        for i in range(config.epoch):
            lr_decay = config.lr_decay ** max(float(i - config.learning_rate), 0.0)
            m.assign_lr(sess, config.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))

            train_perplexity = ner_model.run(sess, m, train_dataset, m.train_op,
                                                   ner_train_dir=FLAGS.ner_train_dir, verbose=True)

            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            valid_perplexity = ner_model.run(sess, valid_m, valid_dataset, tf.no_op(),
                                                   ner_train_dir=FLAGS.ner_train_dir)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            train_dataset.reset()
            valid_dataset.reset()

        test_perplexity = ner_model.run(sess, test_m, test_dataset, tf.no_op(),
                                                ner_train_dir=FLAGS.ner_train_dir)
        print("Test Perplexity: %.3f" % test_perplexity)

def train_bilstm(data_path):
    if not data_path:
        raise ValueError("No data files found in 'data_path' folder")

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
        with tf.variable_scope(FLAGS.ner_scope_name, reuse=None, initializer=initializer):
            m = ner_model_bilstm.NERTagger(is_training=True, config=config)
        with tf.variable_scope(FLAGS.ner_scope_name, reuse=True, initializer=initializer):
            valid_m = ner_model_bilstm.NERTagger(is_training=False, config=config)
            test_m = ner_model_bilstm.NERTagger(is_training=False, config=eval_config)

            # CheckPoint State
            ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.ner_train_dir, "bilstm"))
            if ckpt:
                print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
                m.saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.ner_train_dir, "bilstm")))
            else:
                print("Created model with fresh parameters.")
                sess.run(tf.global_variables_initializer())

            for i in range(config.epoch):
                lr_decay = config.lr_decay ** max(float(i - config.max_epoch), 0.0)
                m.assign_lr(sess, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))
                train_perplexity = ner_model_bilstm.run(sess, m, train_dataset, m.train_op,
                                                              ner_train_dir=FLAGS.ner_train_dir, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = ner_model_bilstm.run(sess, valid_m, valid_dataset, tf.no_op(),
                                                              ner_train_dir=FLAGS.ner_train_dir)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

                train_dataset.reset()
                valid_dataset.reset()

            test_perplexity = ner_model_bilstm.run(sess, test_m, test_dataset, tf.no_op(),
                                                         ner_train_dir=FLAGS.ner_train_dir)
            print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
    data_path = os.path.join(pkg_path, "data/ner/data")
    #train_lstm(data_path)
    train_bilstm(data_path)