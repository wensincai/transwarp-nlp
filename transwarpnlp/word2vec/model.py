# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math

class Word2VecModel(object):
    def __init__(self, batch_size, valid_window, valid_size, vocabulary_size, embedding_size, num_sampled):
        self._valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        # Input data.
        self._train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        self._train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(self._valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, self._train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
        self._loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=self._train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))

        # Construct the SGD optimizer using a learning rate of 1.0.
        self._optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self._loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        self._normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            self._normalized_embeddings, valid_dataset)
        self._similarity = tf.matmul(
            valid_embeddings, self._normalized_embeddings, transpose_b=True)

    @property
    def train_inputs(self):
        return self._train_inputs

    @property
    def train_labels(self):
        return self._train_labels

    @property
    def valid_examples(self):
        return self._valid_examples

    @property
    def loss(self):
        return self._loss

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def normalized_embeddings(self):
        return self._normalized_embeddings

    @property
    def similarity(self):
        return self._similarity
