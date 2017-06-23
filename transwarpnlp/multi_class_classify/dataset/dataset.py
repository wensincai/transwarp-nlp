# -*- coding: utf-8 -*-

# Reading POS data input_data and target_data

"""Utilities for reading POS train, dev and test files files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode

import numpy as np

class Dataset(object):
    def __init__(self, sentences, labels):
        self._start = 0
        self._cursor = 0
        self._num_samples = sentences.shape[0]
        self._sentences = sentences
        self._labels = labels

    @property
    def sentences(self):
        return self._sentences

    @property
    def labels(self):
        return self._labels

    @property
    def num_samples(self):
        return self._num_samples

    def hasNext(self):
        return self._cursor < self._num_samples

    def reset(self):
        self._cursor = 0
        self._start = 0

    def nextBatch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        self._start = self._cursor
        self._cursor += batch_size
        if self._start + batch_size > self._num_samples:
            rest_num_samples = self._num_samples - self._start
            word_batch = np.zeros((batch_size, self._sentences.shape[1]), dtype=np.int32)
            tag_batch = np.zeros((batch_size), dtype=np.int32)
            word_batch[0:rest_num_samples] = self._sentences[self._start:self._num_samples]
            tag_batch[0:rest_num_samples] = self.labels[self._start:self._num_samples]

            return word_batch, tag_batch
        else:
            end = self._cursor
            return self._sentences[self._start:end], self._labels[self._start:end]

