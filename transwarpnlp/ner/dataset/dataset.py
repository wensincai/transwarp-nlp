#!/usr/bin/python
# -*- coding:utf-8 -*-
# Reading POS data input_data and target_data

"""Utilities for reading POS train, dev and test files files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode

import numpy as np

class Dataset(object):
    def __init__(self, word_data, tag_data):
        self._start = 0
        self._cursor = 0
        self._num_samples = word_data.shape[0]
        self._word_data = word_data
        self._tag_data = tag_data

    @property
    def word_data(self):
        return self._word_data

    @property
    def tag_data(self):
        return self._tag_data

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
        if self._start + batch_size > self._num_samples:
            rest_num_samples = self._num_samples - self._start
            word_batch = np.zeros((batch_size, self._word_data.shape[1]), dtype=np.int32)
            tag_batch = np.zeros((batch_size, self._word_data.shape[1]), dtype=np.int32)
            word_batch[0:rest_num_samples] = self._word_data[self._start:self._num_samples]
            tag_batch[0:rest_num_samples] = self._tag_data[self._start:self._num_samples]

            return word_batch, tag_batch
        else:
            self._cursor += batch_size
            end = self._cursor
            return self._word_data[self._start:end], self._tag_data[self._start:end]