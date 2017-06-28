# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import random

class Dataset(object):
    def __init__(self, word_data, batch_size, num_skips, skip_window):
        self._data_index = 0
        self._batch_size = batch_size
        self._word_data = word_data
        self._num_skips = num_skips
        self._skip_window = skip_window

    @property
    def data_index(self):
        return self._data_index

    @property
    def word_data(self):
        return self._word_data

    def nextBatch(self):
        batch = np.ndarray(shape=(self._batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self._batch_size, 1), dtype=np.int32)

        span = 2 * self._skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self._word_data[self._data_index])
            self._data_index = (self._data_index + 1) % len(self._word_data)

        for i in range(self._batch_size // self._num_skips):
            target = self._skip_window  # target label at the center of the buffer
            targets_to_avoid = [self._skip_window]
            for j in range(self._num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self._num_skips + j] = buffer[self._skip_window]
                labels[i * self._num_skips + j, 0] = buffer[target]

            buffer.append(self._word_data[self._data_index])
            self._data_index = (self._data_index + 1) % len(self._word_data)
        return batch, labels


