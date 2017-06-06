# -*- coding:utf-8 -*-

import data
import tensorflow as tf
from collections import namedtuple
import numpy as np
from transwarpnlp.textsum.textsum_config import Config

ModelInput = namedtuple('ModelInput',
                        'enc_input dec_input target enc_len dec_len '
                        'origin_article origin_abstract')
textsum_config = Config()

class DataSet(object):
    def __init__(self, enc_inputs, dec_inputs, targets):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._enc_inputs = enc_inputs
        self._dec_inputs = dec_inputs
        self._targets = targets
        self._num_examples = enc_inputs.shape[0]

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def enc_inputs(self):
        return self._enc_inputs

    @property
    def dec_inputs(self):
        return self._dec_inputs

    @property
    def targets(self):
        return self._targets

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._enc_inputs = self._enc_inputs[perm0]
            self._dec_inputs = self._dec_inputs[perm0]
            self._targets = self._targets[perm0]

        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            enc_rest_part = self._enc_inputs[start:self._num_examples]
            dec_rest_part = self._dec_inputs[start:self._num_examples]
            target_rest_part = self._targets[start:self._num_examples]
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._enc_inputs = self._enc_inputs[perm]
                self._dec_inputs = self._dec_inputs[perm]
                self._targets = self._targets[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            enc_new_part = self._enc_inputs[start:end]
            dec_new_part = self._dec_inputs[start:end]
            target_new_part = self._targets[start:end]

            return  np.concatenate((enc_rest_part, enc_new_part), axis=0),\
                    np.concatenate((dec_rest_part, dec_new_part), axis=0),\
                    np.concatenate((target_rest_part, target_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._enc_inputs[start:end], self._dec_inputs[start:end], self._targets[start:end]

def read_data_sets(train_dir, vocab, hps):
    start_id = vocab.WordToId(data.SENTENCE_START)
    end_id = vocab.WordToId(data.SENTENCE_END)
    pad_id = vocab.WordToId(data.PAD_TOKEN)
    articles_abstracts = data.getArticlesAndAbstracts(train_dir)
    enc_inputs = np.zeros((len(articles_abstracts), hps.enc_timesteps), dtype=np.int32)
    dec_inputs = np.zeros((len(articles_abstracts), hps.dec_timesteps), dtype=np.int32)
    targets = np.zeros((len(articles_abstracts), hps.dec_timesteps), dtype=np.int32)
    for index, (article, abstract) in enumerate(articles_abstracts):
        # Use the <s> as the <GO> symbol for decoder inputs.
        enc_input = []
        dec_input = [start_id]
        enc_input += data.GetWordIds(article, vocab)
        dec_input += data.GetWordIds(abstract, vocab)

        enc_input[:] = data.GetWordIds(article, vocab)
        dec_input[1:] = data.GetWordIds(abstract, vocab)

        enc_input = enc_input[:hps.enc_timesteps]
        dec_input = dec_input[:hps.dec_timesteps]

        # targets is dec_inputs without <s> at beginning, plus </s> at end
        target = dec_input[1:]
        target.append(end_id)

        # Now len(enc_inputs) should be <= enc_timesteps, and
        # len(targets) = len(dec_inputs) should be <= dec_timesteps

        #enc_input_len = len(enc_inputs)
        #dec_output_len = len(targets)

        # Pad if necessary
        while len(enc_input) < hps.enc_timesteps:
            enc_input.append(pad_id)
        while len(dec_input) < hps.dec_timesteps:
            dec_input.append(end_id)
        while len(target) < hps.dec_timesteps:
            target.append(end_id)

        enc_inputs[index] = enc_input
        dec_inputs[index] = dec_input
        targets[index] = target

    return DataSet(enc_inputs, dec_inputs, targets)