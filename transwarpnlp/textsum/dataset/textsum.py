# -*- coding:utf-8 -*-

import data
import os
import numpy as np
from transwarpnlp.textsum.textsum_config import Config
from transwarpnlp.textsum import seq2seq_attention_model

pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

textsum_config = Config()

class DataSet(object):
    def __init__(self, enc_inputs, dec_inputs, targets, origin_articles, origin_abstracts):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._enc_inputs = enc_inputs
        self._dec_inputs = dec_inputs
        self._targets = targets
        self._origin_articles = origin_articles
        self._origin_abstracts = origin_abstracts
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

    @property
    def origin_abstracts(self):
        return self._origin_abstracts

    @property
    def origin_articles(self):
        return self._origin_articles

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch

        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            print("Finished epoch %d"  % self._epochs_completed)
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            enc_rest_part = self._enc_inputs[start:self._num_examples]
            dec_rest_part = self._dec_inputs[start:self._num_examples]
            target_rest_part = self._targets[start:self._num_examples]
            origin_article_rest_part = self._origin_articles[start:self._num_examples]
            origin_abstract_rest_part = self._origin_abstracts[start:self._num_examples]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            enc_new_part = self._enc_inputs[start:end]
            dec_new_part = self._dec_inputs[start:end]
            target_new_part = self._targets[start:end]
            origin_article_new_part = self._origin_articles[start:end]
            origin_abstract_new_part = self._origin_abstracts[start:end]

            return  np.concatenate((enc_rest_part, enc_new_part), axis=0),\
                    np.concatenate((dec_rest_part, dec_new_part), axis=0),\
                    np.concatenate((target_rest_part, target_new_part), axis=0), \
                    origin_article_rest_part + origin_article_new_part, \
                    origin_abstract_rest_part + origin_abstract_new_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._enc_inputs[start:end], self._dec_inputs[start:end],\
                   self._targets[start:end], self._origin_articles[start:end],\
                   self._origin_abstracts[start:end]

def read_data_sets(train_dir, vocab, hps):
    start_id = vocab.WordToId(data.SENTENCE_START)
    end_id = vocab.WordToId(data.SENTENCE_END)
    pad_id = vocab.WordToId(data.PAD_TOKEN)
    articles_abstracts = data.getArticlesAndAbstracts(train_dir)
    enc_inputs = np.zeros((len(articles_abstracts), hps.enc_timesteps), dtype=np.int32)
    dec_inputs = np.zeros((len(articles_abstracts), hps.dec_timesteps), dtype=np.int32)
    targets = np.zeros((len(articles_abstracts), hps.dec_timesteps), dtype=np.int32)

    origin_articles = []
    origin_abstract = []
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
        origin_articles.append(article)
        origin_abstract.append(abstract)

    return DataSet(enc_inputs, dec_inputs, targets, origin_articles, origin_abstract)


if __name__ == "__main__":
    vocab_path = os.path.join(pkg_path, "data/textsum/data/vocab.txt")
    data_path = os.path.join(pkg_path, "data/textsum/data/train.txt")
    hps = seq2seq_attention_model.HParams(
        mode='train',  # train, eval, decode
        min_lr=0.01,  # min learning rate.
        lr=0.15,  # learning rate
        batch_size=2,
        enc_layers=2,
        enc_timesteps=100,
        dec_timesteps=20,
        min_input_len=2,  # discard articles/summaries < than this
        num_hidden=256,  # for rnn cell
        emb_dim=128,  # If 0, don't use embedding
        max_grad_norm=2,
        num_softmax_samples=0)  # If 0, no sampled softmax.
    vocab = data.Vocab(vocab_path, 10000)
    dataset = read_data_sets(data_path, vocab, hps)

    article_batch, abstract_batch, targets, source_article, source_abstract = dataset.next_batch(hps.batch_size)
    print(article_batch)
    print(source_article)

    print(abstract_batch)
    print(source_abstract)

    print(targets)
    print("\n")

    article_batch1, abstract_batch1, targets1, _, _ = dataset.next_batch(hps.batch_size)
    print(article_batch1)
    print(abstract_batch1)
    print(targets1)

