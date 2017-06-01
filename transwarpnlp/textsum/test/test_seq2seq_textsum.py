# -*- coding:utf-8 -*-

"""test a seq2seq model.
WORK IN PROGRESS.
Implement "Abstractive Text Summarization using Sequence-to-sequence RNNS and Beyond."
"""
from __future__ import unicode_literals

import os
import tensorflow as tf
from transwarpnlp.textsum.textsum_config import Config
from transwarpnlp.textsum import data, batch_reader
from transwarpnlp.textsum import seq2seq_attention_model, seq2seq_attention_decode

pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

textsum_config = Config()

def test_textsum(vocab_path, data_path, decode_dir, log_root):
    vocab = data.Vocab(vocab_path, 10000)
    # Check for presence of required special tokens.
    assert vocab.CheckVocab(data.PAD_TOKEN) > 0
    assert vocab.CheckVocab(data.UNKNOWN_TOKEN) >= 0
    assert vocab.CheckVocab(data.SENTENCE_START) > 0
    assert vocab.CheckVocab(data.SENTENCE_END) > 0

    batch_size = textsum_config.beam_size

    hps = seq2seq_attention_model.HParams(
        mode='decode',  # train, eval, decode
        min_lr=0.01,  # min learning rate.
        lr=0.15,  # learning rate
        batch_size=batch_size,
        enc_layers=2,
        enc_timesteps=100,
        dec_timesteps=20,
        min_input_len=2,  # discard articles/summaries < than this
        num_hidden=256,  # for rnn cell
        emb_dim=128,  # If 0, don't use embedding
        max_grad_norm=2,
        num_softmax_samples=0)  # If 0, no sampled softmax.

    batcher = batch_reader.Batcher(data_path, vocab, hps, textsum_config.article_key,
                                   textsum_config.abstract_key, textsum_config.max_article_sentences,
                                   textsum_config.max_abstract_sentences, bucketing=textsum_config.use_bucketing,
                                   truncate_input=textsum_config.truncate_input)
    tf.set_random_seed(textsum_config.random_seed)

    # decode_mdl_hps = hps
    # Only need to restore the 1st step and reuse it since
    # we keep and feed in state for each step's output.
    decode_mdl_hps = hps._replace(dec_timesteps=1)
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        decode_mdl_hps, vocab)

    decoder = seq2seq_attention_decode.BSDecoder(model, batcher, hps, vocab, decode_dir, log_root)
    decoder.DecodeLoop()

if __name__ == "__main__":
    vocab_path = os.path.join(pkg_path, "data/textsum/data/vocab.txt")
    data_path = os.path.join(pkg_path, "data/textsum/data/test.txt")
    test_dir = os.path.join(pkg_path, "data/textsum/test/")
    log_root = os.path.join(pkg_path, "data/textsum/ckpt/")
    test_textsum(vocab_path, data_path, test_dir, log_root)