# -*- coding:utf-8 -*-

"""test a seq2seq model.
WORK IN PROGRESS.
Implement "Abstractive Text Summarization using Sequence-to-sequence RNNS and Beyond."
"""
from __future__ import unicode_literals

import os
import time
import tensorflow as tf
import numpy as np
from transwarpnlp.textsum.dataset import data, textsum
from transwarpnlp.textsum.textsum_config import Config
from transwarpnlp.textsum import seq2seq_attention_model
from transwarpnlp.textsum.train.train_seq2seq_textsum import _RunningAvgLoss

pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

textsum_config = Config()

def _Eval(model, dataset, hps, eval_dir, log_root, vocab=None):
    """Runs model eval."""
    model.build_graph()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(eval_dir)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    running_avg_loss = 0
    step = 0
    while True:
        time.sleep(textsum_config.eval_interval_secs)
        try:
            ckpt_state = tf.train.get_checkpoint_state(log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue

        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', log_root)
            continue

        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        loss_weights = np.ones((hps.batch_size, hps.dec_timesteps), dtype=np.float32)
        article_lens = np.full(hps.batch_size, fill_value=hps.enc_timesteps, dtype=np.int32)
        abstract_lens = np.full(hps.batch_size, fill_value=hps.dec_timesteps, dtype=np.int32)

        article_batch, abstract_batch, targets, _, _ = dataset.next_batch(hps.batch_size)

        (summaries, loss, train_step) = model.run_eval_step(
            sess, article_batch, abstract_batch, targets, article_lens,
            abstract_lens, loss_weights)

        tf.logging.info(
            'article:  %s',
            ' '.join(data.Ids2Words(article_batch[0][:].tolist(), vocab)))
        print('article:' + ' '.join(data.Ids2Words(article_batch[0][:].tolist(), vocab)))
        tf.logging.info(
            'abstract: %s',
            ' '.join(data.Ids2Words(abstract_batch[0][:].tolist(), vocab)))
        print('abstract:' + ' '.join(data.Ids2Words(abstract_batch[0][:].tolist(), vocab)))

        summary_writer.add_summary(summaries, train_step)
        running_avg_loss = _RunningAvgLoss(
            running_avg_loss, loss, summary_writer, train_step)
        if step % 100 == 0:
            summary_writer.flush()


def valid_textsum(vocab_path, data_path, eval_dir, log_root):
    vocab = data.Vocab(vocab_path, 10000)
    # Check for presence of required special tokens.
    assert vocab.CheckVocab(data.PAD_TOKEN) > 0
    assert vocab.CheckVocab(data.UNKNOWN_TOKEN) >= 0
    assert vocab.CheckVocab(data.SENTENCE_START) > 0
    assert vocab.CheckVocab(data.SENTENCE_END) > 0

    batch_size = 2

    hps = seq2seq_attention_model.HParams(
        mode='eval',  # train, eval, decode
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

    dataset = textsum.read_data_sets(data_path, vocab, hps)

    tf.set_random_seed(textsum_config.random_seed)

    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab)
    _Eval(model, dataset, hps, eval_dir, log_root, vocab=vocab)

if __name__ == "__main__":
    vocab_path = os.path.join(pkg_path, "data/textsum/data/vocab.txt")
    data_path = os.path.join(pkg_path, "data/textsum/data/dev.txt")
    eval_dir = os.path.join(pkg_path, "data/textsum/eval/")
    log_root = os.path.join(pkg_path, "data/textsum/ckpt/")
    valid_textsum(vocab_path, data_path, eval_dir, log_root)