# -*- coding: utf-8 -*-

import tensorflow as tf
import cPickle
import os
from sklearn.manifold import TSNE
import pylab


def word2vec(embeddings, train_path, sess):
    with sess.as_default():
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        final_embeddings = normalized_embeddings.eval()
        filename = os.path.join(train_path, "encoding/CNN_result_embeddings")
        cPickle.dump(final_embeddings, open(filename, "wb"))
        return final_embeddings

def display_word2vec(final_embeddings, idx_word_map):
    num_points = 200
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points + 1, :])

    def plot(embeddings, labels):
        assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
        pylab.figure(figsize=(15, 15))
        for i, label in enumerate(labels):
            x, y = embeddings[i, :]
            pylab.scatter(x, y)
            pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                           ha='right', va='bottom')
        pylab.show()
