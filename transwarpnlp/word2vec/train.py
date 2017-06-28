# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from transwarpnlp.word2vec.dataset import rawdata, dataset
from transwarpnlp.word2vec import model

flags = tf.app.flags

flags.DEFINE_string('data_file', '/data/word2vec/train.txt', 'Train file url.')
flags.DEFINE_string('output_file', '/data/word2vec/embedding.txt', 'Embedding file.')

flags.DEFINE_integer('vocabulary_size', 50000, 'Max vocabulary size')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('embedding_size', 128, 'Embedding size')
flags.DEFINE_integer('skip_window', 1, 'How many words to consider left and right')
flags.DEFINE_integer('num_skips', 2, 'How many times to reuse an input to generate a label')

flags.DEFINE_integer('valid_size', 16, 'Random set of words to evaluate similarity on')
flags.DEFINE_integer('valid_window', 100, 'Only pick dev samples in the head of the distribution.')
flags.DEFINE_integer('num_sampled', 64, 'Number of negative examples to sample')

flags.DEFINE_integer('num_steps', 100001, 'Number of train step')

FLAGS = flags.FLAGS

def main(_):
    # 读取词典
    vocabulary = rawdata.read_data(FLAGS.data_file)
    print('Data size', len(vocabulary))

    data, count, dictionary, reverse_dictionary =\
        rawdata.build_dataset(vocabulary, FLAGS.vocabulary_size)

    del vocabulary  # Hint to reduce memory.
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    result = dataset.Dataset(data, FLAGS.batch_size, FLAGS.num_skips, FLAGS.skip_window)
    batch, labels = result.nextBatch()

    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]],
              '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

    graph = tf.Graph()

    with graph.as_default():
        w2vModel = model.Word2VecModel(FLAGS.batch_size, FLAGS.valid_window, FLAGS.valid_size,
                                       FLAGS.vocabulary_size, FLAGS.embedding_size, FLAGS.num_sampled)

        init = tf.global_variables_initializer()

        # begin train
        with tf.Session(graph=graph) as session:
            # We must initialize all variables before we use them.
            init.run()
            print('Initialized')
            average_loss = 0
            for step in xrange(FLAGS.num_steps):
                batch_inputs, batch_labels = result.nextBatch()
                feed_dict = {w2vModel.train_inputs: batch_inputs, w2vModel.train_labels: batch_labels}
                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([w2vModel.optimizer, w2vModel.loss], feed_dict=feed_dict)
                average_loss += loss_val
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = w2vModel.similarity.eval()
                    for i in xrange(FLAGS.valid_size):
                        valid_word = reverse_dictionary[w2vModel.valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in xrange(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = '%s %s,' % (log_str, close_word)
                        print(log_str)

            final_embeddings = w2vModel.normalized_embeddings.eval()

            rawdata.save_embeding(final_embeddings, reverse_dictionary, FLAGS.output_file)

            # try:
            #     # pylint: disable=g-import-not-at-top
            #     from sklearn.manifold import TSNE
            #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            #     plot_only = 500
            #     low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
            #     labels = [reverse_dictionary[i] for i in xrange(plot_only)]
            #     rawdata.plot_with_labels(low_dim_embs, labels)
            # except ImportError:
            #     print('Please install sklearn, matplotlib, and scipy to show embeddings.')

if __name__ == "__main__":
    tf.app.run()