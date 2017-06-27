# -*- coding:utf-8 -*-

import tensorflow as tf
from ner.train import train_ner
from ner.predict import predict_ner

r'''
Example usage:
    ./run \
        --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --predict_file=path/to/predict_file \
        --output_file=path/to/output_file \
        --process=train or predict \
        --method=lstm or bilstm
'''
flags = tf.app.flags

flags.DEFINE_string('train_dir', '/data/ner/ckpt', 'Directory to save the checkpoints and training summaries.')
flags.DEFINE_string('data_dir', '/data/ner/data', 'Directory to save the train and test data.')
flags.DEFINE_string('predict_file', '', 'File to recognition ne.')
flags.DEFINE_string('output_file', '', 'File to save results of predict_file.')

flags.DEFINE_string('process', 'train', 'process to train or predict')
flags.DEFINE_string('method', 'lstm', 'The method to recognition name entity by lstm or bilstm')

FLAGS = flags.FLAGS

def main(_):
    if FLAGS.process == 'train':
        if FLAGS.method == "lstm":
            train_ner.train_lstm(FLAGS.data_dir, FLAGS.train_dir)
        else:
            train_ner.train_bilstm(FLAGS.data_dir, FLAGS.train_dir)
    else:
        if FLAGS.method == "lstm":
            predict_ner.predict(FLAGS.data_dir, FLAGS.train_dir, 'lstm', FLAGS.predict_file, FLAGS.output_file)
        else:
            predict_ner.predict(FLAGS.data_dir, FLAGS.train_dir, 'bilstm', FLAGS.predict_file, FLAGS.output_file)


if __name__ == "__main__":
    tf.app.run()


