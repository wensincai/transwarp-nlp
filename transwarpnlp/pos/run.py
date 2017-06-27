# -*- coding:utf-8 -*-

import tensorflow as tf
from pos.train import train_pos
from pos.predict import predict_pos

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

flags.DEFINE_string('train_dir', '/data/pos/ckpt', 'Directory to save the checkpoints and training summaries.')
flags.DEFINE_string('data_dir', '/data/pos/data', 'Directory to save the train and test data.')
flags.DEFINE_string('predict_file', '', 'File to pos tagging.')
flags.DEFINE_string('output_file', '', 'File to save results of predict_file.')

flags.DEFINE_string('process', 'train', 'process to train or predict')
flags.DEFINE_string('method', 'lstm', 'The method to pos tagging by lstm or bilstm')

FLAGS = flags.FLAGS

def main(_):
    if FLAGS.process == 'train':
        if FLAGS.method == "lstm":
            train_pos.train_lstm(FLAGS.data_dir, FLAGS.train_dir)
        else:
            train_pos.train_bilstm(FLAGS.data_dir, FLAGS.train_dir)
    else:
        if FLAGS.method == "lstm":
            predict_pos.predict(FLAGS.data_dir, FLAGS.train_dir, 'lstm', FLAGS.predict_file, FLAGS.output_file)
        else:
            predict_pos.predict(FLAGS.data_dir, FLAGS.train_dir, 'bilstm', FLAGS.predict_file, FLAGS.output_file)


if __name__ == "__main__":
    tf.app.run()