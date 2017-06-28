# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from transwarpnlp.joint_seg_tagger.train import train_seg_tagger
from transwarpnlp.joint_seg_tagger.test import test_seg_tagger

flags = tf.app.flags

flags.DEFINE_string('train_dir', '/data/joint/model', 'The location of model file')

flags.DEFINE_string('train_file', '/data/joint/data/train.txt', 'The location of train file')
flags.DEFINE_string('dev_file', '/data/joint/data/dev.txt', 'The location of dev file')
flags.DEFINE_string('glove_file', '/data/joint/data/glove.txt', 'The location of dev file')

flags.DEFINE_string('test_file', '/data/joint/data/test.txt', 'The location of test file')
flags.DEFINE_string('test_output_file', '/data/joint/data/test_output.txt', 'The location of test output file')

flags.DEFINE_string('predict_file', '/data/joint/data/test.txt', 'The location of predict file')
flags.DEFINE_string('predict_output_file', '/data/joint/data/predict_output.txt', 'The location of predict output file')


flags.DEFINE_string('method', 'train', 'The method to process. Include train, test, tag')

FLAGS = flags.FLAGS

def main(_):

    if FLAGS.method == 'train':
        train_seg_tagger.train_joint(FLAGS.train_dir, FLAGS.train_file, FLAGS.dev_file, FLAGS.glove_file)
    elif FLAGS.method == 'test':
        test_seg_tagger.joint_test(FLAGS.train_dir, FLAGS.test_file, FLAGS.test_output_file)
    else:
        test_seg_tagger.joint_predict(FLAGS.train_dir, FLAGS.predict_file, FLAGS.predict_output_file)


if __name__ == '__main__':
    tf.app.run()
