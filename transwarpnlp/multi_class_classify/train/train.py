# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os
from multi_class_classify.cnn_config import CnnConfig
from multi_class_classify import cnn_model
from multi_class_classify.dataset import rawdata

config = CnnConfig()
pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

def train_cnn_classfier(train_path):
    x = rawdata.get_data(os.path.join(train_path, "data"))
    # 读取出预处理后的数据 revs {"y":label,"text":"word1 word2 ..."}
    #                          word_idx_map["word"]==>index
    #                        vocab["word"]==>frequency
    revs, _, word_idx_map, idx_word_map, vocab = x[0], x[1], x[2], x[3], x[4]

    revs = np.random.permutation(revs)  # 原始的sample正负样本是分别聚在一起的，这里随机打散

    # 开始定义模型============================================
    with tf.Graph().as_default(), tf.Session().as_default() as sess:
        # 构建模型
        model = cnn_model.CNNModel(is_training=True, config=config)
        loss, _, embeddings = model.build_model()

        # 训练模型========================================
        num_steps = config.num_epoch
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(1e-4, global_step, num_steps, 0.99, staircase=True)  # 学习率递减
        train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

        checkpoint_dir = os.path.join(train_path, "ckpt")
        checkpoint_prefix = os.path.join(checkpoint_dir, "classify")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables())

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        for i in range(num_steps):
            current_step, test_accuracy = model.run(sess, train_step, train_path, revs, word_idx_map, global_step)
            print("Update step %d, test accuracy %g" % (current_step, test_accuracy))
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))

        return embeddings, sess, idx_word_map


if __name__ == "__main__":
    train_path = os.path.join(pkg_path, "data/multi_class_classify")
    embeddings, sess, idx_word_map = train_cnn_classfier(train_path)
    #final_embeddings = model_cnn.word2vec(embeddings, train_path, sess)
    # cnn_classfier.display_word2vec(final_embeddings, idx_word_map)