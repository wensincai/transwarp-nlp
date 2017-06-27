# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
import os, re
from multi_label_classify.cnn_config import CnnConfig
import codecs

config = CnnConfig()

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def build_data(data_folder, vocab, clean_string=False):
    revs = []
    with codecs.open(data_folder, "rb") as f:
        for line in f:
            if line != '\n':
                rev = []
                tags_doc = line.split("|")
                rev.append(tags_doc[1])
                if clean_string:
                    orig_rev = clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum = {"y": tags_doc[0],
                        "text": orig_rev,
                        "num_words": len(orig_rev.split())}
                revs.append(datum)
    return revs

def add_unknown_words(word_vecs, vocab, min_df=1, k=config.vector_size):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)

def get_W(word_vecs, k=config.vector_size):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    idx_word_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float64')
    W[0] = np.zeros(k, dtype='float64')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        idx_word_map[i] = word
        i += 1
    return W, word_idx_map, idx_word_map  # W为一个词向量矩阵 一个word可以通过word_idx_map得到其在W中的索引，进而得到其词向量

def get_data(data_dir):
    vocab = defaultdict(float)
    print("loading data...")
    train_revs = build_data(os.path.join(data_dir, "train/train.txt"), vocab, clean_string=True)
    test_revs = build_data(os.path.join(data_dir, "test/test.txt"), vocab, clean_string=True)

    train_revs = np.random.permutation(train_revs)  # 原始的sample正负样本是分别聚在一起的，这里随机打散

    print("loading word2vec vectors...")
    w2v = {}
    add_unknown_words(w2v, vocab)
    _, word_idx_map, idx_word_map = get_W(w2v)
    return train_revs, test_revs, idx_word_map, word_idx_map

# 一些数据预处理的方法======================================
def get_idx_from_sent(sent, word_idx_map, max_l):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for i, word in enumerate(words):
        if word in word_idx_map and i < max_l:
            x.append(word_idx_map[word])
    while len(x) < max_l:  # 长度不够的，补充0
        x.append(0)
    # 一个训练的一个输入 形式为[0,0,0,0,x11,x12,,,,0,0,0] 向量长度为max_l+2*filter_h-2
    return x


def generate_batch(revs, word_idx_map, minibatch_index, class_num):
    batch_size = config.batch_size
    sentence_length = config.sentence_length
    minibatch_data = revs[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
    batchs = np.ndarray(shape=(batch_size, sentence_length), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, class_num), dtype=np.int32)

    for i in range(batch_size):
        sentece = minibatch_data[i]["text"]
        label_ids = minibatch_data[i]["y"].split(",")

        label = []
        for j in range(class_num): # one-hot格式
            if int(j) in label_ids:
                label.append(1)
            else:
                label.append(0)

        labels[i] = label
        batch = get_idx_from_sent(sentece, word_idx_map, sentence_length)
        batchs[i] = batch
    return batchs, labels

def get_test_batch(revs, word_idx_map, class_num):
    sentence_length = config.sentence_length
    test_szie = len(revs)
    batchs = np.ndarray(shape=[test_szie, sentence_length], dtype=np.int64)
    labels = np.ndarray(shape=[test_szie, class_num], dtype=np.int64)
    for i in range(test_szie):
        sentence = revs[i]["text"]
        label_ids = revs[i]["y"].split(",")

        label = []
        for j in range(class_num):
            if str(j) in label_ids:
                label.append(1)
            else:
                label.append(0)

        labels[i] = label

        batch = get_idx_from_sent(sentence, word_idx_map, sentence_length)
        batchs[i] = batch

    return batchs, labels


if __name__ == "__main__":
    pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    data_dir = os.path.join(pkg_path, "data/multi_label_classify/data")
    get_data(data_dir)

