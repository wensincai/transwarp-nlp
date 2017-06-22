# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
import os, re
import pandas as pd

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

def build_data(data_folder, cv=10, clean_string=False):
    revs = []
    vocab = defaultdict(float)
    for label, cfile in enumerate(os.listdir(data_folder)):
        file = os.path.join(data_folder, cfile)
        with open(file, "rb") as f:
            for line in f:
                if line != '\n':
                    rev = []
                    rev.append(line.strip())
                    if clean_string:
                        orig_rev = clean_str(" ".join(rev))
                    else:
                        orig_rev = " ".join(rev).lower()
                    words = set(orig_rev.split())
                    for word in words:
                        vocab[word] += 1
                    datum = {"y": label,
                             "text": orig_rev,
                             "num_words": len(orig_rev.split()),
                             "split": np.random.randint(0, cv)}
                    revs.append(datum)
    return revs, vocab

def add_unknown_words(word_vecs, vocab, min_df=1, k=50):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)

def get_W(word_vecs, k=50):
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
    # W为一个词向量矩阵 一个word可以通过word_idx_map得到其在W中的索引，进而得到其词向量
    return W, word_idx_map, idx_word_map


def get_data(data_dir):
    print("loading data...")
    revs, vocab = build_data(data_dir, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))

    print("loading word2vec vectors...")
    w2v = {}
    add_unknown_words(w2v, vocab)
    W, word_idx_map, idx_word_map = get_W(w2v)

    return revs, W, word_idx_map, idx_word_map, vocab