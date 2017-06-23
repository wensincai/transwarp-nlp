# -*- coding: utf-8 -*-

import numpy as np
import cPickle
from collections import defaultdict
import os, re
import pandas as pd
from transwarpnlp.segment import segmenter

pkg_path = os.path.dirname(os.path.dirname(os.getcwd()))

vector_size = 50

def segment(all_the_text):
    re = ""
    relist = ""
    words = segmenter.seg(all_the_text)
    count = 0
    for w in words:

        if len(w) > 1 and w >= u'/u4e00' and w <= u'\u9fa5':
            re = re + " " + w
            count = count + 1
        if count % 100 == 0:
            re = re.replace("\n", " ")
            relist = relist + "\n" + re
            re = ""
            count = count + 1
    re = re.replace("\n", " ").replace("\r", " ")
    if len(relist) > 1 and len(re) > 40:
        relist = relist + "\n" + re
    elif len(re) > 40:
        relist = re
    relist = relist + "\n"
    relist = relist.replace("\r\n", "\n").replace("\n\n", "\n")

    return relist


def handleTrainData(input_path, output_file):
    fw = open(output_file, "a")
    for filename in os.listdir(input_path):
        print(filename)
        file_object = open(input_path + "/" + filename)
        try:
            all_the_text = file_object.read()
            all_the_text = all_the_text.decode("utf-8")
            pre_text = segment(all_the_text)
            if len(pre_text) > 30:
                fw.write(pre_text.encode("utf-8"))
        except Exception:
            print(Exception.message)
        finally:
            file_object.close()


"""
输入参数：
    data_folder：训练数据路径
    cv：cv值
    clean_string：是否清洗数据
返回值：
    revs：原始文本，文本中单词个数，所属cv值
    vocab：单词表（包括词和词频）
"""
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

def get_W(word_vecs, k=vector_size):
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


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    ''''
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float64').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float64')  
            else:
                f.read(binary_len)
    '''
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=vector_size):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


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


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def createTrainData(data_folder, result_path):
    w2v_file = ""

    print("loading data...")
    revs, vocab = build_data(data_folder, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    print("loading word2vec vectors...")
    w2v = load_bin_vec(w2v_file, vocab)
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map, idx_word_map = get_W(w2v)  # 利用一个构建好的word2vec向量来初始化词向量矩阵及词-向量映射表
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)  # 得到一个{word:word_vec}词典
    W2, _, _ = get_W(rand_vecs)  # 构建一个随机初始化的W2词向量矩阵
    cPickle.dump([revs, W, W2, word_idx_map, idx_word_map, vocab], open(result_path, "wb"))
    print("train data created!")

if __name__ == "__main__":
    input_path = os.path.join(pkg_path, "data/source/sogo/C000010")
    output_file = os.path.join(pkg_path, "data/source/sogo", "C000010.txt")
    handleTrainData(input_path, output_file)

    data_folder = os.path.join(pkg_path, "data/multi_class_classify/data")
    result_path = os.path.join(pkg_path, "data/multi_class_classify/model/mr.txt")
    createTrainData(data_folder, result_path)