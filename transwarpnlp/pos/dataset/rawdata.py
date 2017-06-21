#!/usr/bin/python
# -*- coding:utf-8 -*-
# Reading POS data input_data and target_data

"""Utilities for reading POS train, dev and test files files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode
import os
import codecs
import re
import collections
import numpy as np

UNKNOWN = "*"
DELIMITER = "\s+" # line delimiter

def _read_file(filename):
    sentences = []  # list(list(str))
    words = []
    file = codecs.open(filename, encoding='utf-8')
    for line in file:
        wordsplit = re.split(DELIMITER, line.replace("\n", ""))
        sentences.append(wordsplit)  # list(list(str))
        words.extend(wordsplit)  # list(str)
    return words, sentences

# input format word2/tag2 word2/tag2
def _split_word_tag(data):
    word = []
    tag = []
    for word_tag_pair in data:
        pairs = word_tag_pair.split("/")
        if (len(pairs)==2):
            # word or tag not equal to ""
            if (len(pairs[0].strip())!=0 and len(pairs[1].strip())!=0):
                word.append(pairs[0])
                tag.append(pairs[1])
    return word, tag

def _build_vocab(filename):
    words, sentences = _read_file(filename)
    word, tag = _split_word_tag(words)
    # split word and tag data

    # word dictionary
    word.append(UNKNOWN)
    counter_word = collections.Counter(word)
    count_pairs_word = sorted(counter_word.items(), key=lambda x: (-x[1], x[0]))

    wordlist, _ = list(zip(*count_pairs_word))
    word_to_id = dict(zip(wordlist, range(len(wordlist))))

    # tag dictionary
    tag.append(UNKNOWN)
    counter_tag = collections.Counter(tag)
    count_pairs_tag = sorted(counter_tag.items(), key=lambda x: (-x[1], x[0]))

    taglist, _ = list(zip(*count_pairs_tag))
    tag_to_id = dict(zip(taglist, range(len(taglist))))
    return word_to_id, tag_to_id

def _save_vocab(dict, path):
  # save utf-8 code dictionary
  file = codecs.open(path, "w", encoding='utf-8')
  for k, v in dict.items():
    # k is unicode, v is int
    line = k + "\t" + str(v) + "\n" # unicode
    file.write(line)

def _read_vocab(path):
  # read utf-8 code
  file = codecs.open(path, encoding='utf-8')
  vocab_dict = {}
  for line in file:
    pair = line.replace("\n","").split("\t")
    vocab_dict[pair[0]] = int(pair[1])
  return vocab_dict

def sentence_to_word_ids(data_path, words):
  word_to_id = _read_vocab(os.path.join(data_path, "word_to_id"))
  wordArray = [word_to_id[w] if w in word_to_id else word_to_id[UNKNOWN] for w in words]
  return wordArray

def word_ids_to_sentence(data_path, ids):
  tag_to_id = _read_vocab(os.path.join(data_path, "tag_to_id"))
  id_to_tag = {id:tag for tag, id in tag_to_id.items()}
  tagArray = [id_to_tag[i] if i in id_to_tag else id_to_tag[0] for i in ids]
  return tagArray

def _file_to_word_ids(filename, word_to_id, tag_to_id, num_step):
  _, sentences = _read_file(filename)
  word_array = np.zeros((len(sentences), num_step), np.int32)
  tag_array = np.zeros((len(sentences), num_step), np.int32)

  for index, sentence in enumerate(sentences):
      words, tags = _split_word_tag(sentence)
      word_ids = [word_to_id.get(w, word_to_id[UNKNOWN]) for w in words]
      tag_ids = [tag_to_id.get(w, tag_to_id[UNKNOWN]) for w in tags]

      if len(words) >= num_step:
          word_ids = word_ids[:num_step]
          tag_ids = tag_ids[:num_step]
      else:
          rest_len = num_step - len(words)
          word_ids.extend([word_to_id[UNKNOWN]] * rest_len)
          tag_ids.extend([tag_to_id[UNKNOWN]] * rest_len)
      word_array[index] = word_ids
      tag_array[index] = tag_ids

  return word_array, tag_array


def load_data(data_path, num_step):
    """Load POS raw data from data directory "data_path".
    Args: data_path
    Returns:
      tuple (train_data, valid_data, test_data, vocab_size)
      where each of the data objects can be passed to iterator.
    """

    train_path = os.path.join(data_path, "train.txt")
    dev_path = os.path.join(data_path, "dev.txt")
    test_path = os.path.join(data_path, "test.txt")

    word_to_id, tag_to_id = _build_vocab(train_path)
    # Save word_dict and tag_dict
    _save_vocab(word_to_id, os.path.join(data_path, "word_to_id"))
    _save_vocab(tag_to_id, os.path.join(data_path, "tag_to_id"))
    print("word dictionary size " + str(len(word_to_id)))
    print("tag dictionary size " + str(len(tag_to_id)))

    train_word, train_tag = _file_to_word_ids(train_path, word_to_id, tag_to_id, num_step)
    print("train dataset: " + str(len(train_word)) + " " + str(len(train_tag)))
    dev_word, dev_tag = _file_to_word_ids(dev_path, word_to_id, tag_to_id, num_step)
    print("dev dataset: " + str(len(dev_word)) + " " + str(len(dev_tag)))
    test_word, test_tag = _file_to_word_ids(test_path, word_to_id, tag_to_id, num_step)
    print("test dataset: " + str(len(test_word)) + " " + str(len(test_tag)))
    vocab_size = len(word_to_id)

    return (train_word, train_tag, dev_word, dev_tag, test_word, test_tag, vocab_size)


