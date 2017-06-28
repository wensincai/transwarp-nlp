# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import matplotlib.pyplot as plt

def read_data(filename):
  data = []
  with open(filename, 'r') as f:
    for line in f.readlines():
        words = line.split()
        for word in words:
            data.append(word.split("/")[0])
  return data

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
      dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
      if word in dictionary:
          index = dictionary[word]
      else:
          index = 0  # dictionary['UNK']
          unk_count += 1
      data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label.decode('utf-8'),
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
  plt.savefig(filename)

def save_embeding(final_embeding, reverse_dictionary, output_file):
    with open(output_file, 'w') as output:
        for i, vec in enumerate(final_embeding):
            word = reverse_dictionary[i]
            output.write(word + " ")
            output.write(','.join(str(value) for value in vec))
            output.write("\n")

