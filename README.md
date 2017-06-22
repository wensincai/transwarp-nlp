# Transwarp NLP

深度自然语言处理工具。代码正在持续实现和改进中，部分实现的功能现在都能够跑通，但是没有使用大规模语料来训练。后续会基于大规模语料来进行训练，
以求达到生产环境的要求。

## 功能实现

### 1 中文分词

- CRF实现

利用CRF++训练模型，使用训练好的模型分词

- BILSTM + CRF

参考文献：[Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)

### 2 序列化标注

- LSTM
- BILSTM

### 3 中文命名实体识别

- LSTM
- BILSTM

### 4 中文关键词抽取

NOT FINISHED

### 5 中文文本自动摘要

- SEQ2SEQ ATTENTION

参考文献：[A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685)

### 6 英文情感分析

- MEMORY NETWORK

参考文献：[Aspect Level Sentiment Classification with Deep Memory Network](https://arxiv.org/abs/1605.08900),
[Memory Network](https://arxiv.org/pdf/1410.3916.pdf)

本任务需要下载`glove.6B.300d.txt`，下载地址为：[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)。

本任务现在仅仅使用英文语料，后续会基于中文语料进行情感分析。

### 7 依存句法分析

NOT FINISHED

### 8 中文文本分类

- CNN

参考文献：[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- LSTM

### 9 中文多标签文本分类
- CNN

参考文献：[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- LSTM

### 10 依存句法分析

NOT FINISHED

### 11 中文自由写诗

NOT FINISHED

### 12 中文对话系统

NOT FINISHED

### 13 中文问答系统

NOT FINISHED

### 14 中英机器翻译

NOT FINISHED

## 依赖库

* python2.7
* tensorflow (>= r1.0)
* numpy
* pandas
* matplotlib
* sklearn
* future
* cPickle

## 参考项目

* https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb
* https://github.com/koth/kcws
* https://github.com/google/seq2seq
* https://github.com/tensorflow/models/tree/master/textsum
* https://github.com/qhduan/Seq2Seq_Chatbot_QA
* https://github.com/jinfagang/tensorflow_poems
* https://github.com/ganeshjawahar/mem_absa
* https://github.com/yanshao9798/tagger
* https://github.com/rockingdingo/deepnlp
* https://github.com/luchi007/RNN_Text_Classify
* https://github.com/LambdaWx/CNN_sentence_tensorflow

## LICENSE

All code in this repository is under the MIT license as specified by the [LICENSE](LICENSE) file.
