# Transwarp NLP

深度自然语言处理工具。代码正在持续实现和改进中，部分实现的功能现在都能够跑通，但是没有使用大规模语料来训练。后续会基于大规模语料来进行训练，
以求达到生产环境的要求。

## 功能实现

### 1 中文Word2Vec训练

- skip gram

```
cd transwarpnlp/word2vec

训练：./train --data_file=path/to/data_file \
        --output_file=path/to/data_dir \
        --vocabulary_size=50000 \
        --batch_size=128 \
        --embedding_size=128 \
        --skip_window=1 \
        --num_skips=2 \
        --valid_size=16 \
        --valid_window=100 \
        --num_sampled=64 \
        --num_steps=100001
```

`data_file`表示训练数据集，`output_file`保存结果词向量。`vocabulary_size`表示训练的词个数。

### 2 中文分词

- CRF实现

利用CRF++训练模型，使用训练好的模型分词

- BILSTM + CRF

参考文献：[Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)

### 3 序列化标注

- LSTM

```
cd transwarpnlp/pos

训练：./run --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --process=train \
        --method=lstm
        
预测：./run --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --predict_file=path/to/predict_file \
        --output_file=path/to/output_file \
        --process=predict \
        --method=lstm
```

`train_dir`表示训练模型存放的路径，`data_dir`表示训练、测试数据存放的路径。
 `predict_file`表示待标注的文件。 `output_file`表示标注后的结果文件。

- BILSTM

```
cd transwarpnlp/pos

训练：./run --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --process=train \
        --method=bilstm
        
预测：./run --train_dir=path/to/train_dir \ 
        --data_dir=path/to/data_dir \
        --predict_file=path/to/predict_file \
        --output_file=path/to/output_file \
        --process=predict \
        --method=bilstm
```

### 4 中文命名实体识别

- LSTM

```
cd transwarpnlp/ner

训练：./run --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --process=train \
        --method=lstm
        
预测：./run --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --predict_file=path/to/predict_file \
        --output_file=path/to/output_file \
        --process=predict \
        --method=lstm
```

`train_dir`表示训练模型存放的路径，`data_dir`表示训练、测试数据存放的路径。
 `predict_file`表示待标注的文件。 `output_file`表示标注后的结果文件。
 
- BILSTM

```
cd transwarpnlp/ner

训练：./run --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --process=train \
        --method=bilstm
        
预测：./run --train_dir=path/to/train_dir \ 
        --data_dir=path/to/data_dir \
        --predict_file=path/to/predict_file \
        --output_file=path/to/output_file \
        --process=predict \
        --method=bilstm
```

### 5 中文关键词抽取

NOT FINISHED

### 6 中文文本自动摘要

- SEQ2SEQ ATTENTION

参考文献：[A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685)

### 7 英文情感分析

- MEMORY NETWORK

参考文献：[Aspect Level Sentiment Classification with Deep Memory Network](https://arxiv.org/abs/1605.08900),
[Memory Network](https://arxiv.org/pdf/1410.3916.pdf)

本任务需要下载`glove.6B.300d.txt`，下载地址为：[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)。

本任务现在仅仅使用英文语料，后续会基于中文语料进行情感分析。

### 8 依存句法分析

NOT FINISHED

### 9 中文文本分类

- CNN

参考文献：[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

- LSTM

### 10 中文多标签文本分类

- CNN

参考文献：[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

- LSTM


### 11 中文自由写诗

NOT FINISHED

### 12 中文对话系统

NOT FINISHED

### 13 中文问答系统

NOT FINISHED

### 14 中英机器翻译

NOT FINISHED

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
