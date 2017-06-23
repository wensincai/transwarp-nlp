#coding:utf-8

from __future__ import unicode_literals # compatible with python3 unicode

from transwarpnlp import segmenter
import ner_tagger
import os

pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

tagger = ner_tagger.load_model(pkg_path, "bilstm")

#Segmentation
text = "我爱吃北京烤鸭"
words = segmenter.seg(text)
print (" ".join(words).encode('utf-8'))

#NER tagging
tagging = tagger.predict(words)
for (w,t) in tagging:
    str = w + "/" + t
    print (str.encode('utf-8'))

