#coding:utf-8


from __future__ import unicode_literals
import os

from transwarpnlp import segmenter
import pos_tagger

pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

tagger = pos_tagger.load_model(pkg_path, 'bilstm')

#tagger = pos_tagger.load_model(pkg_path, 'bilstm')


#Segmentation
text = "我爱吃北京烤鸭"
words = segmenter.seg(text)
print(" ".join(words).encode('utf-8'))

#POS Tagging
tagging = tagger.predict(words)
for (w,t) in tagging:
    str = w + "/" + t
    print(str.encode('utf-8'))
