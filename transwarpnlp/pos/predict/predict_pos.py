#coding:utf-8

from __future__ import unicode_literals
import codecs

from transwarpnlp.segment import segmenter
from transwarpnlp.pos.predict import pos_tagger

def predict(data_dir, train_dir, method, predict_file, output_file):
    tagger = pos_tagger.load_model(data_dir, train_dir, method)
    with codecs.open(predict_file, 'r', 'utf-8') as predict,\
            codecs.open(output_file, 'w', 'utf-8') as output:
        lines = predict.readlines()
        for line in lines:
            words = segmenter.seg(line)
            tagging = tagger.predict(words)
            for (w, t) in tagging:
                str = w + "/" + t
                output.write(str + ' ')
            output.write('\n')
