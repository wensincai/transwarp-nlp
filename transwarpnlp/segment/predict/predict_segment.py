#coding=utf-8
from __future__ import unicode_literals

from transwarpnlp.segment import segmenter

text = "我刚刚在浙江卫视看了电视剧老九门，觉得陈伟霆很帅"
segList = segmenter.seg(text)
text_seg = " ".join(segList)

print (text)
print (text_seg)
