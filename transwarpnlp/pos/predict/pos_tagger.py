#!/usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode coding

import os
import tensorflow as tf
import numpy as np
import glob

from transwarpnlp.pos import pos_model as pos_model
from transwarpnlp.pos import pos_model_bilstm
from transwarpnlp.pos.config import LargeConfig
from transwarpnlp.pos.dataset import rawdata, dataset

class ModelLoader(object):
    
    def __init__(self, data_path, ckpt_path, method):
        self.method = method
        self.data_path = data_path
        self.ckpt_path = ckpt_path  # the path of the ckpt file, e.g. ./ckpt/zh/pos.ckpt
        print("Starting new Tensorflow session...")
        self.session = tf.Session()
        print("Initializing pos_tagger class...")
        self.model = self._init_pos_model(self.session, self.ckpt_path)
    
    def predict(self, words):
        '''
        Coding: utf-8 for Chinese Characters
        Return tuples of [(word, tag),...]
        '''
        tagging = self._predict_pos_tags(self.session, self.model, words, self.data_path)
        return tagging
    
    ## Initialize and Instance, Define Config Parameters for POS Tagger
    def _init_pos_model(self, session, ckpt_path):
        """Create POS Tagger model and initialize with random or load parameters in session."""
        # initilize config
        # config = POSConfig()   # Choose the config of language option
        config = LargeConfig()
        config.batch_size = 1
        config.num_steps = 1 # iterator one token per time
      
        with tf.variable_scope("pos_var_scope"):  #Need to Change in Pos_Tagger Save Function
            if self.method == "lstm":
                model = pos_model.POSTagger(is_training=False, config=config)
            else:
                model = pos_model_bilstm.POSTagger(is_training=False, config=config)
        
        if len(glob.glob(ckpt_path + '.data*')) > 0: # file exist with pattern: 'pos.ckpt.data*'
            print("Loading model parameters from %s" % ckpt_path)
            all_vars = tf.global_variables()
            model_vars = [k for k in all_vars if k.name.startswith("pos_var_scope")]
            tf.train.Saver(model_vars).restore(session, ckpt_path)
        else:
            print("Model not found, created with fresh parameters.")
            session.run(tf.global_variables_initializer())
        return model
    
    def _predict_pos_tags(self, session, model, words, data_path):
        '''
        Define prediction function of POS Tagging
        return tuples [(word, tag)]
        '''
        word_data = rawdata.sentence_to_word_ids(data_path, words)

        word_arr = np.zeros((len(word_data), model.num_steps), np.int32)
        tag_arr = np.zeros((len(word_data), model.num_steps), np.int32)

        for i, word in enumerate(word_data):
            word_arr[i] = word

        dat = dataset.Dataset(word_arr, tag_arr)
        
        predict_id =[]
        step = 0
        while dat.hasNext():
            step = step + 1
            x,y = dat.nextBatch(model.batch_size)
            fetches = [model.cost, model.logits]
            feed_dict = {}
            feed_dict[model.input_data] = x
            feed_dict[model.targets] = y

            _, logits  = session.run(fetches, feed_dict)
            predict_id.append(int(np.argmax(logits)))    
        predict_tag = rawdata.word_ids_to_sentence(data_path, predict_id)
        return zip(words, predict_tag)
    
def load_model(data_dir, train_dir, method="lstm"):
    if method == "lstm":
        ckpt_path = os.path.join(train_dir, "lstm/lstm.ckpt") # POS model checkpoint path
    else:
        ckpt_path = os.path.join(train_dir, "bilstm/bilstm.ckpt") # POS model checkpoint path

    return ModelLoader(data_dir, ckpt_path, method)

