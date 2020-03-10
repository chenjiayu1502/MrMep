import os
import json

import tensorflow as tf
import numpy as np


from model import Encoder, QASystem, Decoder,CNN
from config import NYT_Config, WebNLG_Config



from data_utils import *
from os.path import join as pjoin



def run_func():
    conf=json.load(open('config.json'))
    print(conf)
    if conf['data_name']=="NYT":
        config=NYT_Config(conf)
    else:
        config=WebNLG_Config(conf)
    train = dataset(config.question_train, config.context_train, config.answer_train,config.cnn_output_train,config.cnn_list_train)
    dev = dataset(config.question_dev, config.context_dev, config.answer_dev, config.cnn_output_dev,config.cnn_list_dev)
    test = dataset(config.question_test, config.context_test, config.answer_test, config.cnn_output_test,config.cnn_list_test)

    print(len(train))
    print(len(dev))
    print(len(test))
   


    encoder = Encoder(config.hidden_state_size)
    decoder = Decoder(config.hidden_state_size)
    cnn = CNN(config,is_training=True)
    qa = QASystem(encoder, decoder, cnn, config)
    
    sess = tf.Session()
    qa.initialize_model(sess, config.train_dir)
    qa.train(sess, [train, dev, test], config.train_dir,config)
    # qa.train(sess, [train, dev], config.train_dir,config)




if __name__ == "__main__":
    run_func()
