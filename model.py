# -*- coding: utf-8 -*-  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
import sys
from six.moves import xrange  
import tensorflow as tf
from general_utils import Progbar
from data_utils_for_opt import *
from collections import defaultdict as ddict


from tensorflow.python import debug as tf_debug
from tensorflow.contrib.rnn import BasicLSTMCell  

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


from transformer_modules import multihead_attention

from tqdm import tqdm


import os
import json


def get_embedding_table(config):
    with tf.variable_scope("embedding_init") as scope:
        if os.path.isfile(config.words_id2vector_filename):
            print('Word Embedding init from %s' % config.words_id2vector_filename)
            words_id2vec = json.load(open(config.words_id2vector_filename, 'r'))
            words_vectors = [0] * len(words_id2vec)
            for id, vec in words_id2vec.items():
                words_vectors[int(id)] = vec
            words_embedding_table = tf.Variable(name='words_emb_table', initial_value=words_vectors, dtype=tf.float32)
        else:
            print('Word Embedding random init')
            words_embedding_table = tf.get_variable(name='words_emb_table',
                                                    shape=[config.words_number + 1, config.embedding_dim],
                                                    dtype=tf.float32)

        embedding_table=words_embedding_table
    return embedding_table
def concat_embedding(self, word_embedding, pos_embedding):
    if pos_embedding is None:
        return word_embedding
    else:
        return tf.concat(values = [word_embedding, pos_embedding], axis = 2)



class CNN(object):
    def __init__(self, config, is_training, drop_prob = None):
        self.config = config
        self.is_training = is_training
        self.dropout = self.is_training

    def __cnn_cell__(self, x, kernel_size, stride_size):
        x = tf.expand_dims(x, axis=1)
        x = tf.layers.conv2d(inputs=x, 
            filters = self.config.hidden_state_size, #*self.config.split_num
            kernel_size = [1, kernel_size], 
            strides = [1, stride_size], 
            padding = 'same', 
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        return x
    def __pooling__(self, x, max_length):
        temp = tf.unstack(x,axis=1)
        x=temp[0]
        x1 = tf.reduce_max(x, axis = 1)
        x1=tf.reshape(x1, [-1, self.config.hidden_state_size])#*self.config.split_num
        x1=tf.reshape(x1, [-1, self.config.hidden_state_size])
        x2 = tf.reduce_mean(x, axis = 1)
        x2=tf.reshape(x2, [-1, self.config.hidden_state_size])#*self.config.split_num
        x2=tf.reshape(x2, [-1, self.config.hidden_state_size])
        return x1,x2#*self.config.split_num

           


    def __dropout__(self, x):
        if self.dropout:
            return tf.nn.dropout(x,self.config.dropout_val)
        else:
            return x


    def cnn_op(self, x, dropout,sen_pre ,kernel_size = 3, stride_size = 1, activation=tf.nn.relu):
        with tf.name_scope("cnn"):
            max_length = x.get_shape()[1]
            
            x1 = self.__cnn_cell__(x, kernel_size, stride_size)
            
            x1,x2 = self.__pooling__(x1,max_length)

            x1 = tf.concat([x1,sen_pre],axis=1)

            # print('x2--------',x1)
            
            x1 = activation(x1)
            x1 = tf.nn.dropout(x1,dropout)
            
            
            

        
        return x1
    def __logits__(self, x):
        with tf.variable_scope('logits'):
            logits_list=[]
            cnn_repre =[]
            for i in range(self.config.relation_num):
                w1 = tf.get_variable('classifier_matrix_w1__%s'%str(i), [self.config.hidden_state_size, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                bias1 = tf.get_variable('bias1_%s'%str(i), [self.config.hidden_state_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                w2 = tf.get_variable('classifier_matrix_w2__%s'%str(i), [self.config.num_classes, self.config.hidden_state_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                bias2 = tf.get_variable('bias2_%s'%str(i), [self.config.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                layer1 = tf.matmul(x, tf.transpose(w1)) + bias1
                logits = tf.matmul(layer1, tf.transpose(w2)) + bias2
                logits_list.append(logits)
                cnn_repre.append(tf.reshape(layer1,[-1,1,self.config.hidden_state_size]))
            logits_list = tf.stack(logits_list,axis=1)
        return logits_list,cnn_repre
    def softmax_cross_entropy(self, logits, outputs):
        with tf.name_scope("loss"):
            label_onehot = tf.one_hot(indices=outputs, depth=self.config.num_classes, on_value=1, dtype=tf.int32)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_onehot, logits=logits)
            loss = tf.reduce_mean(loss)
            return loss
    def predict(self, x):
        with tf.name_scope("predict"):
            pred = tf.argmax(x, 2, name="predict")
            return pred
    def _acc(self,logits, outputs):
        correct_prediction = tf.equal(tf.cast(tf.argmax(logits,2),tf.int32), outputs)    
        acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return acc
    def classify(self,words,outputs,dropout,sen_pre):
        x = self.cnn_op(words,dropout,sen_pre)
        cnn_output = tf.concat(x, 1)
        logits,cnn_repre = self.__logits__(cnn_output)
        pred = self.predict(logits)
        acc = self._acc(logits,outputs)
        loss = self.softmax_cross_entropy(logits, outputs)
        return logits,pred,acc,loss,cnn_repre,x



class Encoder(object):
    def __init__(self, hidden_size, initializer = lambda : None):#tf.contrib.layers.xavier_initializer):
        self.hidden_size = hidden_size
        self.init_weights = initializer

    def encode_pssg(self, passage, masks_passage, encoder_state_input = None):
        
        with tf.variable_scope("encoded_passage"):
            lstm_cell_passage  = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
            encoded_passage, (p_rep, p_state) =  tf.nn.dynamic_rnn(lstm_cell_passage, passage, masks_passage, dtype=tf.float32) # (-1, P, H)



        return  encoded_passage , p_rep, p_state
    def encode_query(self, question, masks_question, encoder_state_input = None):
        
        with tf.variable_scope("encoded_question"):
            lstm_cell_question = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
            encoded_question, (q_rep, _) = tf.nn.dynamic_rnn(lstm_cell_question, question, masks_question, dtype=tf.float32) # (-1, Q, H)

        
        return encoded_question , q_rep

        
   
class Decoder(object):
    def __init__(self, hidden_size, initializer= lambda : None):
        self.hidden_size = hidden_size
        self.init_weights = initializer

   
    def run_my_match(self,encoded_rep, masks, config):
        encoded_question, encoded_passage = encoded_rep
        masks_question, masks_passage = masks
        q=tf.reshape(encoded_question, [-1, config.hidden_state_size])

        q_list=self.run_concat(q,config)
        passage=tf.concat([encoded_passage,q_list],axis=-1)
        with tf.variable_scope("bi-lstm"):
            cell_fw = BasicLSTMCell(config.hidden_state_size)
            cell_bw = BasicLSTMCell(config.hidden_state_size)
            (output_fw_seq, output_bw_seq), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=passage,
                sequence_length=masks_passage,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            _,fw_h=output_state_fw
            _,bw_h=output_state_bw
            state = tf.concat([fw_h, bw_h], axis=-1)
            print(output_state_fw, output_state_bw)
            print('state----',state)
            print('output---',output)
            # output = tf.nn.dropout(output, self.dropout_pl)
        return output, state

    
    def run_concat(self,emb, config):
        q = [emb]
        for i in range(config.max_length-1):
            # q=tf.concat([q,emb],axis=1)
            q.append(emb)
        q = tf.stack(q, axis=1)
        
        return q
    def separated_attention(self,encoded_passage, h, rel,config):
        print("separated_attention")
        input_size = h.get_shape().as_list()[1]
        matrix_p = tf.get_variable("Matrix_p", [input_size], dtype=h.dtype)
        matrix_r = tf.get_variable("Matrix_r", [input_size], dtype=h.dtype)
        matrix_h = tf.get_variable("Matrix_h", [input_size], dtype=h.dtype)
        v = tf.get_variable("Matrix_v", [input_size,1], dtype=h.dtype)
        hh=h 
        rr=rel 
        h_list=self.run_concat(hh,config)
        r_list=self.run_concat(rr,config)
        temp = math_ops.tanh(encoded_passage*matrix_p + h_list*matrix_h + r_list*matrix_r)
        temp = tf.reshape(temp, [-1, input_size])
        temp = tf.matmul(temp, v)
        temp = tf.reshape(temp,[-1, config.max_length])
        logits=tf.nn.softmax(temp)
        weight=tf.reshape(logits, [-1,config.max_length,1])
        ct = tf.reduce_sum(tf.multiply(encoded_passage, weight),axis=1)


        return logits,ct
    def combined_attention(self,encoded_passage, h, rel,config):
        with tf.variable_scope("attention_step") as scope:
            input_size = h.get_shape().as_list()[1]
            hh=h 
            rr=rel 
            matrix_h = tf.get_variable("Matrix_h", [input_size], dtype=h.dtype)
            matrix_r = tf.get_variable("Matrix_r", [input_size], dtype=h.dtype)
            hr = math_ops.tanh(hh*matrix_h + rr*matrix_r)
            print('hr==',hr)

            matrix_1 = tf.get_variable("Matrix_1", [input_size,input_size], dtype=h.dtype)
            
            enp=tf.reshape(encoded_passage, [-1, input_size])
            temp=tf.matmul(enp, matrix_1)
            temp=tf.reshape(temp, [-1,config.max_length, input_size])
            hr=tf.reshape(hr,[-1,input_size,1])
            temp=tf.matmul(temp,hr)
            temp = tf.reshape(temp, [-1, config.max_length])
            logits = tf.nn.softmax(temp)
            weight=tf.reshape(logits, [-1,config.max_length,1])
            ct = tf.reduce_sum(tf.multiply(encoded_passage, weight),axis=1)
        return logits, ct
    def attention_no_rel(self,encoded_passage, h,config):
        input_size = h.get_shape().as_list()[1]
        matrix_p = tf.get_variable("Matrix_p", [input_size], dtype=h.dtype)
        # matrix_r = tf.get_variable("Matrix_r", [input_size], dtype=h.dtype)
        matrix_h = tf.get_variable("Matrix_h", [input_size], dtype=h.dtype)
        v = tf.get_variable("Matrix_v", [input_size,1], dtype=h.dtype)
        hh=h 
        # rr=rel 
        h_list=self.run_concat(hh,config)
        temp = math_ops.tanh(encoded_passage*matrix_p + h_list*matrix_h)
        
        temp = tf.reshape(temp, [-1, input_size])
        temp = tf.matmul(temp, v)
        temp = tf.reshape(temp,[-1, config.max_length])
        logits=tf.nn.softmax(temp)
        weight=tf.reshape(logits, [-1,config.max_length,1])
        ct = tf.reduce_sum(tf.multiply(encoded_passage, weight),axis=1)


        return logits,ct

    def run_match_answer_ptr(self, encoded_rep, state, masks, labels, config):
        with tf.variable_scope("decode_step") as scope:
            # h,p_states = p_encode
            h=state
            encoded_passage = encoded_rep
            # print('encoded_question--',encoded_question)
            print('encoded_passage--',encoded_passage)
            print('h--',h)
            init_state = tf.concat([h,h],axis=1)
            print('state---',init_state)
            cell = tf.contrib.rnn.BasicLSTMCell(config.hidden_state_size*2, state_is_tuple = False)
            # state = cell.zero_state(encoded_passage.get_shape().as_list()[0], dtype=tf.float32)

            logits=[]
            state = init_state
            for step in range(config.max_decode_size):
                logit,inputs = self.attention_no_rel(encoded_passage, h,config)
                h,state = cell(inputs, state)
                logits.append(logit)
                tf.get_variable_scope().reuse_variables()
        return tf.stack(logits,axis=1)
    def run_multi_head_answer_ptr(self, encoded_rep, p_encode, masks, labels, config):
        with tf.variable_scope("decode_step") as scope:
            encoded_question, encoded_passage = encoded_rep
            print('encoded_question--',encoded_question)
            print('encoded_passage--',encoded_passage)
            h=tf.reshape(p_encode, [-1,config.hidden_state_size])
            rel=tf.reshape(encoded_question, [-1,config.hidden_state_size])
            try:
                state = array_ops.concat(1, [h,h])
            except:
                state = array_ops.concat([h,h], 1)
            
            cell = tf.contrib.rnn.BasicLSTMCell(config.hidden_state_size, state_is_tuple = False)

            logits=[]
            if config.model_name=="sep":
                att=self.separated_attention
            else:
                att=self.combined_attention
            for step in range(config.max_decode_size):
                logit,inputs = att(encoded_passage, h, rel,config)
                h,state = cell(inputs, state)
                logits.append(logit)
                tf.get_variable_scope().reuse_variables()
        return tf.stack(logits,axis=1)


    def decode_match(self, encoded_rep, p_states, masks, labels, config):
        output_attender , state =  self.run_my_match(encoded_rep, masks, config)
        logits=self.run_match_answer_ptr(output_attender, state, masks, labels, config)
        
    
        return logits
    def decode_multi_head(self, encoded_rep, p_encode, masks, labels, config, dropout):
        encoded_question, encoded_passage = encoded_rep
        # enp=encoded_passage
        
        enp=multihead_attention(queries=encoded_passage, 
                            keys=encoded_passage, 
                            values=encoded_passage,
                            num_heads=config.head_num
                            )
        # print('enp:',enp)
        enp=tf.nn.dropout(enp,dropout)
  
        logits=self.run_multi_head_answer_ptr([encoded_question,enp], p_encode, masks, labels, config)
        return logits
        
    
class QASystem(object):    
    def __init__(self, encoder, decoder, cnn, config):

        # ==== set up logging ======

        logger = logging.getLogger("QASystemLogger")
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(config.log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        logger.addHandler(handler)
        logger.addHandler(console)
        self.logger = logger
        
        self.encoder = encoder
        self.decoder = decoder
        self.cnn = cnn
        self.config = config
        self.setup_placeholders()

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            self.setup_word_embeddings()
            self.setup_system()
            self.setup_loss()
            self.setup_train_op()
            self.saver = tf.train.Saver(max_to_keep=100)

        



    def setup_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step") as scope:
            

            learning_rate = self.config.learning_rate
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.init = tf.global_variables_initializer()


    def get_feed_dict(self, questions, contexts, answers, cnn_output, dropout_val):
        """
        -arg questions: A list of list of ids representing the question sentence
        -arg contexts: A list of list of ids representing the context paragraph
        -arg dropout_val: A float representing the keep probability for dropout 

        :return: dict {placeholders: value}
        """

        padded_questions, question_lengths = pad_sequences_for_query(questions, 0, self.config)
        padded_contexts, passage_lengths = pad_sequences_for_passage(contexts, 0, self.config)
        padded_answers, answer_lengths = pad_sequences_for_answer(answers, 0, self.config)
        weighted_loss = [self.config.label_weight]*self.config.max_length
        weighted_loss[0]=1

        feed = {
            self.question_ids : padded_questions,
            self.passage_ids : padded_contexts,
            self.question_lengths : question_lengths,
            self.passage_lengths : passage_lengths,
            self.labels : padded_answers,
            self.cnn_output : cnn_output,
            self.dropout : dropout_val,
            self.weighted_loss : weighted_loss
        }

        return feed


    def setup_word_embeddings(self):
        
        with tf.variable_scope("vocab_embeddings"):
            self.embeddings_relation = tf.get_variable('embedding_relation', [self.config.relation_num, self.config.hidden_state_size],initializer=tf.random_normal_initializer(mean=0, stddev=1))
            _word_embeddings = get_embedding_table(self.config)
            question_emb = tf.nn.embedding_lookup(self.embeddings_relation, self.question_ids, name = "question") # (-1, Q, D)
            passage_emb = tf.nn.embedding_lookup(_word_embeddings, self.passage_ids, name = "passage") # (-1, P, D)
            # Apply dropout
            self.question = tf.nn.dropout(question_emb, self.dropout)
            self.passage  = tf.nn.dropout(passage_emb, self.dropout)
            



    def setup_placeholders(self):
        self.question_ids = tf.placeholder(tf.int32, shape = [None,1], name = "question_ids")
        self.passage_ids = tf.placeholder(tf.int32, shape = [None, None], name = "passage_ids")

        self.cnn_output=tf.placeholder(dtype=tf.int32, shape=[None,self.config.relation_num], name='cnn_output')
        self.weighted_loss = tf.placeholder(dtype=tf.float32, shape=[self.config.max_length], name='weighted_loss')

        self.question_lengths = tf.placeholder(tf.int32, shape=[None], name="question_lengths")
        self.passage_lengths = tf.placeholder(tf.int32, shape = [None], name = "passage_lengths")

        self.labels = tf.placeholder(tf.int64, shape = [None, self.config.max_decode_size], name = "gold_labels")
        self.dropout = tf.placeholder(tf.float32, shape=[], name = "dropout")

    def setup_system(self):
        """
           Apply the encoder to the question and passage embeddings. Follow that up by Match-LSTM and Answer-Ptr 
        """
        encoder = self.encoder
        decoder = self.decoder
        cnn = self.cnn

        
        encoded_passage , p_encode, p_state= encoder.encode_pssg(self.passage,self.passage_lengths , encoder_state_input = None)
        
        cnn_logits, cnn_pred, cnn_acc, cnn_loss, cnn_repre, sen_repre = cnn.classify(encoded_passage,self.cnn_output,self.dropout,p_encode)

        
        q_ids = tf.reshape(self.question_ids,[-1])
        q_repre = tf.nn.embedding_lookup(cnn_repre,q_ids)

        # encoded_question, q_rep = encoder.encode_query(q_repre, self.question_lengths,encoder_state_input = None)
        
        if self.config.model_name=="baseline":
            encoded_question, q_rep = encoder.encode_query(q_repre, self.question_lengths,encoder_state_input = None)

            logits= decoder.decode_match([encoded_question, encoded_passage], [p_encode, p_state], [self.question_lengths, self.passage_lengths], self.labels, self.config)
        else:
            logits=decoder.decode_multi_head([q_repre, encoded_passage], p_encode, [self.question_lengths, self.passage_lengths], self.labels, self.config, self.dropout)


        self.output_logits = tf.unstack(logits,axis=1)

        self.logits = logits
        self.cnn_loss = cnn_loss
        self.cnn_pred = cnn_pred
        


    def setup_loss(self):
        
        labels = self.labels
        logits = self.logits
        
        print('logits::',logits)
        print('labels::',labels)
        label_onehot = tf.one_hot(indices=labels, depth=self.config.max_length, on_value=1.0, dtype=tf.float32)
        print('labels_onehot::',label_onehot)
        weighted_label_onehot = tf.multiply(label_onehot,self.weighted_loss)
        print(weighted_label_onehot*tf.log(logits))
        losses =  -tf.reduce_mean(weighted_label_onehot*tf.log(logits)) 

        match_loss = losses
        cnn_pred = tf.cast(self.cnn_pred,dtype=tf.float32)
        self.loss = 3.0*tf.reduce_mean(match_loss)+self.cnn_loss

    def initialize_model(self, session, train_dir):
        session.run(self.init)
       
    def answer_sequence(self, outputs):
        def func(y1, y2):
            max_ans = -999999
            a_s, a_e= 0,0
            num_classes = len(y1)
            for i in xrange(num_classes):
                for j in xrange(15):
                    if i+j >= num_classes:
                        break

                    curr_a_s = y1[i];
                    curr_a_e = y2[i+j]
                    if (curr_a_e+curr_a_s) > max_ans:
                        max_ans = curr_a_e + curr_a_s
                        a_s = i
                        a_e = i+j

            return (a_s, a_e)
        res=[]
        for i in range(len(outputs)//4):
            yp, yp2, ypro,ypro2 = outputs[4*i],outputs[4*i+1],outputs[4*i+2],outputs[4*i+3]
            yp=yp[0].tolist()
            yp2=yp2[0].tolist()
            ypro=ypro[0].tolist()
            ypro2=ypro2[0].tolist()
            _a_s, _a_e = func(yp, yp2)
            _a_s2, _a_e2 = func(ypro, ypro2)
            res.append([_a_s, _a_e, _a_s2, _a_e2])
        return res
    def evaluate_cnn(self, session, dataset, config, mode):
        q, c, a, co = zip(*[[_q, _c, _a, _co] for (_q, _c, _a, _co) in dataset])
        input_feed =  self.get_feed_dict(q, c, a, co, 1.0)

        output_feed = [self.cnn_pred,self.logits]


        cnn_pred,outputs = session.run(output_feed, input_feed)
        cnn_pred=np.array(cnn_pred)


        gold_cnn = np.array([co for (_,_,_,co) in dataset])
        if mode=='test':
            lines = open(config.test_match_file,'r').readlines()
        else:
            lines = open(config.dev_match_file,'r').readlines()




        sample = len(dataset)
        pbar = tqdm(total=sample)

        pred_cnn_cnt = np.sum(cnn_pred)
        true_cnn_cnt = np.sum(gold_cnn)
        pred_match_cnt = 0
        true_match_cnt = 0
        pred=0.0
        em_score=0.0

        f=open(config.result_file,'w')

        for i in range(sample):
            if i>100:
                break
            res={'id':i,'content':[]}
            gold_match = json.loads(lines[i].strip())['data']
            
            for k in range(1,len(cnn_pred[i])):
                temp={'rel':k}
                true_match_cnt+=len(gold_match[k])

                if cnn_pred[i][k]==1:
                    
                    input_feed = self.get_feed_dict([[k]],[c[i]],[a[i]],[gold_cnn[i]],1.0)
                    output_feed = [self.cnn_pred,self.output_logits]
                    _,outputs = session.run(output_feed, input_feed)
                    pred_res =self.answer_sequence(outputs)
                    new_pred_res=[]
                    for ind in range(len(pred_res)):
                        if 0 in pred_res[ind]:
                            break
                        if pred_res[ind] not in new_pred_res:
                            new_pred_res.append(pred_res[ind])
                    pred_match_cnt+=len(new_pred_res)

                    if gold_cnn[i][k]==1:
                        pred+=1.0
                        for ind in range(len(new_pred_res)):
                            if pred_res[ind] in gold_match[k]:
                                em_score+=1.0
                    temp['label']=[int(cnn_pred[i][k]),int(gold_cnn[i][k])]
                    temp['pred']=pred_res
                    temp['true']=gold_match[k]
                else:
                    temp['label']=[int(cnn_pred[i][k]),int(gold_cnn[i][k])]
                    temp['pred']=[]
                    temp['true']=gold_match[k]
                res['content'].append(temp)
            f.write(json.dumps(res)+'\n')
            pbar.set_description("pred: %s em_score:%s" % (str(pred),str(em_score)))
            pbar.update(1)
        f.close()
        pbar.close()

        print('%f %f %f %f %f %f\n'%(em_score, pred,pred_cnn_cnt, true_cnn_cnt, pred_match_cnt, true_match_cnt))
        if pred==0:
            p=0.0
            r=0.0
            f1=0.0
        else:
            p=pred/pred_cnn_cnt
            r=pred/true_cnn_cnt
            f1=2*p*r/(p+r)
        print('%f %f %f \n'%(p,r,f1))

        if em_score == 0:
            p_all=0.0
            r_all=0.0
            f1_all=0.0
        else:
            p_all = em_score/pred_match_cnt
            r_all = em_score/true_match_cnt
            f1_all = 2*p_all*r_all/(p_all+r_all)
        print('%f\t%f\t%f\t\n'%(p_all,r_all,f1_all))
        return f1_all



   

    def run_epoch(self, session, train):
        nbatches = int((len(train) + self.config.batch_size - 1) / self.config.batch_size)
        pbar = tqdm(total=nbatches)
        train_loss_list=[]
        cnn_loss_list=[]


        for i, (q_batch, c_batch, a_batch, co_batch) in enumerate(minibatches(train, self.config.batch_size)):

            # at training time, dropout needs to be on.
            if i>3:
                break
            input_feed = self.get_feed_dict(q_batch, c_batch, a_batch, co_batch, self.config.dropout_val)

            _, train_loss,cnn_loss= session.run([self.train_op, self.loss, self.cnn_loss], feed_dict=input_feed)
            train_loss_list.append(train_loss)
            cnn_loss_list.append(cnn_loss)
            pbar.set_description("train loss: %s cnn_loss:%s" % (str(sum(train_loss_list)/len(train_loss_list)),str(sum(cnn_loss_list)/len(cnn_loss_list))))
            pbar.update(1)

        pbar.close()
        return sum(train_loss_list)/len(train_loss_list)





    def train(self, session, dataset, train_dir,config):
        
        if not tf.gfile.Exists(train_dir):
            tf.gfile.MkDir(train_dir)


        train, dev, test = dataset
        self.saver.restore(session,"%s/best_model.pkl" %train_dir)
        f1 = self.evaluate_cnn(session, dev, config, 'dev')
        f1 = self.evaluate_cnn(session, test, config, 'test')

        print("#-----------Initial F1 on dev set: %5.4f ---------------#" %f1)

        best_em = f1
        # best_em = 0.0

        for epoch in xrange(self.config.num_epochs):
            print("\n*********************EPOCH: %d running %s*********************\n" %(epoch+1,config.data_name))
            loss=self.run_epoch(session, train)
            if epoch%2==1:

                dev_f1 = self.evaluate_cnn(session, dev, config, 'dev')
                f1 = self.evaluate_cnn(session, test, config, 'test')
                self.logger.info('epoch:%d\tloss:%5.4f\tdev_f1:%5.4f\ttest_f1:%5.4f' %(epoch+1,loss,dev_f1,f1))
                print("\n#-----------Exact match on dev set: %5.4f #-----------\n" %f1)



                # if (em >= best_em):
                #     self.saver.save(session, "%s/best_model.pkl" %train_dir)
                #     print('saving to :'+train_dir+'/best_model.pkl')
                #     best_em = em
                # else:
                #     print('lower=------',em,best_em)

