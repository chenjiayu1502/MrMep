# -*- coding: utf-8 -*- 
import json
import numpy as np
def assert_pad_into_sentence(data_dir, in_file1,out_file1,in_file2,out_file2):
	'''
	在每个句子前插入占位符号，占位符号的词向量初始化为全零
	两个实体的起始位置和终止位置都加一
	'''
	words2id = json.load(open(data_dir+'words2id.json'))
	pad_id =len(words2id)

	data = json.load(open(in_file1))
	sen_len = data[0]
	sen_id = data[1]
	en_id = json.load(open(in_file2))


	assert len(sen_id)==len(en_id)
	new_sen_id=[]
	for i in range(len(sen_id)):
		temp=[pad_id]
		temp.extend(sen_id[i])
		new_sen_id.append(temp)
		sen_len[i]+=1

		for j in range(len(en_id[i])):
			for k in range(1,len(en_id[i][j])):
				for m in range(len(en_id[i][j][k])):
					en_id[i][j][k][m]+=1
		if i%10==0:
			print(i)
		# break
	json.dump([sen_len, new_sen_id, data[2]],open(out_file1,'w'),indent=2)


	json.dump(en_id,open(out_file2,'w'),indent=2)

def prepare(data_dir, relation_num):
	# #train
	in_file = data_dir+'train_new.json'
	in_entity_file = data_dir+'train_entity_new.json'

	out_context_file = data_dir+'train.ids.context'
	out_question_file = data_dir+'train.ids.question'
	out_answer_file = data_dir+'train.span'
	#01向量表示的cnn输出结果
	out_cnn_file = data_dir+'train_cnn_output.span'
	#关系id以列表形式表示的结果
	out_cnn_json_file = data_dir+'train_cnn_output.json'

	relation_num=25

	

	f_context = open(out_context_file,'w')
	f_question = open(out_question_file,'w')
	f_answer = open(out_answer_file,'w')
	f_cnn = open(out_cnn_file,'w')
	f_cnn_json = open(out_cnn_json_file,'w')

	data = json.load(open(in_file))
	en_data = json.load(open(in_entity_file))

	sen=data[1]
	assert len(sen) == len(en_data)
	cnn_all_list=[]
	len_cnt={}
	for i in range(len(sen)):
		cnn_01_list=['0']*relation_num
		cnn_list=[]
		temp_rel = {}#存储关系以及对应的实体对
		for j in range(len(en_data[i])):
			rel=en_data[i][j][0]
			if rel not in temp_rel:
				temp_rel[rel]=[]
			en = en_data[i][j]
			ans = [en[1][0],en[1][-1],en[2][0],en[2][-1]]
			ans_str = [str(k) for k in ans]
			temp_rel[rel].extend(ans_str)
			# f_answer.write(' '.join(ans_str)+'\n')

			cnn_01_list[rel]='1'
			cnn_list.append(rel)
		for rel,v in temp_rel.items():
			lens=len(v)//4
			if lens not in len_cnt:
				len_cnt[lens]=1
			else:
				len_cnt[lens]+=1
			temp_sen = [str(k) for k in sen[i]]
			f_context.write(' '.join(temp_sen)+'\n')
			f_question.write(str(rel)+'\n')
			f_answer.write(' '.join(v)+'\n')
		cnn_all_list.append(cnn_list)
		for j in range(len(temp_rel)):
			f_cnn.write(' '.join(cnn_01_list)+'\n')
	print(len_cnt)

def prepare_test(data_dir, relation_num,mode):

	#test
	# in_file = 'test_new.json'
	# in_entity_file = 'test_entity_new.json'

	# out_context_file = 'test.ids.context'
	# out_question_file = 'test.ids.question'
	# out_answer_file = 'test.span'
	# #01向量表示的cnn输出结果
	# out_cnn_file = 'test_cnn_output.span'
	# #关系id以列表形式表示的结果
	# out_cnn_json_file = 'test_cnn_output.json'
	# out_match_json_file = 'test_match_output.json'
	#dev
	# in_file = 'dev_new.json'
	# in_entity_file = 'dev_entity_new.json'

	# out_context_file = 'dev.ids.context'
	# out_question_file = 'dev.ids.question'
	# out_answer_file = 'dev.span'
	# #01向量表示的cnn输出结果
	# out_cnn_file = 'dev_cnn_output.span'
	# #关系id以列表形式表示的结果
	# out_cnn_json_file = 'dev_cnn_output.json'
	# out_match_json_file = 'dev_match_output.json'

	# relation_num=25


	in_file = data_dir+'%s_new.json'%mode
	in_entity_file = data_dir+'%s_entity_new.json'%mode

	out_context_file = data_dir+'%s.ids.context'%mode
	out_question_file = data_dir+'%s.ids.question'%mode
	out_answer_file = data_dir+'%s.span'%mode
	#01向量表示的cnn输出结果
	out_cnn_file = data_dir+'%s_cnn_output.span'%mode
	#关系id以列表形式表示的结果
	out_cnn_json_file = data_dir+'%s_cnn_output.json'%mode
	out_match_json_file = data_dir+'%s_match_output.json'%mode

	

	f_context = open(out_context_file,'w')
	f_question = open(out_question_file,'w')
	f_answer = open(out_answer_file,'w')
	f_cnn = open(out_cnn_file,'w')
	f_cnn_json = open(out_cnn_json_file,'w')
	f_match_json = open(out_match_json_file,'w')

	data = json.load(open(in_file))
	en_data = json.load(open(in_entity_file))

	sen=data[1]
	assert len(sen) == len(en_data)
	cnn_all_list=[]
	match_all_list = []
	for i in range(len(sen)):
		cnn_01_list=['0']*relation_num
		cnn_list=[]
		
		match_list=[]
		for k in range(relation_num):
			match_list.append([])
		for j in range(len(en_data[i])):

			temp_sen = [str(k) for k in sen[i]]

			rel=en_data[i][j][0]

			en = en_data[i][j]
			ans = [en[1][0],en[1][-1],en[2][0],en[2][-1]]
			if 0 in ans:
				print(ans)
			ans_str = [str(k) for k in ans]

			cnn_01_list[rel]='1'
			cnn_list.append(rel)
			match_list[rel].append(ans)
			
		cnn_all_list.append(cnn_list)
		match_all_list.append(match_list)

		f_context.write(' '.join(temp_sen)+'\n')
		f_question.write(str(rel)+'\n')
		f_answer.write(' '.join(ans_str)+'\n')
		f_match_json.write(json.dumps({'data':match_list})+'\n')



		# print(cnn_list)
		# for j in range(len(en_data[i])):
		f_cnn.write(' '.join(cnn_01_list)+'\n')
		# # break
		# json.dump(match_all_list,f_match_json)
		# json.dump(cnn_all_list,f_cnn_json)
def add_pad_embedding(data_dir):
	words_id2vector = json.load(open(data_dir+'words_id2vector.json'))
	pad_id = len(words_id2vector)
	print(pad_id)
	words_id2vector[str(pad_id)]=['0.0']*100
	json.dump(words_id2vector,open(data_dir+'words_id2vector_new.json','w'))




def make_dev_data(in_file1,out_file1,in_file2,out_file2, num):
	
	data=json.load(open(in_file1))
	entity=json.load(open(in_file2))
	import random
	new_data=[[],[],[]]
	new_entity=[]
	ids=random.sample(list(range(0,len(data[0]))),num)
	for ind in ids:
		for i in range(3):
			new_data[i].append(data[i][ind])
		new_entity.append(entity[ind])
	json.dump(new_data, open(out_file1, 'w'))
	json.dump(new_entity, open(out_file2, 'w'))




if __name__ == "__main__":
	data_dir='data/nyt/'
	dev_num=5000
	relation_num=25
	in_file1 = data_dir+'train.json'
	in_file2 = data_dir+'train_entity.json'
	out_file1 = data_dir+'train_new.json'
	out_file2 = data_dir+'train_entity_new.json'
	assert_pad_into_sentence(data_dir,in_file1,out_file1,in_file2,out_file2)
	in_file1=data_dir+'train_new.json'
	in_file2=data_dir+'train_entity_new.json'
	out_file1=data_dir+'dev_new.json'
	out_file2=data_dir+'dev_entity_new.json'
	make_dev_data(in_file1,out_file1,in_file2,out_file2, dev_num)
	in_file1 = data_dir+'test.json'
	in_file2 = data_dir+'test_entity.json'
	out_file1 = data_dir+'test_new.json'
	out_file2 = data_dir+'test_entity_new.json'
	assert_pad_into_sentence(data_dir,in_file1,out_file1,in_file2,out_file2)

	add_pad_embedding(data_dir)
	prepare(data_dir, relation_num)
	prepare_test(data_dir, relation_num, 'dev')
	prepare_test(data_dir, relation_num, 'test')

	
