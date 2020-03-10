import json

class NYT_Config:
    def __init__(self, conf):
        self.data_name = conf["data_name"]
        self.num_epochs = 100
        self.batch_size = 500
        self.hidden_state_size=100
        self.learning_rate = 0.001
        self.data_dir="data/nyt_seq/"
        self.dropout_val=0.5
        self.max_length = 101
        self.test_match_file = self.data_dir+'test_match_output.json'
        self.dev_match_file = self.data_dir+'dev_match_output.json'
        self.words_id2vector_filename=self.data_dir+'words_id2vector_new.json'
        self.words_number= 90761
        self.embedding_dim = 100
        self.num_classes = 2
        self.relation_num = 25
        self.max_decode_size = 16
        self.label_weight =  5
        self.head_num=4

        self.model_name=conf["model_name"]

        

        print(self.model_name)
        self.train_dir='pkl/models_nyt_'+self.model_name
        self.log_file = 'log/log_nyt_%s.txt' %self.model_name
        self.result_file = 'result/result_nyt_%s.txt' %self.model_name
        print(self.result_file)


    
    
        self.question_train, self.context_train, self.answer_train,self.cnn_output_train,self.cnn_list_train = self.get_paths(self.data_dir,"train")
        self.question_dev ,self.context_dev ,self.answer_dev ,self.cnn_output_dev,self.cnn_list_dev= self.get_paths(self.data_dir,"dev")
        self.question_test ,self.context_test ,self.answer_test ,self.cnn_output_test,self.cnn_list_test= self.get_paths(self.data_dir,"test")

    def get_paths(self,data_dir,mode):
        question = data_dir+"%s.ids.question" %mode
        context = data_dir+"%s.ids.context" %mode
        answer = data_dir+"%s.span" %mode
        cnn_output = data_dir+"%s_cnn_output.span" %mode
        cnn_list = data_dir+"%s_cnn_output.json" %mode

        return question, context, answer ,cnn_output, cnn_list
class WebNLG_Config:
    def __init__(self, conf):
        self.data_name = conf["data_name"]
        
        self.num_epochs = 151
        self.batch_size = 100
        self.hidden_state_size=300

        self.learning_rate = 0.001
        self.decay=0.95
        self.data_dir="data/webnlg_seq/"
        self.dropout_val=0.5
        self.max_length = 81
        self.test_match_file = self.data_dir+'test_match_output.json'
        self.dev_match_file = self.data_dir+'dev_match_output.json'
        # self.words_id2vector_filename=self.data_dir+'words_id2vector_new.json'
        self.words_id2vector_filename=self.data_dir+'words_id2vector_new_big.json'

        self.words_number= 5052
        self.embedding_dim = 300
        self.num_classes = 2
        self.relation_num = 247
        self.max_decode_size = 8
        self.label_weight =  5
        self.head_num=4

        self.model_name=conf["model_name"]

        self.lambda_value=3.0

        # ablation='_no_cnn'
        ablation=''

        self.train_dir='pkl/models_webnlg_'+self.model_name+ablation
        self.log_file = 'log/log_webnlg_%s.txt' %self.model_name+ablation
        self.result_file = 'result/result_webnlg_%s.txt' %self.model_name+ablation
        print(self.model_name)
        print(self.result_file)


    
    
        self.question_train, self.context_train, self.answer_train,self.cnn_output_train,self.cnn_list_train = self.get_paths(self.data_dir,"train")
        self.question_dev ,self.context_dev ,self.answer_dev ,self.cnn_output_dev,self.cnn_list_dev= self.get_paths(self.data_dir,"dev")
        self.question_test ,self.context_test ,self.answer_test ,self.cnn_output_test,self.cnn_list_test= self.get_paths(self.data_dir,"test")

    def get_paths(self,data_dir,mode):
        question = data_dir+"%s.ids.question" %mode
        context = data_dir+"%s.ids.context" %mode
        answer = data_dir+"%s.span" %mode
        cnn_output = data_dir+"%s_cnn_output.span" %mode
        cnn_list = data_dir+"%s_cnn_output.json" %mode

        return question, context, answer ,cnn_output, cnn_list
                                           
if __name__=="__main__":
    conf=json.load(open('config.json'))
    print(conf)
    if conf['data_name']=="NYT":
        config=NYT_Config(conf)
    else:
        config=WebNLG_Config(conf)
    print(config.data_name)
