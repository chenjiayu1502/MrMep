import json

class Config:
    def __init__(self, conf):
        self.data_name = conf["data_name"]
        self.num_epochs = 100
        self.batch_size = 500
        self.train_embeddings=0
        self.hidden_state_size=100
        self.learning_rate = 0.001
        self.data_dir="data/nyt_seq/"
        self.dropout_val=0.5
        self.max_length = 101
        self.match_file = self.data_dir+'test_match_output.json'
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
        self.question_dev ,self.context_dev ,self.answer_dev ,self.cnn_output_dev,self.cnn_list_dev= self.get_paths(self.data_dir,"test")
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
    config=Config(conf)
    print(config.data_name)