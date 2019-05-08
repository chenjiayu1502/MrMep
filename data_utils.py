import numpy as np

class dataset(object):
    def __init__(self, question_file, context_file, answer_file, cnn_output_file, cnn_list_file):
        """
        Args:
            filename: path to the files
        """
        self.question_file = question_file
        self.context_file = context_file
        self.answer_file = answer_file
        self.cnn_output_file = cnn_output_file
        self.cnn_list_file = cnn_list_file

        self.length = None

    def iter_file(self, filename):
        
        with open(filename) as f:
            for line in f:
                line = line.strip().split()
                # line = map(lambda tok: int(tok), line)
                new_line=[int(k) for k in line]
                # print(line)
                yield new_line
    def iter_question_file(self,filename):
        with open(filename) as f:
            for line in f:
                line = line.strip().split(" ")
                line = map(lambda tok: [int(tok)], line)
                # print(line)
                yield line
    def iter_answer_file(self,filename):
        with open(filename) as f:
            for line in f:
                line = line.strip().split(" ")
                line = map(lambda tok: int(tok), line)
                temp=[]
                for i in range(int(len(line)/4)):
                    temp.append([line[i*4],line[i*4+1],line[i*4+2],line[i*4+3]])
                yield temp



    def __iter__(self):
        niter = 0
        # print('context_file_iter ')
        context_file_iter = self.iter_file(self.context_file)
        # print('answer_file_iter ')
        answer_file_iter = self.iter_file(self.answer_file)
        # print('question_file_iter')
        question_file_iter = self.iter_file(self.question_file)
        # print('cnn_file_iter ')
        cnn_file_iter = self.iter_file(self.cnn_output_file)

        for question, context, answer, cnn_output in zip(question_file_iter, context_file_iter, answer_file_iter, cnn_file_iter):
            yield (question, context, answer, cnn_output)



    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            # lines=open(self.question_file,'r').readlines()
            # self.length = len(lines)
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length



def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return np.array(sequence_padded), np.array(sequence_length)
def pad_sequences(sequences, pad_tok):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    max_length = max([len(x) for x in sequences])
    # max_length = config.max_length
    # print('max_length:',max_length)
    sequence_padded, sequence_length = _pad_sequences(sequences, 
                                            pad_tok, max_length)
    # print(max(sequence_length))
    return sequence_padded, sequence_length


def pad_sequences_for_passage(sequences, pad_tok,config):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    # max_length = max([len(x) for x in sequences])
    max_length = config.max_length
    # print('max_length:',max_length)
    sequence_padded, sequence_length = _pad_sequences(sequences, 
                                            pad_tok, max_length)
    # print(max(sequence_length))
    return sequence_padded, sequence_length

def pad_sequences_for_query(sequences, pad_tok,config):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    max_length = max([len(x) for x in sequences])
    # max_length = config.max_length
    # print('max_length:',max_length)
    sequence_padded, sequence_length = _pad_sequences(sequences, 
                                            pad_tok, max_length)
    return sequence_padded, sequence_length

def pad_sequences_for_answer(sequences, pad_tok,config):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    # max_length = max([len(x) for x in sequences])
    max_length = config.max_decode_size
    # max_length = config.max_length
    # print('max_length:',max_length)
    sequence_padded, sequence_length = _pad_sequences(sequences, 
                                            pad_tok, max_length)

    return sequence_padded, sequence_length



def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (question, context, answer) tuples
        minibatch_size: (int)
    Returns: 
        list of tuples
    """
    # print('minibatch_size====',minibatch_size)
    question_batch, context_batch, answer_batch ,cnn_batch= [], [], [],[]

    for (q, c, a, co) in data:
        if len(question_batch) == minibatch_size:
            # print(cnn_batch[0])
            yield question_batch, context_batch, answer_batch,cnn_batch
            question_batch, context_batch, answer_batch ,cnn_batch= [], [], [],[]
        
        question_batch.append(q)
        context_batch.append(c)
        answer_batch.append(a)
        cnn_batch.append(co)
        # break

    if len(question_batch) != 0:
        yield question_batch, context_batch, answer_batch,cnn_batch


