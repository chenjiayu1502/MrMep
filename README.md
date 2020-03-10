# MrMep

enviroment:
tensorflow 1.4
python 3.5
tqdm

Samples for three datasets are shown in data_sample/
preprocessed data can be found in data/webnlg_seq/


To start training :

First, set the experiment setting in config.json file. For example, "data_name" can be written as "NYT" or "WEBNLG", "model_name" can be written as "baseline", "para" or "layer"

Second, in file "config.py", "train_flag" can be set as "0", means that initialing the parameters randomly and start training, "train_flag" set as "1", means that initialing the parameters from pkl file.

Third, use the command "python main.py", and start training.


