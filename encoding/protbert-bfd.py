# import tensorflow as tf
# from transformers import TFBertModel, BertTokenizer,BertConfig
# import re
# import numpy as np
#
# tokenizer = BertTokenizer.from_pretrained("E:/翻译后修饰/代码+压缩包/hugging-face/protbert_bfd", do_lower_case=False)
# model = TFBertModel.from_pretrained("E:/翻译后修饰/代码+压缩包/hugging-face/protbert_bfd", from_pt=True)
# sequences_Example = ["A E T C Z A O","S K T Z P"]
# sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
# ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True, return_tensors="tf")
# input_ids = ids['input_ids']
# attention_mask = ids['attention_mask']
# embedding = model(input_ids)[0]
# embedding = np.asarray(embedding)
# attention_mask = np.asarray(attention_mask)
# features = []
# for seq_num in range(len(embedding)):
#     seq_len = (attention_mask[seq_num] == 1).sum()
#     seq_emd = embedding[seq_num][1:seq_len-1]
#     features.append(seq_emd)
# print(features)

from transformers import BertTokenizer, TFBertModel
import numpy as np
import re
from tqdm import tqdm
import math
import tensorflow as tf
from keras.callbacks import EarlyStopping
from Bio import SeqIO
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pickle
from collections import Counter

def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ""
        for line in file:
            if line.startswith('>'):
                if sequence:
                    sequences.append(" ".join(sequence)) 
                    sequence = "" 
            else:
                sequence += line.strip()  
        if sequence:  
            sequences.append(" ".join(sequence))
    return sequences

# Xtest_file = "E:/抗癌肽预测/原始data/ACP20mainTest.fasta"
# Xtrain_file = "E:/抗癌肽预测/原始data/ACP20mainTrain.fasta"
Xtest_file = "E:/抗癌肽预测/补齐50数据/新测试集.txt"
Xtrain_file = "E:/抗癌肽预测/补齐50数据/新训练集.txt"

# y_train = np.array([1] * 874+[0] * 4253)
y_train = np.array([1] * 688+[0] * 688)
y_test = np.array([1] * 171+[0] * 171)

Xtest_sequences = read_fasta(Xtest_file)
Xtrain_sequences = read_fasta(Xtrain_file)


Xtest_sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in Xtest_sequences]
Xtrain_sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in Xtrain_sequences]

tokenizer = BertTokenizer.from_pretrained("E:/翻译后修饰/代码+压缩包/hugging-face/protbert_bfd", do_lower_case=False)
model1 = TFBertModel.from_pretrained("E:/翻译后修饰/代码+压缩包/hugging-face/protbert_bfd", from_pt=True)

def get_embeddings(sequences):
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True, return_tensors="tf")
    input_ids = ids['input_ids']
    attention_mask = ids['attention_mask']
    embedding = model1(input_ids)[0]
    embedding = np.array(embedding)
    attention_mask = np.array(attention_mask)
    features = []
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][1:seq_len - 1]
        features.append(seq_emd)
    return features
#
x_test_features = get_embeddings(Xtest_sequences)
x_train_features = get_embeddings(Xtrain_sequences)

x_test_features = np.array(x_test_features)
x_train_features = np.array(x_train_features)
print(x_train_features.shape)
print(x_test_features.shape)
protbert_Xtrain_reshaped = x_train_features.reshape(1376, -1)
protbert_Xtest_reshaped = x_test_features.reshape(342, -1)
print(protbert_Xtrain_reshaped.shape)
print(protbert_Xtest_reshaped.shape)
np.save('新protbert_Xtrain.npy', protbert_Xtrain_reshaped)
np.save('新protbert_Xtest.npy', protbert_Xtest_reshaped)

