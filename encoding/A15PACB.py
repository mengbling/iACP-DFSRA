import sys
import numpy as np
from tqdm import tqdm
import math
import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping
from Bio import SeqIO
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pickle
from collections import Counter


letter = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B']
def load_aessnn3_data(filename):
    aessnn3_data = {}
    # print(letter[0])
    with open(filename, 'r') as file:
        lines = file.readlines()
        aessnn3_name = None
        aessnn3_values = []
        i = 0
        for line in lines:
            line = line.strip()  
            if line != "":  
                if aessnn3_name is None:  
                    aessnn3_name = line
                else:  
                    values = line.split()
                    aessnn3_data[i] = {}  
                    j = 0
                    for let in letter:
                        if let=='B':  
                            aessnn3_data[i][let]=0.001
                        else:
                            aessnn3_data[i][let]=float(values[j])
                            j=j+1
                    i=i+1
    return aessnn3_data

def aessnn3_encode(sequence, aessnn3_data):
    encoded_sequence = []

    for residue in sequence:
        for key in aessnn3_data:
            print(key)
            if residue in letter:
              residue_encoding = []
              print(aessnn3_data[key][residue])
              residue_encoding.append(aessnn3_data[key][residue])
            else:
                residue_encoding.extend([0.0] * len(aessnn3_data[key]))  # Default values as zeros
            encoded_sequence.append(residue_encoding)

    return encoded_sequence
aessnn3_data = load_aessnn3_data('E:/翻译后修饰/新去冗余数据/去掉X/AESNN3数值.txt')
# ##############################
def load_properties_data(filename):
    properties_data = {}

    print(letter[0])
    with open(filename, 'r') as file:
        lines = file.readlines()
        property_name = None
        i = 0
        for line in lines:
            line = line.strip() 
            if line!="":  
                if property_name is None: 
                    property_name = line
                else: 
                    values = line.split()
                    properties_data[values[0]]={}  
                    j = 0
                    for let in letter:
                        if let=='B':
                            properties_data[values[0]][let]=0.001
                        else:
                            properties_data[values[0]][let]=float(values[j+1])
                            j=j+1
                    i = i+1

    print(properties_data)
    return properties_data

def AAindex_encode(sequence, properties_data):
    encoded_sequence = []
    m=0
    for residue in sequence:
        for key in properties_data:
            if residue in letter:
              residue_encoding = []
              print(properties_data[key][residue])
              residue_encoding.append(properties_data[key][residue])
            else:
                residue_encoding.extend([0.0] * len(properties_data[key]))  # Default values as zeros
            encoded_sequence.append(residue_encoding)
    return encoded_sequence
aaindex_data = load_properties_data('E:/抗癌肽预测/补齐50数据/抗菌肽性质15.txt')

# ## #######################
def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def PAAC(fastas, lambdaValue=30, w=0.05, **kw):
    if get_min_sequence_length(fastas) < lambdaValue + 1:
        print('Error: all the sequence length should be larger than lambdaValue + 1: ' + str(lambdaValue + 1) + '\n\n')
        return 0

    with open('../组合编码/PAAC.txt', 'r') as f:
        records = f.readlines()

    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {AA[i]: i for i in range(len(AA))}
    AAProperty = [list(map(float, records[i].rstrip().split()[1:])) for i in range(1, len(records))]

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    encodings = []
    for sequence in fastas:
        code = []
        theta = []
        if 'B' in sequence:
            sequence = sequence.replace('B', '0.001')
        for n in range(1, lambdaValue + 1):
            # theta.append(
            #     sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
            #         len(sequence) - n))
            theta.append(sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1)
                              for j in range(len(sequence) - n)
                              if sequence[j] in AADict and sequence[j + n] in AADict]) / (len(sequence) - n))
        myDict = Counter(sequence)
        # code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [(myDict[aa] / (1 + w * sum(theta))) if aa in AADict else 0.001 for aa in AA]  # 是否有B
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
    return encodings

def get_min_sequence_length(fastas):
    return min(len(sequence) for sequence in fastas)

# ################
def Count(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >=1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code

def CTDD(sequence):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }
    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    for seq in sequence:
        code = []
        for p in property:
            code = code + Count(group1[p], seq) + Count(group2[p], seq) + Count(group3[p], seq)
        encodings.append(code)
    return encodings
# ####################
def load_properties_data(filename):
    properties_data = {}

    print(letter[0])
    with open(filename, 'r') as file:
        lines = file.readlines()
        property_name = None
        i = 0
        for line in lines:
            line = line.strip() 
            if line!="": 
                if property_name is None:  
                    property_name = line
                else: 
                    values = line.split()
                    properties_data[values[0]]={} 
                    j = 0
                    for let in letter:
                        properties_data[values[0]][let]=float(values[j+1])
                        j=j+1
                    i = i+1

    print(properties_data)
    return properties_data

def blousm62_encode(sequence, properties_data):
    encoded_sequence = []
    for residue in sequence:
        for key in properties_data:
            if residue in letter:
              residue_encoding = []
              print(properties_data[key][residue])
              residue_encoding.append(properties_data[key][residue])
            else:
                residue_encoding.extend([0.0] * len(properties_data[key]))  # Default values as zeros
            encoded_sequence.append(residue_encoding)
    return encoded_sequence
blousm62_data = load_properties_data('E:/翻译后修饰/新去冗余数据/去掉X/blousm62.txt')
# ####################
Xtest_file = "E:/抗癌肽预测/补齐50数据/新测试集.txt"
Xtrain_file = "E:/抗癌肽预测/补齐50数据/新训练集.txt"

# y_train = np.array([1] * 874+[0] * 4253)
y_train = np.array([1] * 688+[0] * 688)
y_test = np.array([1] * 171+[0] * 171)
# 
def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('>'):
                sequences.append(line)
    return sequences
# #####################

Xtrain_sequences = read_fasta(Xtrain_file)
Xtest_sequences = read_fasta(Xtest_file)

AAindex_train_sequences = []
for sequence in Xtrain_sequences:
    encoded_sequence = AAindex_encode(sequence, aaindex_data)
    print(encoded_sequence)
    AAindex_train_sequences.append(encoded_sequence)
AAindex_train_encodings = np.array(AAindex_train_sequences)
print(AAindex_train_encodings.shape)
AAindex_test_sequences = []
for sequence in Xtest_sequences:
    encoded_sequence = AAindex_encode(sequence, aaindex_data)
    AAindex_test_sequences.append(encoded_sequence)
AAindex_test_encodings = np.array(AAindex_test_sequences)

AAindex_train_encodings = AAindex_train_encodings.reshape(1376, -1)
AAindex_test_encodings = AAindex_test_encodings.reshape(342, -1)
# ##########################
AESNN3_train_sequences = []
for sequence in Xtrain_sequences:
    encoded_sequence = aessnn3_encode(sequence, aessnn3_data)
    print(encoded_sequence)
    AESNN3_train_sequences.append(encoded_sequence)
AESNN3_train_encodings = np.array(AESNN3_train_sequences)
print(AESNN3_train_encodings.shape)
AESNN3_test_sequences = []
for sequence in Xtest_sequences:
    encoded_sequence = aessnn3_encode(sequence, aessnn3_data)
    AESNN3_test_sequences.append(encoded_sequence)
AESNN3_test_encodings = np.array(AESNN3_test_sequences)
AESNN3_train_encodings = AESNN3_train_encodings.reshape(1376, -1)
AESNN3_test_encodings = AESNN3_test_encodings.reshape(342, -1)
# ########################
paac_train_encodings = PAAC(Xtrain_sequences)
paac_test_encodings = PAAC(Xtest_sequences)
paac_train_encodings = np.array(paac_train_encodings)
paac_test_encodings = np.array(paac_test_encodings)
# #########################
CTDD_Xtrain_features = CTDD(Xtrain_sequences)
CTDD_Xtrain_sequences = np.array(CTDD_Xtrain_features)
print(CTDD_Xtrain_sequences.shape)
CTDD_Xtest_features = CTDD(Xtest_sequences)
CTDD_Xtest_sequences = np.array(CTDD_Xtest_features)
# ########################################
blousm62_train_sequences = []
for sequence in Xtrain_sequences:
    encoded_sequence = blousm62_encode(sequence, blousm62_data)
    print(encoded_sequence)
    blousm62_train_sequences.append(encoded_sequence)
blousm62_train_encodings = np.array(blousm62_train_sequences)
print(blousm62_train_encodings.shape)
blousm62_test_sequences = []
for sequence in Xtest_sequences:
    encoded_sequence = blousm62_encode(sequence, blousm62_data)
    blousm62_test_sequences.append(encoded_sequence)
blousm62_test_encodings = np.array(blousm62_test_sequences)

blousm62_train_encodings = blousm62_train_encodings.reshape(1376, -1)
blousm62_test_encodings = blousm62_test_encodings.reshape(342, -1)
# ########################
concatenated_Xtrain_features = np.concatenate((AESNN3_train_encodings, AAindex_train_encodings, paac_train_encodings, CTDD_Xtrain_sequences,
                                               blousm62_train_encodings), axis=1)）
concatenated_Xtest_features = np.concatenate((AESNN3_test_encodings, AAindex_test_encodings, paac_test_encodings, CTDD_Xtest_sequences,
                                              blousm62_test_encodings), axis=1)

print(concatenated_Xtrain_features.shape)
print(concatenated_Xtest_features.shape)
# 
np.save('新A15PACB_train.npy', concatenated_Xtrain_features)
np.save('新A15PACB_test.npy', concatenated_Xtest_features)
