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
            line = line.strip()  # 去掉每行头尾空白
            if line != "":  # 跳过空行
                if aessnn3_name is None:  # 当前行是属性名称行
                    aessnn3_name = line
                else:  # 当前行是属性值行
                    # 将属性值转换为浮点数
                    values = line.split()
                    aessnn3_data[i] = {}  # 定义一维字典
                    j = 0
                    for let in letter:
                        if let=='B':  # 本来有0所以设为0.001
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
            line = line.strip() # 去掉每行头尾空白
            if line!="":  # 跳过空行
                if property_name is None:  # 当前行是属性名称行
                    property_name = line
                else:  # 当前行是属性值行
                    # 将属性值转换为浮点数
                    values = line.split()
                    properties_data[values[0]]={}  # 定义一维字典
                    j = 0
                    for let in letter:
                        if let=='B':# 性质里有0，所以这里将B设为0.001
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

    # 从文件中读取氨基酸性质信息
    with open('../组合编码/PAAC.txt', 'r') as f:
        records = f.readlines()

    # 处理氨基酸信息
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {AA[i]: i for i in range(len(AA))}
    AAProperty = [list(map(float, records[i].rstrip().split()[1:])) for i in range(1, len(records))]

    # 对氨基酸信息进行归一化处理
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
            line = line.strip() # 去掉每行头尾空白
            if line!="":  # 跳过空行
                if property_name is None:  # 当前行是属性名称行
                    property_name = line
                else:  # 当前行是属性值行
                    # 将属性值转换为浮点数
                    values = line.split()
                    properties_data[values[0]]={}  # 定义一维字典
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
# 读取FASTA文件内容
def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('>'):
                sequences.append(line)
    return sequences
# #####################
# 读取训练集和测试集FASTA文件内容
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
                                               blousm62_train_encodings), axis=1)  # 拼接）
concatenated_Xtest_features = np.concatenate((AESNN3_test_encodings, AAindex_test_encodings, paac_test_encodings, CTDD_Xtest_sequences,
                                              blousm62_test_encodings), axis=1)

print(concatenated_Xtrain_features.shape)
print(concatenated_Xtest_features.shape)

# 保存数组到文件
np.save('新A15PACB_train.npy', concatenated_Xtrain_features)
np.save('新A15PACB_test.npy', concatenated_Xtest_features)
# 加载数组
# concatenated_Xtrain_features = np.load('concatenated_Xtrain_features.npy')
# concatenated_Xtest_features = np.load('concatenated_Xtest_features.npy')

# def create_cnn_model(input_shape):
#     model = tf.keras.Sequential([
#         # Convolutional layers
#         tf.keras.layers.Conv1D(64, kernel_size=1, activation='PReLU', input_shape=input_shape),
#         tf.keras.layers.MaxPooling1D(pool_size=2),
#         # tf.keras.layers.GlobalAveragePooling1D(),
#         tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
#         tf.keras.layers.MaxPooling1D(pool_size=2),
#         # tf.keras.layers.Conv1D(256, kernel_size=7, activation='relu'),
#         # tf.keras.layers.MaxPooling1D(pool_size=2),
#         # tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
#         tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True)),
#         # Flatten and dense layers for classification
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation='relu'),
#         # tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(1.0000e-02))
# ])
#     return model
# # #############################
# import pandas as pd
# import lightgbm as lgb
# # 假设特征 i 有 150 个特征，特征 j 有 100 个特征
# AESNN3_i=150
# zsacle_i = 250
# BLOUSM62_i = 1050
# CTDD_i = 195
# AAINDEX15_i = 750
# PAAC_i = 50
# # 生成特征名称列表并合并数据
# feature_names = [f'AESNN3_{i}' for i in range(AESNN3_i)] + [f'AAINDEX15_{i}' for i in range(AAINDEX15_i)] \
#                 + [f'PAAC_{i}' for i in range(PAAC_i)] \
#                 + [f'CTDD_{i}' for i in range(CTDD_i)] + [f'BLOUSM62_{i}' for i in range(BLOUSM62_i)]
#
# # 创建 DataFrame，并设置列名为特征名称列表
# df_train = pd.DataFrame(concatenated_Xtrain_features, columns=feature_names)
# df_test = pd.DataFrame(concatenated_Xtest_features, columns=feature_names)
# # 打印 DataFrame，此时 DataFrame 中的列名就是特征名称
# # print(df_train)
# model = lgb.LGBMClassifier(
#     boosting_type='gbdt',  # 设置提升类型为梯度提升决策树gbdt,rf,dart
#     objective='binary',  # 对于分类问题，设置为二分类
#     num_leaves=31,  # 定义叶子节点的数量31
#     learning_rate=0.05,  # 学习率0.05
#     n_estimators=100  # 迭代次数
# )
# model.fit(df_train, y_train)
# feature_importance = model.feature_importances_
# # 创建一个 DataFrame 来存储特征重要性及其对应的列名
# # feature_importance_df = pd.DataFrame({'Feature': df_train.columns, 'Importance': feature_importance})
# # 按照重要性降序排列特征
# # sorted_features = feature_importance_df.sort_values(by='Importance', ascending=False)
# feature_importance_dict = dict(zip(df_train.columns, feature_importance))
# sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
# # 可以打印排序后的特征及其重要性
# for feature, importance in sorted_features:
#     print(f"Feature: {feature}, Importance: {importance}")
# # 根据需要选择保留的特征数量或重要性阈值，进行特征选择
# # selected_features = [f[0] for f in sorted_features[:200]]  # 指定个数
# selected_features = [f[0] for f in sorted_features if f[1] > 2]  # 设置重要性阈值343
# # 使用选择的特征来重新定义训练集和测试集
# X_train_selected = df_train[selected_features]
# X_test_selected = df_test[selected_features]
# print(X_train_selected.shape)
# # #########################################
# x_train = np.expand_dims(X_train_selected, axis=2)
# x_test = np.expand_dims(X_test_selected, axis=2)
# input_shape = x_train.shape[1:]
# # 定义十折交叉验证
# num_folds = 10
# skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
# best_accuracy = 0.0
# best_model = None
# best_params = None
# fold_accuracies = []
# fold_aucs = []
# # 迭代十折交叉验证
# for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
#     print(f"Fold {fold+1}/{num_folds}")
#
#     x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
#     y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
#     # 创建模型
#     model = create_cnn_model(input_shape)
#     # 自定义二分类损失函数
#     def custom_binary_loss(y_true, y_pred, b=0.1):
#         # 将预测值 y_pred 限制在一个很小的范围内，避免 log(0) 或 log(1) 的情况出现
#         epsilon = 1e-7
#         y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
#         y_true = tf.cast(y_true, y_pred.dtype)
#         # 计算二分类交叉熵损失
#         loss = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
#         loss1 = abs(loss - b) + b
#         # return tf.reduce_mean(loss)
#         return tf.reduce_mean(loss1)
#     tf.keras.utils.get_custom_objects()['custom_binary_loss'] = custom_binary_loss
#     # #####################binary_crossentropy
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])  # 默认lr=0.001,binary_crossentropy
#     # early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
#     # # 训练模型
#     # history = model.fit(x_train_fold, y_train_fold, epochs=50, batch_size=64, validation_data=(x_val_fold, y_val_fold),
#     #                     callbacks=[early_stopping], verbose=1)
#     model.fit(x_train_fold, y_train_fold, epochs=50, batch_size=64, validation_data=(x_val_fold, y_val_fold),verbose=1)
#     # 评估模型
#     val_predictions = (model.predict(x_val_fold) > 0.5).astype(int)
#     val_accuracy = accuracy_score(y_val_fold, val_predictions)
#     fold_accuracies.append(val_accuracy)
#     val_proba = model.predict(x_val_fold)
#     val_auc = roc_auc_score(y_val_fold, val_proba)
#     fold_aucs.append(val_auc)
#
#     # 保存最佳模型和参数
#     if val_accuracy > best_accuracy:
#         best_accuracy = val_accuracy
#         best_model = model
#         best_params = model.get_weights()
#     best_model.save('A15PACB+fsmm3.h5')
#
# # 输出每个折的验证准确率和AUC
# for i, (acc, auc) in enumerate(zip(fold_accuracies, fold_aucs)):
#     print(f"Fold {i + 1} Validation Accuracy: {acc:.4f}, AUC: {auc:.4f}")
#
# # 输出平均准确率和AUC
# mean_accuracy = np.mean(fold_accuracies)
# mean_auc = np.mean(fold_aucs)
# print(f"Mean Validation Accuracy: {mean_accuracy:.4f}, Mean AUC: {mean_auc:.4f}")
#
# # 使用最佳模型和参数进行验证
# best_model.set_weights(best_params)
# test_proba = best_model.predict(x_test)
# test_predictions = (test_proba > 0.5).astype(int)
# test_accuracy = accuracy_score(y_test, test_predictions)
# test_auc = roc_auc_score(y_test, test_proba)
# print(f"Best Model Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")
# print("-------Best Model result-------")
# cm = confusion_matrix(y_test, test_predictions)
# tn, fp, fn, tp = cm.ravel()
# SP = tn / (tn + fp)
# SN = tp / (tp + fn)
# precision = tp / (tp + fp)
# F1 = f1_score(y_test, test_predictions)
# MCC = matthews_corrcoef(y_test, test_predictions)
# # Print evaluation metrics
# print("Confusion Matrix:\n", cm)
# print("Specificity (SP):", SP)
# print("Sensitivity (SN):", SN)
# print("Precision:", precision)
# print("F1-score:", F1)
# print("Matthews Correlation Coefficient (MCC):", MCC)
#
# # Plot ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, test_proba)
# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.show()