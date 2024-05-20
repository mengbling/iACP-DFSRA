import numpy as np
from tqdm import tqdm
import math
import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping
from Bio import SeqIO
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import pickle
from collections import Counter
from sklearn.model_selection import KFold
import sys
sys.path.append('.')

y_train = np.array([1] * 688+[0] * 688)
y_test = np.array([1] * 171+[0] * 171)

x_train_A = np.load('E:\代码\pythonProject\抗癌肽\特征选择\A15PACB_train.npy')
x_test_A = np.load('E:\代码\pythonProject\抗癌肽\特征选择\A15PACB_test.npy')
protbert_Xtrain_reshaped = np.load('E:\代码\pythonProject\抗癌肽\ABCAP编码\protbert_Xtrain_reshaped.npy')
protbert_Xtest_reshaped = np.load('E:\代码\pythonProject\抗癌肽\ABCAP编码\protbert_Xtest_reshaped.npy')

#############################
import pandas as pd
import lightgbm as lgb
# 假设特征 i 有 150 个特征，特征 j 有 100 个特征
A_i=2195
porbert_i= 51200
# 生成特征名称列表并合并数据
feature_namesp = [f'{i}' for i in range(porbert_i)]
feature_namesa = [f'{i}' for i in range(A_i)]
# 创建 DataFrame，并设置列名为特征名称列表
p_train = pd.DataFrame(protbert_Xtrain_reshaped, columns=feature_namesp)
p_test = pd.DataFrame(protbert_Xtest_reshaped, columns=feature_namesp)
a_train = pd.DataFrame(x_train_A, columns=feature_namesa)
a_test = pd.DataFrame(x_test_A, columns=feature_namesa)
# 打印 DataFrame，此时 DataFrame 中的列名就是特征名称
# print(df_train)
model = lgb.LGBMClassifier(
    boosting_type='gbdt',  # 设置提升类型为梯度提升决策树gbdt
    objective='binary',  # 对于分类问题，设置为二分类
    num_leaves=31,  # 定义叶子节点的数量
    learning_rate=0.05,  # 学习率
    n_estimators=100  # 迭代次数
)
model.fit(p_train, y_train)
feature_importancep = model.feature_importances_
model.fit(a_train, y_train)
feature_importancea = model.feature_importances_
# 创建一个 DataFrame 来存储特征重要性及其对应的列名
# feature_importance_df = pd.DataFrame({'Feature': df_train.columns, 'Importance': feature_importance})
# 按照重要性降序排列特征
# sorted_features = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_dictp = dict(zip(p_train.columns, feature_importancep))
sorted_featuresp = sorted(feature_importance_dictp.items(), key=lambda x: x[1], reverse=True)
feature_importance_dicta = dict(zip(a_train.columns, feature_importancea))
sorted_featuresa = sorted(feature_importance_dicta.items(), key=lambda x: x[1], reverse=True)
# 可以打印排序后的特征及其重要性
# for feature, importance in sorted_featuresp:
#     print(f"Feature: {feature}, Importance: {importance}")
# 根据需要选择保留的特征数量或重要性阈值，进行特征选择
# selected_features = [f[0] for f in sorted_features[:200]]  # 指定个数
selected_featuresp = [f[0] for f in sorted_featuresp if f[1] > 1]  # 设置重要性阈值
selected_featuresa = [f[0] for f in sorted_featuresa if f[1] > 2]  # 设置重要性阈值
# 使用选择的特征来重新定义训练集和测试集
x_train_pro = p_train[selected_featuresp]
x_test_pro = p_test[selected_featuresp]
print(x_train_pro.shape)
x_train_A = a_train[selected_featuresa]
x_test_A = a_test[selected_featuresa]
# #####################  StandardScaler 标准化数据
from sklearn.preprocessing import StandardScaler
# 初始化StandardScaler
scaler = StandardScaler()
# 对训练集进行拟合并标准化
x_train_A = scaler.fit_transform(x_train_A)
x_test_A = scaler.transform(x_test_A)
# #########################################
x_train_pro = np.expand_dims(x_train_pro, axis=2)
x_test_pro = np.expand_dims(x_test_pro, axis=2)

x_train_A = np.expand_dims(x_train_A, axis=2)
x_test_A = np.expand_dims(x_test_A, axis=2)
x_test = {"probert": x_test_pro, "A15PACB": x_test_A}

def create_cnn_model():
    global encode
    kernel_num =128
    kernel_size_1 = 1
    kernel_size_2 = 3
    kernel_size_3 = 5
    input_pro = tf.keras.Input(shape=(308, 1), name='probert')
    y = tf.keras.layers.Conv1D(128, kernel_size=kernel_size_1, strides=1, padding='same', activation=tf.nn.relu,
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_pro)
    y = tf.keras.layers.Conv1D(128, kernel_size=kernel_size_1, strides=1, padding='same', activation=tf.nn.relu,
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(y)
    block1_output = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Conv1D(128, kernel_size=kernel_size_2, strides=1, padding='same', activation=tf.nn.relu,
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(block1_output)
    y = tf.keras.layers.Conv1D(128, kernel_size=kernel_size_2, strides=1, padding='same', activation=tf.nn.relu,
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(y)
    y = tf.keras.layers.BatchNormalization()(y)
    block2_output = tf.keras.layers.add([y, block1_output])
    y = tf.keras.layers.Conv1D(128, kernel_size=kernel_size_3, padding='same', activation=tf.nn.relu,
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(block2_output)
    y = tf.keras.layers.Conv1D(kernel_num, kernel_size=kernel_size_3, padding='same', activation=tf.nn.relu,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01))(y)
    y = tf.keras.layers.BatchNormalization()(y)
    block3_output = tf.keras.layers.add([y, block2_output])
    y = tf.keras.layers.Conv1D(kernel_num, kernel_size=3, padding='same', activation=tf.nn.relu,
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(block3_output)
    y = tf.keras.layers.GlobalAveragePooling1D()(y)
    y = tf.keras.layers.Dense(128, activation='relu')(y)
    # ############################################编码1
    input_A= tf.keras.Input(shape=(343,), name='A15PACB')

    attention_probs = tf.keras.layers.Dense(343, activation='softmax', name="ATT12")(input_A)
    attention_mul = tf.keras.layers.Multiply()([input_A, attention_probs])
    # ATTENTION PART FINISHES HERE
    x= tf.keras.layers.Dense(256)(attention_mul)  # 原始的全连接

    # x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)  # 输出层

    feature_layer = tf.keras.layers.concatenate([x, y])  # 把三个通道的编码都拼接起来
    att = tf.keras.layers.Attention()([feature_layer, feature_layer, feature_layer])
    # att = tf.keras.layers.Attention()([y, y, y])
    d = tf.keras.layers.Dense(256, activation='relu')(att)
    d = tf.keras.layers.Dropout(0.1)(d)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(d)
    # model = tf.keras.Model([input_text_5, input_text, input_con], output)
    model = tf.keras.Model([input_pro, input_A], output)
    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_pro=x_train_pro.shape[1:]
input_A = x_train_A.shape[1:]
num_folds = 10
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
best_accuracy = 0.0
best_model = None
best_params = None
fold_accuracies = []
fold_aucs = []
fold_sensitivities = []
fold_specificities = []
fold_mccs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_pro, y_train)):
    print(f"Fold {fold+1}/{num_folds}")
    x_train_fold = {"probert": x_train_pro[train_idx], "A15PACB": x_train_A[train_idx]}
    x_val_fold = {"probert": x_train_pro[val_idx], "A15PACB": x_train_A[val_idx]}
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    # 创建模型
    model = create_cnn_model()
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])  # 默认lr=0.001,binary_crossentropy
    early_stopping = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True)
    # 训练模型
    model.summary()
    history = model.fit(x_train_fold, y_train_fold, epochs=300, batch_size=100, validation_data=(x_val_fold, y_val_fold),
                        callbacks=[early_stopping], verbose=1)
    # history = model.fit(x_train_fold, y_train_fold, epochs=300, batch_size=100, validation_data=(x_val_fold, y_val_fold),verbose=1)
    # 评估模型
    val_predictions = (model.predict(x_val_fold) > 0.5).astype(int)
    val_accuracy = accuracy_score(y_val_fold, val_predictions)
    fold_accuracies.append(val_accuracy)
    val_proba = model.predict(x_val_fold)
    val_auc = roc_auc_score(y_val_fold, val_proba)
    fold_aucs.append(val_auc)
    conf_matrix = confusion_matrix(y_val_fold, val_predictions)
    tn, fp, fn, tp = conf_matrix.ravel()
    sp = tn / (tn + fp)
    fold_specificities.append(sp)
    sn = tp / (tp + fn)
    fold_sensitivities.append(sn)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    fold_mccs.append(mcc)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    # plt.savefig('./loss-base1.png')
    # plt.show()

    # 保存最佳模型和参数
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model = model
        best_params = model.get_weights()
    # best_model.save('base-model十折早停35.h5')
# 输出每个折的验证准确率和AUC
for i, (acc, auc) in enumerate(zip(fold_accuracies, fold_aucs)):
    print(f"Fold {i + 1} Validation Accuracy: {acc:.4f}, AUC: {auc:.4f}")

# 输出平均准确率和AUC
mean_accuracy = np.mean(fold_accuracies)
mean_auc = np.mean(fold_aucs)
mean_sensitivity = np.mean(fold_sensitivities)
mean_specificity = np.mean(fold_specificities)
mean_mcc = np.mean(fold_mccs)
print(f"Mean Validation Accuracy: {mean_accuracy:.4f}, Mean AUC: {mean_auc:.4f}")
print(f"Mean Validation SP: {mean_specificity:.4f}, Mean SN: {mean_sensitivity:.4f}")
print(f"Mean Validation MCC: {mean_mcc:.4f}")

# 使用最佳模型和参数进行验证
best_model.set_weights(best_params)
test_proba = best_model.predict(x_test)
test_predictions = (test_proba > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, test_predictions)
test_auc = roc_auc_score(y_test, test_proba)
print(f"Best Model Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")
print("-------Best Model result-------")
cm = confusion_matrix(y_test, test_predictions)
tn, fp, fn, tp = cm.ravel()
SP = tn / (tn + fp)
SN = tp / (tp + fn)
precision = tp / (tp + fp)
F1 = f1_score(y_test, test_predictions)
MCC = matthews_corrcoef(y_test, test_predictions)
# Print evaluation metrics
print("Confusion Matrix:\n", cm)
print("Specificity (SP):", SP)
print("Sensitivity (SN):", SN)
print("Precision:", precision)
print("F1-score:", F1)
print("Matthews Correlation Coefficient (MCC):", MCC)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, test_proba)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()