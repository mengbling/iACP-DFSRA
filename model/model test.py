import numpy as np
from tqdm import tqdm
import math
import tensorflow as tf
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
A_i=2195
porbert_i= 51200
feature_namesp = [f'{i}' for i in range(porbert_i)]
feature_namesa = [f'{i}' for i in range(A_i)]

p_train = pd.DataFrame(protbert_Xtrain_reshaped, columns=feature_namesp)
p_test = pd.DataFrame(protbert_Xtest_reshaped, columns=feature_namesp)
a_train = pd.DataFrame(x_train_A, columns=feature_namesa)
a_test = pd.DataFrame(x_test_A, columns=feature_namesa)
# print(df_train)
model = lgb.LGBMClassifier(
    boosting_type='gbdt',  
    objective='binary',  
    num_leaves=31, 
    learning_rate=0.05,  
    n_estimators=100  
)
model.fit(p_train, y_train)
feature_importancep = model.feature_importances_
model.fit(a_train, y_train)
feature_importancea = model.feature_importances_
feature_importance_dictp = dict(zip(p_train.columns, feature_importancep))
sorted_featuresp = sorted(feature_importance_dictp.items(), key=lambda x: x[1], reverse=True)
feature_importance_dicta = dict(zip(a_train.columns, feature_importancea))
sorted_featuresa = sorted(feature_importance_dicta.items(), key=lambda x: x[1], reverse=True)
 
selected_featuresp = [f[0] for f in sorted_featuresp if f[1] > 1]  
selected_featuresa = [f[0] for f in sorted_featuresa if f[1] > 2] 

x_train_pro = p_train[selected_featuresp]
x_test_pro = p_test[selected_featuresp]
print(x_train_pro.shape)
x_train_A = a_train[selected_featuresa]
x_test_A = a_test[selected_featuresa]
# #####################  StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_A = scaler.fit_transform(x_train_A)
x_test_A = scaler.transform(x_test_A)

# #########################################
x_train_pro = np.expand_dims(x_train_pro, axis=2)
x_test_pro = np.expand_dims(x_test_pro, axis=2)

x_train_A = np.expand_dims(x_train_A, axis=2)
x_test_A = np.expand_dims(x_test_A, axis=2)
x_test = {"probert": x_test_pro, "A15PACB": x_test_A}
# ##################
from keras.models import load_model
model = load_model('./iACP-DFSRA.h5')
test_proba = model.predict(x_test)
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
