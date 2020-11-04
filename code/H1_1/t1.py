from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
# from sklearn.externals import joblib
import joblib
from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
from datahandler import load_data_set_train

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
'''
import warnings
warnings.filterwarnings("ignore")  # 忽略版本问题
'''
import datahandler

'''
import sys
f = open('20.txt', 'a')
sys.stdout = f
sys.stderr = f		# redirect std err, if necessary
'''

def all_Algorithms():
    all_algorithms_score = []
    all_algorithms_score_avg = []
    all_algorithms_name = ["MLPClassifier"]

    # load data
    dataset_train = load_data_set_train()
    array = dataset_train.values
    x = array[:, 1:len(list(dataset_train)) - 1]
    y = array[:, 0]
    validation_size = 0.20
    seed = 7 #default: 7
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=validation_size,
                                                                                    random_state=seed)

    accuracy, average_mae_history = MLPClassifier_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    all_algorithms_score_avg.append(average_mae_history)

    return all_algorithms_name, all_algorithms_score, all_algorithms_score_avg, list(dataset_train), list(
        dataset_train.mean())

def MLPClassifier_Algorithms(x_train, x_validation, y_train, y_validation):
    # MLPClassifier Algorithms
    seed = 7
    all_mae_histories = []
    k = 10
    num_val_samples = len(x_train) // k
    for i in range(k):
        # 准备验证数据，第K个分区的数据
        val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

        # 准备训练数据，其他所有分区的数据
        partial_train_data = np.concatenate(
            [x_train[:i * num_val_samples],
             x_train[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [y_train[:i * num_val_samples],
             y_train[(i + 1) * num_val_samples:]],
            axis=0)
        # 构建 Keras 模型
        mlp = MLPClassifier(random_state=seed, solver='lbfgs')
        # 训练模式
        mlp.fit(partial_train_data, partial_train_targets)
        predictions = mlp.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    # K折验证分数平均,没有使用
    average_mae_history = np.mean(all_mae_histories)

    predictions = mlp.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    precision = precision_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None
    recall = recall_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None
    f1 = f1_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None)
    save_model(mlp, "MLPClassifier")
    print("\n")
    print("MLPClassifier accuracy:    %20.16f" % accuracy)
    print("MLPClassifier precision:   %20.16f" % precision)
    print("MLPClassifier recall:      %20.16f" % recall)
    print("MLPClassifier f1 score:    %20.16f" % f1)
    return accuracy, average_mae_history

def save_model(model_temp, model_name):
    dirs = "testModel_2"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    joblib.dump(model_temp, dirs + "/" + model_name)


all_Algorithms()