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

import sys
f = open('20.txt', 'a')
sys.stdout = f
sys.stderr = f		# redirect std err, if necessary


def all_Algorithms():
    all_algorithms_score = []
    all_algorithms_score_avg = []
    all_algorithms_name = ["KNN", "DecisionTree", "MLPClassifier", "NaiveBayes", "SVM", "RandomForestClassifier",
                           "Bagging"]

    # load data
    dataset_train = load_data_set_train()
    array = dataset_train.values
    x = array[:, 1:len(list(dataset_train)) - 1]
    y = array[:, 0]
    validation_size = 0.20
    seed = 7
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=validation_size,
                                                                                    random_state=seed)
    accuracy, average_mae_history = knn_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    all_algorithms_score_avg.append(average_mae_history)

    accuracy, average_mae_history = DecisionTree_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    all_algorithms_score_avg.append(average_mae_history)

    accuracy, average_mae_history = MLPClassifier_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    all_algorithms_score_avg.append(average_mae_history)

    accuracy, average_mae_history = NaiveBayes_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    all_algorithms_score_avg.append(average_mae_history)

    accuracy, average_mae_history = SVM_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    all_algorithms_score_avg.append(average_mae_history)

    accuracy, average_mae_history = RandomForestClassifier_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    all_algorithms_score_avg.append(average_mae_history)

    accuracy, average_mae_history = Bagging_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    all_algorithms_score_avg.append(average_mae_history)

    return all_algorithms_name, all_algorithms_score, all_algorithms_score_avg, list(dataset_train), list(
        dataset_train.mean())


def knn_Algorithms(x_train, x_validation, y_train, y_validation, max_iter=100000):
    # knn Algorithms
    best_k, max_value = choose_best_k_to_knn(x_train, y_train, x_validation, y_validation)
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    precision = precision_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None
    recall = recall_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None
    f1 = f1_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None)
    save_model(knn, "KNN")
    print("KNN accuracy:    %20.16f" % accuracy)
    print("KNN precision:   %20.16f" % precision)
    print("KNN recall:      %20.16f" % recall)
    print("KNN f1 score:    %20.16f" % f1)
    return accuracy, max_value


def DecisionTree_Algorithms(x_train, x_validation, y_train, y_validation):
    # DecisionTree Algorithms
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
        dtc = DecisionTreeClassifier()
        # 训练模式
        dtc.fit(partial_train_data, partial_train_targets)

        predictions = dtc.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    # K折验证分数平均,没有使用
    average_mae_history = np.mean(all_mae_histories)

    predictions = dtc.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    precision = precision_score(y_validation, predictions, average="macro") #macro, micro, weighted, None
    recall = recall_score(y_validation, predictions, average="macro") #macro, micro, weighted, None
    f1 = f1_score(y_validation, predictions, average="macro") #macro, micro, weighted, None)
    save_model(dtc, "DecisionTree")
    print("\n")
    print("DecisionTree accuracy:     %20.16f" % accuracy)
    print("DecisionTree precision:    %20.16f" % precision)
    print("DecisionTree recall:       %20.16f" % recall)
    print("DecisionTree f1 score:     %20.16f" % f1)
    return accuracy, average_mae_history


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


def NaiveBayes_Algorithms(x_train, x_validation, y_train, y_validation):
    # Naive Bayes Algorithms
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
        nb = GaussianNB()
        # 训练模式
        nb.fit(partial_train_data, partial_train_targets)
        predictions = nb.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    # K折验证分数平均,没有使用
    average_mae_history = np.mean(all_mae_histories)

    predictions = nb.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    precision = precision_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None
    recall = recall_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None
    f1 = f1_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None)
    save_model(nb, "NaiveBayes")
    print("\n")
    print("NaiveBayes accuracy:  %20.16f" % accuracy)
    print("NaiveBayes precision: %20.16f" % precision)
    print("NaiveBayes recall:    %20.16f" % recall)
    print("NaiveBayes f1 score:  %20.16f" % f1)
    return accuracy, average_mae_history


def SVM_Algorithms(x_train, x_validation, y_train, y_validation):
    # SVM Algorithms
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
        svc = SVC(kernel='poly', gamma="auto")
        # 训练模式
        svc.fit(partial_train_data, partial_train_targets)
        predictions = svc.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    # K折验证分数平均,没有使用
    average_mae_history = np.mean(all_mae_histories)

    predictions = svc.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    precision = precision_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None
    recall = recall_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None
    f1 = f1_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None)
    save_model(svc, "SVM")
    print("\n")
    print("SVM accuracy:  %20.16f" % accuracy)
    print("SVM precision: %20.16f" % precision)
    print("SVM recall:    %20.16f" % recall)
    print("SVM f1 score:  %20.16f" % f1)
    return accuracy, average_mae_history


def RandomForestClassifier_Algorithms(x_train, x_validation, y_train, y_validation):
    # SVM Algorithms
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
        rf = RandomForestClassifier(n_estimators=10, max_depth=10)
        # 训练模式
        rf.fit(partial_train_data, partial_train_targets)
        predictions = rf.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    # K折验证分数平均,没有使用
    average_mae_history = np.mean(all_mae_histories)

    predictions = rf.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    precision = precision_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None
    recall = recall_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None
    f1 = f1_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None)
    save_model(rf, "RandomForestClassifier")
    print("\n")
    print("RandomForestClassifier accuracy:  %20.16f" % accuracy)
    print("RandomForestClassifier precision: %20.16f" % precision)
    print("RandomForestClassifier recall:    %20.16f" % recall)
    print("RandomForestClassifier f1 score:  %20.16f" % f1)
    return accuracy, average_mae_history


def Bagging_Algorithms(x_train, x_validation, y_train, y_validation):
    # Naive Bayes Algorithms
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
        clfb = BaggingClassifier(base_estimator=DecisionTreeClassifier()
                                 , max_samples=0.5, max_features=0.5)
        # 训练模式
        clfb.fit(partial_train_data, partial_train_targets)
        predictions = clfb.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    # K折验证分数平均,没有使用
    average_mae_history = np.mean(all_mae_histories)

    predictions = clfb.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    precision = precision_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None
    recall = recall_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None
    f1 = f1_score(y_validation, predictions, average="macro")  # macro, micro, weighted, None)
    save_model(clfb, "Bagging")
    print("\n")
    print("Bagging accuracy:  %20.16f" % accuracy)
    print("Bagging precision: %20.16f" % precision)
    print("Bagging recall:    %20.16f" % recall)
    print("Bagging f1 score:  %20.16f" % f1)
    return accuracy, average_mae_history


def choose_best_k_to_knn(x_train, y_train, x_validation, y_validation):
    all_mae_histories = []
    accuracy = 0
    k = 1
    for i in range(1, 30):
        average_mae_history = K_vertify_knn(x_train, y_train, i)
        all_mae_histories.append(average_mae_history)
    index = -1
    max_value = -999

    # 可视化画图部分
    # plt.plot(range(1, len(all_mae_histories) + 1), all_mae_histories)
    # plt.xlabel('k value')
    # plt.ylabel('grade')
    # plt.show()

    for i, val in enumerate(all_mae_histories):
        if max_value < val:
            index = i
            max_value = val
    return index, max_value


def K_vertify_knn(train_data, train_targets, knnnumber):
    # K折验证，适用于数据集较少的数据集
    all_mae_histories = []
    k = 10
    num_val_samples = len(train_data) // k
    for i in range(k):
        # 准备验证数据，第K个分区的数据
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        # 准备训练数据，其他所有分区的数据
        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)
        # 构建 Keras 模型
        knn = KNeighborsClassifier(n_neighbors=knnnumber)
        # 训练模式
        knn.fit(partial_train_data, partial_train_targets)
        predictions = knn.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    # K折验证分数平均
    average_mae_history = np.mean(all_mae_histories)
    return average_mae_history


def save_model(model_temp, model_name):
    dirs = "testModel"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    joblib.dump(model_temp, dirs + "/" + model_name)


all_Algorithms()