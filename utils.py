import os
import time

import numpy as np
from sklearn.model_selection import KFold

import randomized_greedy
from data import Data, Samples


def model_accuracy(model, features, data):
    model.fit(data.train.x[:, features], data.train.y)  # TODO - make sure fit resets weights and not transfer learning
    return model.score(data.test.x[:, features], data.test.y)
    # prediction = model.predict(data.test.x[:, features])
    # return calc_accuracy(prediction, data.test.y)


def calc_accuracy(prediction, labels_test):  # TODO change impl!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    diff = np.abs(prediction - labels_test)
    accuracy = 1 - sum(diff) / len(diff)
    return accuracy


# #######################################################################################


def calculate_accuracy(model, train_test_data, train_validation_data, cardinality, score_function, feature_selector):
    # run sampleGreedy(k = cardinality)
    return feature_selector(model, train_test_data, train_validation_data, cardinality, score_function)


def build_accuracy_graph(model, train_test_data, max_cardinality, num_experiments, score_function, feature_selector):
    accuracy = np.zeros(max_cardinality)

    kf = KFold(n_splits=num_experiments, shuffle=True)
    for train_index, test_index in kf.split(train_test_data.train.x):
        train_validation_data = Data(
            Samples(train_test_data.train.x[train_index], train_test_data.train.y[train_index]),
            Samples(train_test_data.train.x[test_index], train_test_data.train.y[test_index]),
            False,
            train_test_data.name)
        for c in range(1, max_cardinality + 1):
            accuracy[c - 1] += calculate_accuracy(model, train_test_data, train_validation_data,
                                                  c, score_function, feature_selector)

    return accuracy / num_experiments

    # if os._exists(data.name):
    #     os.remove(data.name)
    # f = open(data.name, "a+")
    # f.write(str(accuracy[c - 1]) + '\n')
    # f.close()
