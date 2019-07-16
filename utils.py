import os
import time

import numpy as np

import randomized_greedy
from data import Data


def model_accuracy(model, features, data):
    model.fit(data.train.x[:, features], data.train.y)
    return model.score(data.test.x[:, features], data.test.y)
    # prediction = model.predict(data.test.x[:, features])
    # return calc_accuracy(prediction, data.test.y)


def calc_accuracy(prediction, labels_test):  # TODO change impl!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    diff = np.abs(prediction - labels_test)
    accuracy = 1 - sum(diff) / len(diff)
    return accuracy


# #######################################################################################


def calculate_average_accuracy(model, data, cardinality, num_experiments, score_function, feature_selector):
    results = np.empty(num_experiments)
    for i in range(num_experiments):
        samplesData = Data.build(data.train.x, data.train.y, False, data.name)
        start = int(round(time.time() * 1000))
        # run sampleGreedy(k = cardinality)
        selected_features = feature_selector(model, samplesData, cardinality, score_function)
        end = int(round(time.time() * 1000))
        results[i] = model_accuracy(model, selected_features, data)
    return np.average(results), np.std(results)


def build_accuracy_graph(model, data, max_cardinality, num_experiments, score_function, feature_selector):
    accuracy = np.empty(max_cardinality)
    errors = np.empty(max_cardinality)
    if os._exists(data.name):
        os.remove(data.name)
    for c in range(1, max_cardinality + 1):
        start = int(round(time.time() * 1000))
        accuracy[c - 1], errors[c - 1] = calculate_average_accuracy(model, data, c, num_experiments, score_function, feature_selector)
        end = int(round(time.time() * 1000))
        f = open(data.name, "a+")
        f.write(str(accuracy[c - 1]) + '\n')
        f.close()
    return accuracy, errors
