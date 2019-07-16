from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import KFold

import datasetReader
import randomized_greedy
import scoring_functions
from data import Data, Samples
from utils import build_accuracy_graph
import numpy as np
import matplotlib.pyplot as plt

def experiment(dataset_name, scoring_function, feature_selector):
    dataset = datasetReader.read(dataset_name)
    num_experiments = 5

    max_cardinality = int(dataset[0].shape[1])
    accuracy = np.zeros((num_experiments, max_cardinality))
    kf = KFold(n_splits=num_experiments, shuffle=True)
    i = 0
    for train_index, test_index in kf.split(dataset[0]):
        data = Data(Samples(dataset[0][train_index], dataset[1][train_index]),
                    Samples(dataset[0][test_index], dataset[1][test_index]),
                    False,
                    dataset_name)
        model = svm.SVC(kernel='rbf', gamma='auto')
        accuracy[i] = build_accuracy_graph(model, data, max_cardinality, num_experiments, scoring_function, feature_selector)
        i = i + 1
    return np.average(accuracy, axis=0), np.std(accuracy, axis=0) #TODO change this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if __name__ == '__main__':
   rand_greedy, rand_greedy_errors = experiment("glass", scoring_functions.wrapper, randomized_greedy.get_features)
   forward, forward_errors = experiment("glass", scoring_functions.wrapper, randomized_greedy.forward_selection)

   fig = plt.figure()
   yerr = np.linspace(0.05, 0.06, 10)

   plt.errorbar(range(1, len(rand_greedy) + 1), rand_greedy, yerr=rand_greedy_errors, label='rand')
   plt.errorbar(range(1, len(forward) + 1), forward, yerr=forward_errors, label='forward')
   plt.legend()
   plt.show()

   print("rand: " + str(rand_greedy))
   print("forward: " + str(forward))