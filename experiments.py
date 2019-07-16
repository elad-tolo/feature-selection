from sklearn.linear_model import LogisticRegression
from sklearn import svm

import datasetReader
import randomized_greedy
import scoring_functions
from data import Data, Samples
from utils import build_accuracy_graph
import numpy as np
import matplotlib.pyplot as plt

def experiment(dataset_name, scoring_function, feature_selector):
    dataset = datasetReader.read(dataset_name)
    data = Data.build(dataset[0], dataset[1], False, dataset_name)
    model = svm.SVC(kernel='rbf', gamma='auto')
    print(data.train.x.shape[1])
    num_features = int(data.train.x.shape[1])
    accuracy, errors = build_accuracy_graph(model, data, num_features, 100, scoring_function, feature_selector)
    return accuracy, errors

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