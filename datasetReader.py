import networkx as nx
import numpy as np
import csv
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from pyitlib import discrete_random_variable as drv

# Format = "one-file", "train-test", "features-label", "train-test-label"
from sklearn import svm
from sklearn.preprocessing import normalize
import time


def read(datasetName, shouldNormalize=False):
    if datasetName == 'glass':
        data = genfromtxt('data/glass/glass.data', delimiter=',')
        x = data[:, :-1]
        y = data[:, -1]
        return x, y
    if datasetName == "yale":
        data = loadmat("data/yale/Yale_32x32.mat")
        x = data['fea']
        x = normalize(x)
        y = data['gnd'].ravel()
        return x, y
    if datasetName == "PCMAC" or datasetName == 'lung':
        data = loadmat("data/"+datasetName+"/"+datasetName+".mat")
        x = data['X']
        y = data['Y'].ravel()
        return x, y
    if datasetName == 'SPECT' or \
            datasetName == 'Sonar' or \
            datasetName == 'Statlog' or \
            datasetName == 'Yaron' or \
            datasetName == 'testKL':
        form = 'one-file'
    elif datasetName == 'PCMAC' or \
            datasetName == 'BASEHOCK' or \
            datasetName == 'RELATHE' or \
            datasetName == 'GISETTE' or \
            datasetName == 'MADELON':
        form = 'features-label'
    else:
        raise RuntimeError("dataset not supported")

    if datasetName == 'PCMAC' or \
            datasetName == 'BASEHOCK' or \
            datasetName == 'RELATHE' or \
            datasetName == 'SPECT' or \
            datasetName == 'Sonar' or \
            datasetName == 'Yaron' or \
            datasetName == 'testKL':
        delim = ','
    elif datasetName == 'GISETTE' or \
            datasetName == 'MADELON' or \
            datasetName == 'Statlog':
        delim = ' '

    if form == 'features-label':
        with open('datasets/' + datasetName + '/features.txt') as fFile:
            reader = csv.reader(fFile, delimiter=delim)
            samples = np.array([list(map(float, row[:-1])) for row in reader])

        with open('datasets/' + datasetName + '/label.txt') as lFile:
            reader = csv.reader(lFile)
            labels = np.array([float(float(row[0]) == 1.0) for row in reader])

    if form == "one-file":
        if datasetName == 'SPECT':
            labelPos = 'first'
        elif datasetName == 'Statlog' or \
                datasetName == 'Sonar' or \
                datasetName == 'Yaron' or \
                datasetName == 'testKL':
            labelPos = 'last'

        if datasetName == 'SPECT' or \
                datasetName == 'Statlog' or \
                datasetName == 'Yaron' or \
                datasetName == 'testKL':
            truth = '1'
        elif datasetName == 'Sonar':
            truth = 'R'

        with open('datasets/' + datasetName + '/data.txt') as fFile:
            reader = csv.reader(fFile, delimiter=delim)
            data = np.array([row for row in reader])

            if labelPos == 'last':
                samples = np.array([list(map(float, d[:-1])) for d in data])
                labels = [float(d == truth) for d in data[:, -1]]
            elif labelPos == 'first':
                samples = np.array([list(map(float, d[1:])) for d in data])
                labels = [float(d == truth) for d in data[:, 0]]

    if shouldNormalize:
        samples = normalizeSamples(samples)

    return samples, labels


def normalizeSamples(samples):
    minValues = samples.min(axis=0)
    samples = samples - minValues

    maxValues = samples.max(axis=0)
    samples = samples / maxValues

    samples = np.floor(samples * 10.0) / 10.0

    return samples


def getGlassData():
    data = genfromtxt('data/glass/glass.data', delimiter=',')
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


if __name__ == '__main__':
    x, y = read("lung")
    num_features = x.shape[1]
    # num_features = 100
    G = nx.Graph()
    f = np.zeros((num_features, num_features))
    for i in range(num_features):
        start = time.time()
        for j in range(num_features):
            f[i, j] = drv.information_mutual_conditional(x[:, i], y, x[:, j]) + drv.information_mutual_conditional(x[:, j], y, x[:, i])
            G.add_edge(i, j, weight=f[i, j])
        end = time.time()
        print(end-start)

    apsp = nx.floyd_warshall(G)
    alpha = 1
    for i in range(num_features):
        for j in range(num_features):
            if f[i,j] == 0 or i == j:
                continue
            alpha = min(alpha, apsp[i][j] / f[i, j])
            if alpha == 0:
                print("ASD")
    print(alpha)






    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    rbf_svc = svm.SVC(kernel='rbf')
    rbf_svc.fit(x_train, y_train)
    pred = rbf_svc.predict(x_test)
    print(1 - sum(pred != y_test) / len(y_test))
    print(rbf_svc.score(x_test, y_test))