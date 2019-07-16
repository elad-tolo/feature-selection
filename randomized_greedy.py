import time

import numpy as np

from utils import model_accuracy


def get_features(model, train_test_data, train_validation_data, max_cardinality, score_function):
    repetitions = 10
    max_score = 0
    max_features = []

    for r in range(repetitions):
        selected_features = []
        selected_features_set = set()
        current_score = 0
        for i in range(max_cardinality):
            feature_scores = get_feature_scores(model, train_validation_data, score_function, selected_features, selected_features_set, current_score)
            max_features = np.argpartition(feature_scores, -max_cardinality)[-max_cardinality:]
            positive_features = [f for f in max_features if feature_scores[f] >= 0]
            skip_probability = 1 - len(positive_features) / max_cardinality
            if np.random.uniform() > skip_probability:
                feature_index = positive_features[np.random.randint(len(positive_features))]
                selected_features.append(feature_index)
                selected_features_set.add(feature_index)
                current_score = feature_scores[feature_index]
        score = model_accuracy(model, selected_features, train_validation_data)
        if max_score > score:
            max_features = list(selected_features)
            max_score = score
    return model_accuracy(model, max_features, train_test_data)


def get_feature_scores(model, data, score_function, selected_features, selected_features_set, previous_score):
    num_features = data.train.x.shape[1]
    feature_scores = np.empty(num_features)
    for i in range(num_features):
        if i in selected_features_set:
            feature_scores[i] = -1
        else:
            feature_scores[i] = score_function((model, selected_features, i, data))
    return feature_scores - previous_score


def forward_selection(model, train_test_data, train_validation_data, cardinality, score_function):
    selected_features = []
    selected_features_set = set()
    current_score = 0
    for i in range(cardinality):
        feature_scores = get_feature_scores(model, train_validation_data, score_function, selected_features, selected_features_set, current_score)
        feature_index = np.argmax(feature_scores)
        # if feature_scores[feature_index] > 0:
        selected_features.append(feature_index)
        selected_features_set.add(feature_index)
        current_score = feature_scores[feature_index]

    return model_accuracy(model, selected_features, train_test_data)
