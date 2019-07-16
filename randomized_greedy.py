import time

import numpy as np


def get_features(model, data, cardinality, score_function):
    selected_features = []
    selected_features_set = set()
    current_score = 0
    for i in range(cardinality):
        feature_scores = get_feature_scores(model, data, score_function, selected_features, selected_features_set, current_score)
        max_features = np.argpartition(feature_scores, -cardinality)[-cardinality:]
        positive_features = [f for f in max_features if feature_scores[f] > 0]
        skip_probability = 1 - len(positive_features) / cardinality
        if np.random.uniform() > skip_probability:
            feature_index = positive_features[np.random.randint(len(positive_features))]
            selected_features.append(feature_index)
            selected_features_set.add(feature_index)
            current_score = feature_scores[feature_index]

    return selected_features


def get_feature_scores(model, data, score_function, selected_features, selected_features_set, previous_scores):
    num_features = data.train.x.shape[1]
    feature_scores = np.empty(num_features)
    for i in range(num_features):
        if i in selected_features_set:
            feature_scores[i] = -1
        else:
            feature_scores[i] = score_function((model, selected_features, i, data))
    return feature_scores - previous_scores


def forward_selection(model, data, cardinality, score_function):
    selected_features = []
    selected_features_set = set()
    current_score = 0
    for i in range(cardinality):
        feature_scores = get_feature_scores(model, data, score_function, selected_features, selected_features_set, current_score)
        feature_index = np.argmax(feature_scores)
        if feature_scores[feature_index] > 0:
            selected_features.append(feature_index)
            selected_features_set.add(feature_index)
            current_score = feature_scores[feature_index]

    return selected_features