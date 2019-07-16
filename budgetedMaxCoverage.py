import itertools
import numpy as np


def get_features(data, threshold):
    num_features = data.train.x.shape[1]
    base_triplets = list(itertools.combinations(range(num_features), 3))
    for triplet in base_triplets:
        triplet_size = get_triplet_size(triplet)
        if triplet_size > threshold:
            continue
        score = get_feature_score(triplet)


def greedy_selection(data, threshold, initial_model):
    selected_features = initial_model
    selected_features_set = set(selected_features)
    while get_model_size(selected_features) < threshold:
        feature_scores = get_feature_scores(_, data, mutual_information, selected_features, selected_features_set)
        best_feature = np.argmax(feature_scores)
        selected_features.append(best_feature)
        selected_features_set.add(best_feature)

    selected_features = []
    selected_features_set = set()
    for i in range(cardinality):
        start = int(round(time.time() * 1000))
        feature_scores = get_feature_scores(model, data, score_function, selected_features, selected_features_set)
        end = int(round(time.time() * 1000))
        print("iteration time: " + str(end - start))
        max_features = np.argpartition(feature_scores, -cardinality)[-cardinality:]
        positive_features = [f for f in max_features if feature_scores[f] > 0]
        skip_probability = 1 - len(positive_features) / cardinality
        if np.random.uniform() > skip_probability:
            feature_index = np.random.randint(len(positive_features))
            selected_features.append(positive_features[feature_index])
            selected_features_set.add(positive_features[feature_index])

    return selected_features
