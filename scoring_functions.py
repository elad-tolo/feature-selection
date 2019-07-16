import numpy as np
from sklearn.feature_selection import mutual_info_regression

import utils


def wrapper(args):
    model, selected_features, candidate, data = args
    features = selected_features + [candidate]
    return utils.model_accuracy(model, features, data)


def mutual_info(args):
    model, selected_features, candidate, data = args
    features = selected_features + [candidate]
    return mutual_info_regression(data.train.x[:, features], data.train.y)



def additional_mutual_information(args):
    _, selected_features, candidate, data = args
    mi = mutual_information(data.x_prob[candidate], data.y_prob, data.xy_prob[candidate])
    cmi = conditional_mutual_information(candidate, selected_features, data)
    return mi - cmi


# assume the input is flattened: flat = np.ndarray.flatten(not_flat)
def mutual_information(model_prob, label_prob, model_label_prob):
    denum = np.ndarray.flatten(np.outer(label_prob, model_prob))
    log = np.log(np.divide(model_label_prob, denum))
    return model_label_prob.dot(log)


def conditional_mutual_information(candidate, model_features, data):
    cmi = 0
    for f in model_features:
        p_xyz = data.xyz_prob[candidate][f]
        p_z = data.x_prob[f]
        numer = np.multiply(p_xyz, p_z)

        denum = np.einsum('ik,kj->ijk', data.xz_prob[candidate][f], data.xy_prob[f])
        log = np.log(np.divide(numer, denum))
        cmi += np.einsum('ijk,ijk', log, p_xyz)
    return cmi
