from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split


class Samples:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Data:
    def __init__(self, train, test, build_probs, name):
        self.train = train
        self.test = test
        self.name = name
        if build_probs:
            self.x_prob = self.__get_feature_probability()
            self.y_prob = self.__get_label_probability()
            self.xy_prob = self.__get_joint_probability()
            self.xz_prob = nestedDictionary(2)
            self.xyz_prob = nestedDictionary(2)

    @staticmethod
    def build(samples, labels, build_probs, name):
        samplesTrain, samplesTest, labelsTrain, labelsTest = train_test_split(samples, labels, test_size=0.2)
        train = Samples(samplesTrain, labelsTrain)
        test = Samples(samplesTest, labelsTest)
        samples = Data(train, test, build_probs, name)
        return samples

    def get_xyz_prob(self, x, z):
        if x in self.xyz_prob:
            if z in self.xyz_prob[x]:
                return self.xyz_prob[x][z]
        elif z in self.xyz_prob:
            if x in self.xyz_prob[z]:
                self.xyz_prob[x][z] = np.transpose(self.xyz_prob[z][x])
                return self.xyz_prob[x][z]
        else:
            self.xyz_prob[x][z] = self.__calculate_xyz_density(x, z)
            return self.xyz_prob[x][z]

    def get_xz_prob(self, x, z):
        if x in self.xz_prob:
            if z in self.xz_prob[x]:
                return self.xz_prob[x][z]
        elif z in self.xz_prob:
            if x in self.xz_prob[z]:
                self.xz_prob[x][z] = np.transpose(self.xz_prob[z][x])
                return self.xz_prob[x][z]
        else:
            self.xz_prob[x][z] = self.__calculate_xz_density(x, z)
            return self.xz_prob[x][z]

    def get_xy_prob(self, x):
        return self.xy_prob[x]

    def __calculate_xyz_density(self, x, z):
        xyz_values = self.__get_xyz_frequency_map(x, z)

        x_value_to_index = self.__value_to_index(self.train.x[:, x])
        y_value_to_index = self.__value_to_index(self.train.y)
        z_value_to_index = self.__value_to_index(self.train.x[:, z])

        xyz_prob = np.zeros((len(x_value_to_index), len(y_value_to_index), len(z_value_to_index)))
        for x_value in xyz_values.keys():
            for y_value in xyz_values[x_value].keys():
                for z_value in xyz_values[x_value][y_value].keys():
                    xyz_prob[x_value_to_index[x_value]][y_value_to_index[y_value]][z_value_to_index[z_value]] = \
                        xyz_values[x_value][y_value][z_value]
        xyz_prob /= self.train.x.shape[0]
        return xyz_prob

    def __get_xyz_frequency_map(self, x, z):
        xyz_values = nestedDictionary(3)
        for sample, label in zip(self.train.x, self.train.y):
            z_value = sample[z]
            x_value = sample[x]
            if x_value in xyz_values and label in xyz_values[x_value] and z_value in xyz_values[x_value][label]:
                xyz_values[x_value][label][z_value] += 1
            else:
                xyz_values[x_value][label][z_value] = 1
        return xyz_values

    def __get_xz_frequency_map(self, x, z):
        xz_values = nestedDictionary(2)
        for sample in self.train.x:
            z_value = sample[z]
            x_value = sample[x]
            if x_value in xz_values and z_value in xz_values[x_value]:
                xz_values[x_value][z_value] += 1
            else:
                xz_values[x_value][z_value] = 1
        return xz_values

    def __get_x_frequency_map(self, x):
        x_values = nestedDictionary(1)
        for sample in self.train.x:
            x_value = sample[x]
            if x_value in x_values:
                x_values[x_value] += 1
            else:
                x_values[x_value] = 1
        return x_values

    def __value_to_index(self, values):
        values = sorted(list(set(values)))
        return {v: i for i, v in dict(enumerate(values)).items()}

    def __calculate_xz_density(self, x, z):
        xz_values = self.__get_xz_frequency_map(x, z)

        x_value_to_index = self.__value_to_index(self.train.x[:, x])
        z_value_to_index = self.__value_to_index(self.train.x[:, z])

        xz_prob = np.zeros((len(x_value_to_index), len(z_value_to_index)))
        for x_value in xz_values.keys():
            for z_value in xz_values[x_value].keys():
                xz_prob[x_value_to_index[x_value]][z_value_to_index[z_value]] = \
                    xz_values[x_value][z_value]
        xz_prob /= self.train.x.shape[0]
        return xz_prob

    def __get_feature_probability(self):
        return [self.__calculate_x_density(x) for x in range(self.train.x.shape[1])]

    def __calculate_x_density(self, x):
        x_values = self.__get_x_frequency_map(x)
        x_value_to_index = self.__value_to_index(self.train.x[:, x])
        x_prob = np.zeros((len(x_value_to_index)))
        for x_value in x_values.keys():
            x_prob[x_value_to_index[x_value]] = x_values[x_value]
        x_prob /= self.train.x.shape[0]
        return x_prob

    def __get_label_probability(self):
        return self.__calculate_y_density()

    def __calculate_y_density(self):
        y_values = self.__get_y_frequency_map()
        y_value_to_index = self.__value_to_index(self.train.y)
        y_prob = np.zeros((len(y_value_to_index)))
        for y_value in y_values.keys():
            y_prob[y_value_to_index[y_value]] = y_values[y_value]
        y_prob /= len(self.train.y)
        return y_prob

    def __get_y_frequency_map(self):
        y_values = nestedDictionary(1)
        for label in self.train.y:
            if label in y_values:
                y_values[label] += 1
            else:
                y_values[label] = 1
        return y_values

    def __get_joint_probability(self):
        return [self.__calculate_xy_density(x) for x in range(self.train.x.shape[1])]

    def __calculate_xy_density(self, x):
        xy_values = self.__get_xy_frequency_map(x)
        x_value_to_index = self.__value_to_index(self.train.x[:, x])
        y_value_to_index = self.__value_to_index(self.train.y)
        xy_prob = np.zeros((len(x_value_to_index), len(y_value_to_index)))
        for x_value in xy_values.keys():
            for y_value in xy_values[x_value].keys():
                xy_prob[x_value_to_index[x_value]][y_value_to_index[y_value]] = xy_values[x_value][y_value]
        xy_prob /= self.train.x.shape[0]
        return xy_prob

    def __get_xy_frequency_map(self, x):
        xy_values = nestedDictionary(2)
        for sample, label in zip(self.train.x, self.train.y):
            x_value = sample[x]
            if x_value in xy_values and label in xy_values[x_value]:
                xy_values[x_value][label] += 1
            else:
                xy_values[x_value][label] = 1
        return xy_values


def nestedDictionary(depth):
    if depth == 1:
        return defaultdict(np.array)
    else:
        return defaultdict(lambda: nestedDictionary(depth - 1))
