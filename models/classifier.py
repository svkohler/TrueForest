import sys
import joblib

import numpy as np

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


def classify(config, data):

    # get the classifier
    clf = get_classifier(config)

    # process data into positive and negative samples
    features, labels = process_data(data, config)

    # fit the binary classifier
    clf.fit(features, labels)

    # save the classifer for later testing
    save_clf(clf, config)


def predict(config, data):
    pass


def get_classifier(config):

    if config.clf == 'xgboost':
        clf = xgb.XGBClassifier()

    if config.clf == 'random_forest':
        clf = RandomForestClassifier()

    return clf


def save_clf(classifier, config):

    if config.clf == 'xgboost':
        classifier.save_model(
            config.dump_path + '/' + config.model_name + '_xgboost_classifier.json')

    if config.clf == 'random_forest':
        joblib.dump(classifier, config.dump_path + '/' +
                    config.model_name + '_rf_classifier.joblib')


def process_data(data, config):

    # data in the format (samples * 2xfeatures)
    # in addition to positive samples (which are given by definition) create negative ones by randomly
    # combining satellite features (1st half) and drone features (2nd half) of different samples

    pos_labels = np.ones(len(data))

    negatives = None
    for i in range(config.neg_samples_factor):
        data_shuffled = data.copy()
        np.random.shuffle(data_shuffled)
        negative_samples = produce_negative_samples(data_shuffled)
        if negatives is None:
            negatives = negative_samples
        else:
            negatives = np.append(negatives, negative_samples, axis=0)

    neg_labels = np.zeros(len(negatives))

    return np.concatenate((data, negatives), axis=0), np.concatenate((pos_labels, neg_labels), axis=0)


def produce_negative_samples(data):

    data_copy = data.copy()
    data_copy = data_copy[1:, :]
    data_copy = np.append(data_copy, [data[0, :]], axis=0)

    return np.concatenate((data[:, :int(data.shape[1]/2)], data_copy[:, int(data_copy.shape[1]/2):]), axis=1)
