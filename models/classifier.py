from sklearn.utils import shuffle
import torch
from tqdm import tqdm
import sys
from sys import getsizeof
import joblib
import psutil

import numpy as np

from utils import *

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

# # Getting % usage of virtual_memory ( 3rd field)
# print('RAM memory % used:', psutil.virtual_memory()[2])


def classify(config, data):

    # get the classifier
    clf = get_classifier(config)

    # process data into positive and negative samples
    print('processing data...')
    features, labels = process_data(data, config)

    # fit the binary classifier
    print('fitting classifier...')
    clf.fit(features, labels)

    pred_labels = clf.predict(features)

    acc = accuracy_score(labels, pred_labels)

    print('Training accuracy score of: ', acc)

    # save the classifer for later testing
    save_clf(clf, config)


def predict(config, data):

    clf = load_clf(config)

    # process data
    features, labels = process_data(data, config)

    pred_labels = clf.predict(features)

    acc = accuracy_score(labels, pred_labels)

    print('Accuracy score of: ', acc)


def get_classifier(config):

    if config.clf == 'linear':
        clf = LogisticRegression()

    if config.clf == 'xgboost':
        clf = XGBoost()

    if config.clf == 'random_forest':
        clf = RandomForestClassifier(oob_score=True, verbose=3, n_jobs=-1)

    if config.clf == 'MLP':
        clf = MLPClassifier(
            max_iter=200, early_stopping=True, verbose=3)

    return clf


def save_clf(classifier, config):

    if config.clf == 'linear':
        torch.save(classifier.state_dict(), config.dump_path +
                   '/' + config.model_name + '_linear_classifier.pth')

    if config.clf == 'xgboost':
        classifier.save_model(
            config.dump_path + '/' + config.model_name + '_xgboost_classifier.json')

    if config.clf == 'random_forest':
        joblib.dump(classifier, config.dump_path + '/' +
                    config.model_name + '_rf_classifier.joblib')

    if config.clf == 'MLP':
        joblib.dump(classifier, config.dump_path + '/' +
                    config.model_name + '_MLP_classifier.joblib')


def load_clf(config):

    if config.clf == 'linear':
        clf = LogisticRegression()
        clf.load_state_dict(torch.load(config.dump_path +
                            '/' + config.model_name + '_linear_classifier.pth'))

    if config.clf == 'xgboost':
        clf = xgb.XGBClassifier()
        clf.load_model(
            config.dump_path + '/' + config.model_name + '_xgboost_classifier.json')

    if config.clf == 'random_forest':
        clf = joblib.load(config.dump_path + '/' +
                          config.model_name + '_rf_classifier.joblib')

    if config.clf == 'MLP':
        clf = joblib.load(config.dump_path + '/' +
                          config.model_name + '_MLP_classifier.joblib')

    return clf


def process_data(data, config):

    # data in the format (samples * 2xfeatures)
    # in addition to positive samples (which are given by definition) create negative ones by randomly
    # combining satellite features (1st half) and drone features (2nd half) of different samples

    pos_labels = np.ones(len(data), dtype=np.int8)

    if config.run_mode == 'test':
        return data, pos_labels

    negatives = None
    for i in range(config.neg_samples_factor):
        data_shuffled = data.copy()
        np.random.shuffle(data_shuffled)
        negative_samples = produce_negative_samples(data_shuffled)
        if negatives is None:
            negatives = negative_samples
        else:
            negatives = np.append(negatives, negative_samples, axis=0)

    neg_labels = np.zeros(len(negatives), dtype=np.int8)

    return np.concatenate((data, negatives), axis=0), np.concatenate((pos_labels, neg_labels), axis=0, dtype=np.int8)


class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.epochs = 100
        self.input_dim = 4096
        self.output_dim = 1
        self.learning_rate = 0.001
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

    def fit(self, X_train, y_train):
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)

        iter = 0
        for epoch in tqdm(range(int(self.epochs)), desc='Training Epochs'):
            x = torch.from_numpy(X_train)
            labels = torch.from_numpy(y_train).type(torch.float)
            self.optimizer.zero_grad()  # Setting our stored gradients equal to zero
            outputs = self(x)
            loss = self.criterion(torch.squeeze(outputs), labels)

            loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias

            self.optimizer.step()  # Updates weights and biases with the optimizer (SGD)

            iter += 1
            if iter % 100 == 0:
                with torch.no_grad():
                    # Calculating the loss and accuracy for the train dataset
                    total = 0
                    correct = 0
                    total += labels.size(0)
                    correct += np.sum(torch.squeeze(outputs).round().detach().numpy()
                                      == labels.detach().numpy())
                    accuracy = 100 * correct/total
                    print(
                        f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")

    def predict(self, X_test):
        with torch.no_grad():
            x = torch.from_numpy(X_test)
            #labels = torch.from_numpy(y_test).type(torch.float)
            outputs = torch.squeeze(self(x)).round().detach().numpy()
            return outputs


class XGBoost():
    def __init__(self):
        super(XGBoost, self).__init__()

    def fit(self, X, y):
        param = {'objective': 'binary:logistic',
                 'tree_method': 'gpu_hist',
                 'gpu_id': 0,
                 'eval_metric': 'logloss'}
        idx = np.random.randint(X.shape[0], size=75000)
        dtrain = xgb.DMatrix(X[idx, :], label=y[idx])
        self.bst = xgb.train(param, dtrain)

    def predict(self, X_test):
        X_test = xgb.DMatrix(X_test)
        pred_proba = self.bst.predict(X_test)
        pred_label = pred_proba > 0.5
        pred_label = pred_label.astype(int)

        return pred_label

    def save_model(self, path):
        self.bst.save_model(path)
