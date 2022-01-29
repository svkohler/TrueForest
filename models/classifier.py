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


def create_embeddings(config, model, tester):

    if os.path.isfile(config.dump_path+'/train_embeddings_'+config.model_name+'_'+str(config.patch_size)+'.pth') == False:
        train_embeddings = tester.test(model, data='train')
        torch.save(train_embeddings, config.dump_path+'/train_embeddings_' +
                   config.model_name+'_'+str(config.patch_size)+'.pth')
        print('train embeddings created')
    else:
        print('train embeddings already exist')

    if os.path.isfile(config.dump_path+'/test_embeddings_'+config.model_name+'_'+str(config.patch_size)+'.pth') == False:
        test_embeddings = tester.test(model, data='test')
        torch.save(test_embeddings, config.dump_path+'/test_embeddings_' +
                   config.model_name+'_'+str(config.patch_size)+'.pth')
        print('test embeddings created')
    else:
        print('test embeddings already exist')


def get_embeddings(config):
    train_embeddings = torch.load(
        config.dump_path+'/train_embeddings_' +
        config.model_name+'_'+str(config.patch_size)+'.pth')
    test_embeddings = torch.load(
        config.dump_path+'/test_embeddings_' +
        config.model_name+'_'+str(config.patch_size)+'.pth')

    return train_embeddings.cpu().detach().numpy(), test_embeddings.cpu().detach().numpy()


def test_mult(config, device, train_data, test_data, num_runs, verbose=0):
    '''
    function to run classification with subsequent testing multiple times
    '''

    clf = get_classifier(config, verbose, device)
    print('classifier used: ', config.clf)

    # check if already some accuracies are stored and continue from there
    if os.path.isfile(config.dump_path + '/'+config.model_name+'_'+str(config.patch_size)+'_test_accuracies_'+config.clf+'.pkl') == True:
        with open(config.dump_path + '/'+config.model_name+'_'+str(config.patch_size)+'_test_accuracies_'+config.clf+'.pkl', 'rb') as data:
            acc_coll = pickle.load(data)
        acc_coll = clean_acc(acc_coll, num_runs)

    else:
        acc_coll = np.zeros(num_runs)

    runs_completed = sum(acc_coll != 0)

    for i in range(runs_completed, num_runs):
        print('run ' + str(i+1) + ' of ' + str(num_runs))
        print('RAM used: ', psutil.virtual_memory()[2])
        print('processing data...')
        train_features, train_labels = process_data(
            train_data, config, mode='train')
        test_features, test_labels = process_data(
            test_data, config, mode='test')
        print('fitting classifier...')
        clf.fit(train_features, train_labels)
        pred_labels = clf.predict(train_features)

        acc = accuracy_score(train_labels, pred_labels)

        print('Training accuracy score of: ', acc)

        print('predicting labels...')
        pred_labels = clf.predict(test_features)
        acc = accuracy_score(test_labels, pred_labels)
        acc_coll[i] = acc
        with open(config.dump_path + '/'+config.model_name+'_'+str(config.patch_size)+'_test_accuracies_'+config.clf+'.pkl', 'wb') as fp:
            pickle.dump(acc_coll, fp)
        print('test accuracy of ' + str(round(acc*100, 2))+'%')
        print('--------------------')
        print('\n')
        del train_features, train_labels, test_features, test_labels

    with open(config.dump_path + '/'+config.model_name+'_'+str(config.patch_size)+'_test_accuracies_'+config.clf+'.pkl', 'wb') as fp:
        pickle.dump(acc_coll, fp)
    print('Finished runs')
    print('average accuracy: ', round(np.mean(acc_coll), 2), '%')
    print('accuracies go from ', round(np.min(acc_coll), 2),
          '% to ', round(np.max(acc_coll), 2), '%')


def classify(config, data):

    # get the classifier
    clf = get_classifier(config, verbose=3)

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


def get_classifier(config, verbose=0, device=None):

    if config.clf == 'linear':
        clf = LogisticRegression(verbose=verbose)

    if config.clf == 'xgboost':
        clf = XGBoost()

    if config.clf == 'random_forest':
        clf = RandomForestClassifier(
            oob_score=True, verbose=verbose, n_jobs=-1)

    if config.clf == 'MLP':
        clf = MLPClassifier(
            max_iter=200, early_stopping=True, verbose=verbose)

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


class LogisticRegression(torch.nn.Module):
    '''
    linear classifier
    '''

    def __init__(self, verbose):
        super(LogisticRegression, self).__init__()
        self.epochs = 100
        self.input_dim = 4096
        self.output_dim = 1
        self.learning_rate = 0.001
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)
        if verbose > 0:
            self.verbose = True
        else:
            self.verbose = False

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

    def fit(self, X_train, y_train):
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)

        iter = 0
        for epoch in tqdm(range(int(self.epochs)), desc='Training Epochs', disable=self.verbose):
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
                    if self.verbose > 1:
                        print(
                            f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")

    def predict(self, X_test):
        with torch.no_grad():
            x = torch.from_numpy(X_test)
            #labels = torch.from_numpy(y_test).type(torch.float)
            outputs = torch.squeeze(self(x)).round().detach().numpy()
            return outputs


class XGBoost():
    '''
    boosting classifier
    '''

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
