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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
from torch import nn
import torch.nn.functional as F


def create_embeddings(config, model, tester):
    '''
    function to create train/test embeddings for different locations from pretrained models

    '''

    if not os.path.exists(config.dump_path+'/embeddings'):
        os.mkdir(config.dump_path+'/embeddings')

    if os.path.isfile(config.dump_path+'/embeddings/train_embeddings_'+config.model_name+'_'+config.location[0]+'_'+str(config.patch_size)+'.pth') == False:
        # call tester to create embeddings
        train_embeddings = tester.test(model, data='train')
        torch.save(train_embeddings, config.dump_path+'/embeddings/train_embeddings_' +
                   config.model_name+'_'+config.location[0]+'_'+str(config.patch_size)+'.pth')
        print('train embeddings created')
    else:
        print('train embeddings already exist')

    for loc in config.location[1:]:
        if os.path.isfile(config.dump_path+'/embeddings/test_embeddings_'+config.model_name+'_'+loc+'_'+str(config.patch_size)+'.pth') == False:
            # call tester to create embeddings
            test_embeddings = tester.test(model, data='test', location=loc)
            torch.save(test_embeddings, config.dump_path+'/embeddings/test_embeddings_' +
                       config.model_name+'_'+loc+'_'+str(config.patch_size)+'.pth')
            print(f'{loc}: test embeddings created')
        else:
            print(f'{loc}: test embeddings already exist')


def get_train_embeddings(config):
    '''
    function to get specifically train embeddings and load them to CPU and convert to numpy array

    '''
    train_embeddings = torch.load(
        config.dump_path+'/embeddings/train_embeddings_' +
        config.model_name+'_'+config.location[0]+'_'+str(config.patch_size)+'.pth', map_location=f'cuda:{config.gpu_ids[0]}')

    return train_embeddings.cpu().detach().numpy()


def get_test_embeddings(config):
    '''
    function to get specifically test embeddings and load them to CPU and convert to numpy array.
    test embeddings are organised in a dictionary.

    '''
    test_embeddings = {}
    for loc in config.location[1:]:
        test_embeddings[loc] = torch.load(
            config.dump_path+'/embeddings/test_embeddings_' +
            config.model_name+'_'+loc+'_'+str(config.patch_size)+'.pth', map_location=f'cuda:{config.gpu_ids[0]}').cpu().detach().numpy()

    return test_embeddings


def test_mult(config, device, train_data, test_data, num_runs, verbose=0):
    '''
    main testing function.
    run training/testing loop of different classifiers multiple times and collect accuracies on different locations.

    '''

    if not os.path.exists(config.dump_path+'/accuracies'):
        os.mkdir(config.dump_path+'/accuracies')

    clf = get_classifier(config, verbose, device)
    print('classifier used: ', config.clf)

    # check if already some accuracies are stored and continue from there
    if os.path.isfile(config.dump_path + '/accuracies/'+config.model_name+'_'+str(config.patch_size)+'_test_accuracies_'+config.clf+'.pkl') == True:
        with open(config.dump_path + '/accuracies/'+config.model_name+'_'+str(config.patch_size)+'_test_accuracies_'+config.clf+'.pkl', 'rb') as data:
            acc_coll = pickle.load(data)
    else:
        acc_coll = AccuracyCollector(num_runs=config.num_runs)

    # flag from where to continue
    runs_completed = acc_coll.runs

    for i in range(runs_completed, num_runs):
        # training

        # flag to check if classifier is converged
        converged = False
        while not converged:
            print('run ' + str(i+1) + ' of ' + str(num_runs))
            print('RAM used: ', psutil.virtual_memory()[2])
            print('processing data...')
            train_features, train_labels = process_data(
                train_data, config, mode='train')
            print('fitting classifier...')
            clf.fit(train_features, train_labels)
            pred_labels = clf.predict(train_features)

            acc = accuracy_score(train_labels, pred_labels)
            # check if classifier has converged
            if acc != 0.5:
                converged = True
            tn, fp, fn, tp = confusion_matrix(
                train_labels, pred_labels).ravel()

        print(
            f'Training results: Accuracy: {acc:.4f}% \t Pos. Precision: {tp/(tp+fp):.4f}% \t Pos. Recall: {tp/(tp+fn):.4f}% \t Neg. Precision: {tn/(tn+fn):.4f}% \t Neg. Recall: {tn/(tn+fp):.4f}%')

        # testing
        print('predicting labels...')
        for loc in test_data.keys():
            test_features, test_labels = process_data(
                test_data[loc], config, mode='test')

            pred_labels = clf.predict(test_features)
            acc = accuracy_score(test_labels, pred_labels)
            # save classification behaviour for later analysis
            tn, fp, fn, tp = confusion_matrix(
                test_labels, pred_labels).ravel()
            acc_coll.update(loc, (acc, tn, fp, fn, tp), i)
            print(
                f'{loc} Testing results: Accuracy: {acc:.4f}% \t Pos. Precision: {tp/(tp+fp):.4f}% \t Pos. Recall: {tp/(tp+fn):.4f}% \t Neg. Precision: {tn/(tn+fn):.4f}% \t Neg. Recall: {tn/(tn+fp):.4f}%')

        acc_coll.update_runs()
        with open(config.dump_path + '/accuracies/'+config.model_name+'_'+str(config.patch_size)+'_test_accuracies_'+config.clf+'.pkl', 'wb') as fp:
            pickle.dump(acc_coll, fp)
        print('--------------------')
        print('\n')
        del train_features, train_labels, test_features, test_labels

    print('Finished runs')
    acc_coll.end_statement()


def classify(config, data, device):
    '''
    function to train and save a single classifer. Used for validation application.

    '''

    # get the classifier
    clf = get_classifier(config, verbose=0, device=device)

    # process data into positive and negative samples
    print('processing data...')
    features, labels = process_data(data, config, mode='test')

    # fit the binary classifier
    print('fitting classifier...')
    clf.fit(features, labels)

    pred_labels = clf.predict(features)
    print(pred_labels)

    acc = accuracy_score(labels, pred_labels)

    print('Training accuracy score of: ', acc)

    # save the classifer for later testing
    save_clf(clf, config)


def get_classifier(config, verbose=0, device=None):
    '''
    helper function to retrieve classifier

    '''

    if config.clf == 'linear':
        clf = LogisticRegression(verbose=verbose)

    if config.clf == 'xgboost':
        clf = XGBoost(config)

    if config.clf == 'random_forest':
        clf = RandomForestClassifier(
            oob_score=True, verbose=verbose, n_jobs=-1)

    if config.clf == 'MLP':
        clf = MLP_classifier(device=device)

    return clf


def save_clf(classifier, config):
    '''
    helper function to save classifier

    '''

    if not os.path.exists(config.dump_path+'/clf'):
        os.makedirs(config.dump_path+'/clf')

    if config.clf == 'linear':
        torch.save(classifier.state_dict(), config.dump_path +
                   '/clf/' + config.model_name + '_linear_classifier.pth')

    if config.clf == 'xgboost':
        classifier.save_model(
            config.dump_path + '/clf/' + config.model_name + '_xgboost_classifier.json')

    if config.clf == 'random_forest':
        joblib.dump(classifier, config.dump_path + '/clf/' +
                    config.model_name + '_rf_classifier.joblib')

    if config.clf == 'MLP':
        classifier.save_model(config)


'''
Below are implementations of the different classifier.
Except for Random Forest. Here the standard sklearn library was used. 

'''


# -------------------- Logistic regression -------------------- #


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
            # labels = torch.from_numpy(y_test).type(torch.float)
            outputs = torch.squeeze(self(x)).round().detach().numpy()
            return outputs


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
        return out


# -------------------- Multi Layer Perceptron -------------------- #


class MLP_classifier(nn.Module):
    def __init__(self, device):
        super(MLP_classifier, self).__init__()
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.num_epochs = 200
        self.device = device

    def forward(self, x):
        out = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
        return out

    def fit(self, X, y):
        self.model = MLP(X.shape[1], 100, 1)
        self.model.to(self.device)

        self.criterion = torch.nn.BCELoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1)

        training_samples = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train))
        validation_samples = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val))
        data_loader_trn = torch.utils.data.DataLoader(
            training_samples, batch_size=200, drop_last=False, shuffle=True)
        data_loader_val = torch.utils.data.DataLoader(
            validation_samples, batch_size=200, drop_last=False, shuffle=True)

        val_loss = None
        counter = 0
        for epoch in range(self.num_epochs):
            for batch_idx, (data, target) in enumerate(data_loader_trn):

                tr_x, tr_y = data.to(self.device), target.to(self.device)

                pred = self.model(tr_x)
                loss = self.criterion(torch.squeeze(pred), tr_y.float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch > 50:
                with torch.no_grad():
                    current_validation_loss = 0
                    for batch_idx, (data, target) in enumerate(data_loader_val):
                        val_x, val_y = data.float(), target.float()
                        val_x, val_y = data.to(
                            self.device), target.to(self.device)
                        pred = self.model(val_x)
                        current_validation_loss += self.criterion(
                            torch.squeeze(pred), val_y.float())
                    # check for early stopping
                    if val_loss is None:
                        val_loss = current_validation_loss
                    elif val_loss < current_validation_loss + 0.0001:
                        counter += 1
                        if counter >= 10:
                            print(f'early stopping after {epoch} epochs.')
                            break
                    elif val_loss > current_validation_loss + 0.0001:
                        val_loss = current_validation_loss
                        counter = 0

    def predict(self, X_test):
        with torch.no_grad():
            x = torch.from_numpy(X_test)
            x = x.to(self.device)
            # labels = torch.from_numpy(y_test).type(torch.float)
            outputs = torch.squeeze(self.model(
                x)).cpu().detach().numpy()
            return outputs.round()

    def save_model(self, config):
        torch.save(self.model.state_dict(), config.dump_path +
                   '/clf/'+config.model_name+str(config.patch_size)+'.pth')


# -------------------- Extreme Gradient Boosting -------------------- #


class XGBoost():
    '''
    boosting classifier
    '''

    def __init__(self, config):
        super(XGBoost, self).__init__()
        self.gpu_id = config.gpu_ids[0]

    def fit(self, X, y):
        param = {'objective': 'binary:logistic',
                 'tree_method': 'gpu_hist',
                 'gpu_id': self.gpu_id,
                 'eval_metric': 'logloss'}
        idx = np.random.choice(X.shape[0], X.shape[0]-1)
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
