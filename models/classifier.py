
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


def classify(config, data):

    clf = get_classifier(config)


def get_classifier(config):

    if config.clf == 'xgboost':
        clf = xgb.XGBClassifier()

    if config.clf == 'random_forest':
        clf = RandomForestClassifier()

    return clf
