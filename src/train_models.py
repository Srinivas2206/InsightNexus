import os

import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from data_preprocessing import split_and_scale_data, preprocess_data


def train_and_save_models(df):
    """ this function trains various machine learning models like
    1.logistic regression
    2.adaboost classifier
    3.decision tree classifier
    4.gradient boost classifier
    5. SVC
    6.random forest classifier
    7.XG boost classifier
    8.K-neighbour classifier
    These trained models are further saved in a file.
    parameter:
    ---------
    df (pandas.DataFrame)

    """
    X_train, X_test, y_train, y_test = split_and_scale_data(df)

    models = {
        "logistic_regression_model": LogisticRegression(),
        "svc_model": SVC(),
        "decision_tree_model": DecisionTreeClassifier(),
        "adaboost_model": AdaBoostClassifier(),
        "gradient_boosting_model": GradientBoostingClassifier(),
        "random_forest_model": RandomForestClassifier(),
        "xgboost_model": XGBClassifier(),
        "kneighbors_model": KNeighborsClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        dump(model, f'../models/{name}.joblib')

if __name__ == "__main__":
    df = pd.read_csv('../data/online_course_engagement_data.csv')
    df, _ = preprocess_data(df)
    train_and_save_models(df)
