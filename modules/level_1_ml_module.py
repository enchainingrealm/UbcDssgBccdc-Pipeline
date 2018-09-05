import json
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from util.classifier import best_classifier, get_confidences
from util.get_keys import get_keys
from util.preprocessor import preprocess
from util.vectorizer import vectorize


class Level1MLModule:
    def __init__(self):
        """
        Returns a new, untrained Level1MLModule.
        """
        self.vectorizer = None
        self.classifier = None
        self.scale = None

    def retrain(self, raw_df):
        """
        Retrains this Level1MLModule on the given data. Raises a ValueError if
        the given DataFrame is empty.
        - Preprocesses the result_full_descriptions and labels in the given
          DataFrame
        - Fits the vectorizer to the given result_full_descriptions
        - Selects the best classifier by using a 5-fold cross-validation process
        - Trains the selected classifier on the given data
        :param raw_df: a DataFrame containing the raw training data extracted
        from the database
        - required columns: {"result_full_description", "level_1"}
        :return: None
        """
        if raw_df.empty:
            raise ValueError("Cannot retrain Level1MLModule on empty set.")

        print("Level1MLModule: Started retraining")

        df = preprocess(raw_df)

        self.vectorizer = self._get_vectorizer(df)
        self.classifier = best_classifier(
            df, "level_1", self._get_vectorizer,
            self._get_candidate_classifiers()
        )()

        X = self.vectorizer.transform(df["result_full_description"])
        y = df["level_1"]

        self.classifier.fit(X, y)
        if isinstance(self.classifier, LinearSVC):
            confidences, _ = get_confidences(self.classifier, X, scale=1)
            self.scale = np.max(confidences)

        print("Level1MLModule: Finished retraining")

    @staticmethod
    def _get_vectorizer(df_train):
        """
        Returns a new, fitted CountVectorizer with parameters optimized for
        predicting level_1.
        - uni/bi/trigrams are used
        - there is no minimum document frequency
        - the top 100 features (as per the chi-squared test) are kept
        :param df_train: a DataFrame containing the preprocessed training data
        to fit the vectorizer on
        - required columns: {"result_full_description", "level_1"}
        :return: a new, fitted CountVectorizer
        """
        vectorizer = CountVectorizer(ngram_range=(1, 3))
        X_train, _, _ = vectorize(
            vectorizer, df_train["result_full_description"])
        y_train = df_train["level_1"]

        selection = SelectKBest(chi2, 200)
        selection.fit(X_train, y_train)
        support = selection.get_support(indices=True)

        feature_names = vectorizer.get_feature_names()
        vocabulary = [feature_names[index] for index in support]
        return CountVectorizer(vocabulary=vocabulary)

    @staticmethod
    def _get_candidate_classifiers():
        """
        Returns a List of 0-argument lambdas for constructing instances of
        candidate classifiers for predicting level_1.
        - Logistic Regression
        - Random Forest with 100 trees and multicore processing
        - AdaBoost with 100 decision stumps
        - Support Vector Machine with linear kernel
        :return: a List of constructors of candidate classifiers
        """
        return [
            LogisticRegression,
            lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1),
            lambda: AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=100),
            LinearSVC
        ]

    def classify(self, raw_df, observations=False):
        """
        Classifies the given data. Raises a ValueError if this Level1MLModule
        has not been trained.
        :param raw_df: a DataFrame containing the raw test data extracted from
        the database
        - required columns: {"test_key", "result_key", "obs_seq_nbr" (if
          observations is True), "result_full_description"}
        :param observations: True if the data is given at the observation level,
        False if the data is given at the test level
        :return: a DataFrame containing the classification results
        - columns: {"test_key", "result_key", "obs_seq_nbr" (if observations is
          True), "level_1_ml_pred", 'level_1_ml_classifier",
          "level_1_ml_confidence", "level_1_ml_confidence_type"}
        """
        if not self._is_trained():
            raise ValueError("Level1MLModule is not trained.")

        keys = get_keys(observations)

        if raw_df.shape[0] == 0:
            return pd.DataFrame(columns=keys + [
                "level_1_ml_pred",
                "level_1_ml_classifier",
                "level_1_ml_confidence",
                "level_1_ml_confidence_type"
            ])

        df = preprocess(raw_df)

        X = self.vectorizer.transform(df["result_full_description"])
        y_pred = self.classifier.predict(X)

        result = df.loc[:, keys]
        result["level_1_ml_pred"] = y_pred

        result["level_1_ml_classifier"] = json.dumps({
            "type": self.classifier.__class__.__name__,
            "params": self.classifier.get_params()
        })

        confidence, confidence_type\
            = get_confidences(self.classifier, X, self.scale)
        result["level_1_ml_confidence"] = confidence
        result["level_1_ml_confidence_type"] = confidence_type

        return result

    def _is_trained(self):
        """
        Returns True iff the retrain method has been called on this
        Level1MLModule instance at least once.
        :return: whether this Level1MLModule has been trained
        """
        return self.vectorizer is not None\
            and self.classifier is not None

    @staticmethod
    def load_from_file(filepath):
        """
        Returns a new Level1MLModule whose internal state is loaded from the
        pickle file at the given path.
        :param filepath: the absolute path to the pickle file to load state from
        :return: a new Level1MLModule, loaded from the pickle file
        """
        _self = Level1MLModule()

        with open(filepath, "rb") as file:
            _self.vectorizer = pickle.load(file)
            _self.classifier = pickle.load(file)
            _self.scale = pickle.load(file)

        return _self

    def save_to_file(self, filepath):
        """
        Saves the state of this Level1MLModule to the pickle file at the given
        path, overwriting the file if it already exists.
        :param filepath: the absolute path to the pickle file to write to
        :return: None
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file:
            pickle.dump(self.vectorizer, file)
            pickle.dump(self.classifier, file)
            pickle.dump(self.scale, file)
