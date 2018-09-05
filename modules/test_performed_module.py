import json
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from util.classifier import best_classifier, get_confidences
from util.get_keys import get_keys
from util.preprocessor import preprocess
from util.vectorizer import vectorize


class TestPerformedModule:
    def __init__(self, organisms=True):
        """
        Returns a new, untrained TestPerformedModule.
        :param organisms: whether to replace all organism names in the training
        and test result_full_descriptions with "_ORGANISM_"
        """
        self.vectorizer = None
        self.classifier = None
        self.organisms = organisms
        self.scale = None

    def retrain(self, raw_df):
        """
        Retrains this TestPerformedModule on the given data. Raises a ValueError
        if the given DataFrame is empty.
        - Preprocesses the result_full_descriptions and labels in the given
          DataFrame
        - Fits the vectorizer to the given result_full_descriptions
        - Selects the best classifier by using a 5-fold cross-validation process
        - Trains the selected classifier on the given data
        :param raw_df: a DataFrame containing the raw training data extracted
        from the database
        - required columns: {"result_full_description", "test_performed",
          "candidates" (if self.organisms is True)}
        :return: None
        """
        if raw_df.empty:
            raise ValueError("Cannot retrain TestPerformedModule on empty set.")

        print("TestPerformedModule: Started retraining")

        df = preprocess(raw_df, organisms=self.organisms)

        self.vectorizer = self._get_vectorizer(df)
        self.classifier = best_classifier(
            df, "test_performed", self._get_vectorizer,
            self._get_candidate_classifiers()
        )()

        X = self.vectorizer.transform(df["result_full_description"])
        y = df["test_performed"]

        self.classifier.fit(X, y)
        if isinstance(self.classifier, LinearSVC):
            confidences, _ = get_confidences(self.classifier, X, scale=1)
            self.scale = np.max(confidences)

        print("TestPerformedModule: Finished retraining")

    @staticmethod
    def _get_vectorizer(df_train):
        """
        Returns a new, fitted CountVectorizer with parameters optimized for
        predicting test_performed.
        - uni/bi/trigrams are used
        - minimum document frequency is 10
        - features with variance below 0.001 are removed
        :param df_train: a DataFrame containing the preprocessed training data
        to fit the vectorizer on
        - required columns: {"result_full_description"}
        :return: a new, fitted CountVectorizer
        """
        vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=10)
        X_train, _, _ = vectorize(
            vectorizer, df_train["result_full_description"])

        selection = VarianceThreshold(threshold=0.001)
        selection.fit(X_train)
        support = selection.get_support(indices=True)

        feature_names = vectorizer.get_feature_names()
        vocabulary = [feature_names[index] for index in support]
        return CountVectorizer(vocabulary=vocabulary)

    @staticmethod
    def _get_candidate_classifiers():
        """
        Returns a List of 0-argument lambdas for constructing instances of
        candidate classifiers for predicting test_performed.
        - Logistic Regression with l2 or l1 regularization
        - Random Forest with 100 trees and multicore processing
        - Support Vector Machine with linear kernel, and l2 or l1 regularization
        :return: a List of constructors of candidate classifiers
        """
        return [
            LogisticRegression,
            lambda: LogisticRegression(penalty="l1"),
            lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1),
            lambda: LinearSVC(dual=False),
            lambda: LinearSVC(penalty="l1", dual=False)
        ]

    def classify(self, raw_df, observations=False):
        """
        Classifies the given data. Raises a ValueError if this
        TestPerformedModule has not been trained.
        :param raw_df: a DataFrame containing the raw test data extracted from
        the database
        - required columns: {"test_key", "result_key", "obs_seq_nbr" (if
          observations is True), "result_full_description", "candidates" (if
          self.organisms is True)}
        :param observations: True if the data is given at the observation level,
        False if the data is given at the test level
        :return: a DataFrame containing the classification results
        - columns: {"test_key", "result_key", "obs_seq_nbr" (if observations is
          True), "test_performed_pred", 'test_performed_classifier",
          "test_performed_confidence", "test_performed_confidence_type"}
        """
        if not self._is_trained():
            raise ValueError("TestPerformedModule is not trained.")

        keys = get_keys(observations)

        if raw_df.shape[0] == 0:
            return pd.DataFrame(columns=keys + [
                "test_performed_pred",
                "test_performed_classifier",
                "test_performed_confidence",
                "test_performed_confidence_type"
            ])

        df = preprocess(raw_df, organisms=self.organisms)

        X = self.vectorizer.transform(df["result_full_description"])
        y_pred = self.classifier.predict(X)

        result = df.loc[:, keys]
        result["test_performed_pred"] = y_pred

        result["test_performed_classifier"] = json.dumps({
            "type": self.classifier.__class__.__name__,
            "params": self.classifier.get_params()
        })

        confidence, confidence_type\
            = get_confidences(self.classifier, X, self.scale)
        result["test_performed_confidence"] = confidence
        result["test_performed_confidence_type"] = confidence_type

        return result

    def _is_trained(self):
        """
        Returns True iff the retrain method has been called on this
        TestPerformedModule instance at least once.
        :return: whether this TestPerformedModule has been trained
        """
        return self.vectorizer is not None\
            and self.classifier is not None

    @staticmethod
    def load_from_file(filepath):
        """
        Returns a new TestPerformedModule whose internal state is loaded from
        the pickle file at the given path.
        :param filepath: the absolute path to the pickle file to load state from
        :return: a new TestPerformedModule, loaded from the pickle file
        """
        _self = TestPerformedModule()

        with open(filepath, "rb") as file:
            _self.vectorizer = pickle.load(file)
            _self.classifier = pickle.load(file)
            _self.organisms = pickle.load(file)
            _self.scale = pickle.load(file)

        return _self

    def save_to_file(self, filepath):
        """
        Saves the state of this TestPerformedModule to the pickle file at the
        given path, overwriting the file if it already exists.
        :param filepath: the absolute path to the pickle file to write to
        :return: None
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file:
            pickle.dump(self.vectorizer, file)
            pickle.dump(self.classifier, file)
            pickle.dump(self.organisms, file)
            pickle.dump(self.scale, file)
