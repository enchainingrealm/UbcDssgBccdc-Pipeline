import json
from math import inf

import numpy as np
from numpy import nan
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

from util.extrema import ind_max
from util.vectorizer import vectorize


def best_classifier(df, output, vectorizer_factory, classifier_factories):
    """
    Evaluates the expected performance of each classifier on the given data
    using 5-fold cross-validation. The given vectorizer is used to convert the
    data into features. Returns an instance of the best classifier.
    :param df: the preprocessed DataFrame containing the data to run
    cross-validation with
    - required columns: {"result_full_description", output}
    :param output: the name of the DataFrame column containing the true labels
    :param vectorizer_factory: a lambda that takes in a training DataFrame and
    returns a new, fitted instance of the vectorizer to use
    :param classifier_factories: an Iterable of 0-argument lambdas that return
    new, untrained instances of the classifiers to evaluate
    :return: a new, untrained instance of the best classifier
    """
    accuracies = [[] for _ in classifier_factories]

    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    for fold, (train_indices, test_indices) in enumerate(kf.split(df)):
        print(f"Started evaluating fold {fold + 1} of {N_SPLITS}")

        fold_accuracies = _evaluate_fold(
            df, train_indices, test_indices, output,
            vectorizer_factory, classifier_factories)

        for index, accuracy in enumerate(fold_accuracies):
            accuracies[index].append(accuracy)

        print(f"Finished evaluating fold {fold + 1} of {N_SPLITS}")

    mean_accuracies = [
        np.mean(accuracies[i]) for i in range(len(classifier_factories))
    ]
    return classifier_factories[ind_max(mean_accuracies)]


def _evaluate_fold(
        df, train_indices, test_indices, output,
        vectorizer_factory, classifier_factories
):
    """
    Evaluates the performance of the given classifiers on the given training and
    test sets. Returns a List containing the classifiers' accuracies.
    :param df: the preprocessed DataFrame containing the training and test data
    - required columns: {"result_full_description", output}
    :param train_indices: the indices of the training rows
    :param test_indices: the indices of the test rows
    :param output: the name of the DataFrame column containing the true labels
    :param vectorizer_factory: a lambda that takes in a training DataFrame and
    returns a new, fitted instance of the vectorizer to use
    :param classifier_factories: an Iterable of 0-argument lambdas that return
    new, untrained instances of the classifiers to evaluate
    :return: a List whose ith element is the ith classifier's accuracy
    """
    df_train = df.iloc[train_indices, :]
    df_test = df.iloc[test_indices, :]

    vectorizer = vectorizer_factory(df_train)
    X_train, _, _ = vectorize(vectorizer, df_train["result_full_description"])
    y_train = df_train[output]

    accuracies = []

    n = len(classifier_factories)
    for index, classifier_factory in enumerate(classifier_factories):
        print(f"Evaluating classifier {index + 1} out of {n}... ",
              end="", flush=True)

        classifier = classifier_factory()
        classifier.fit(X_train, y_train)

        X_test = vectorizer.transform(df_test["result_full_description"])
        y_true = df_test[output]
        y_pred = classifier.predict(X_test)

        accuracy = np.mean(y_true == y_pred)
        accuracies.append(accuracy)

        print("Finished")

    return accuracies


def get_confidences(classifier, X, scale):
    """
    Computes the given classifier's prediction confidences on the given data.
    - For any classifier with a "predict_proba" method, probability measures are
      used as confidences.
    - For LinearSVC, the distance from a data point to the hyperplane separating
      the classes, scaled by the maximum such distance in the training set is
      used as a confidence measure.
    - Throws a ValueError for any other classifier.
    :param classifier: the classifier to compute classification confidences for
    :param X: the feature matrix for the (test) data
    :param scale: the maximum distance from a data point in the training set to
    the hyperplane separating the classes, if classifier is an instance of
    LinearSVC. Ignored otherwise.
    :return: the List of confidence measures (the ith element is the confidence
             measure for the ith test row);
             a List containing the confidence type for each row (all
             "probability" if "predict_proba" was used; all "scaled_distance"
             if classifier is a LinearSVC instance)
    """
    if callable(getattr(classifier, "predict_proba", None)):
        # MultinomialNB, LogisticRegression, RandomForestClassifier,
        # GradientBoostingClassifier, AdaBoostClassifier, MLPClassifier
        confidences = classifier.predict_proba(X).max(axis=1)
        confidence_type = ["probability" for x in range(X.shape[0])]
    elif isinstance(classifier, LinearSVC):
        decision = classifier.decision_function(X)
        if len(decision.shape) == 1:
            # binary case
            confidences = np.abs(decision) / scale
        else:
            # multiclass case
            assert len(decision.shape) == 2
            confidences = np.apply_along_axis(np.max, 1, np.abs(decision))\
                          / scale

        # deal with divide-by-zero problem (occurs if scale is 0)
        confidences[confidences == nan] = 0
        confidences[confidences == inf] = 0

        confidence_type = ["scaled_distance" for x in range(X.shape[0])]
    else:
        raise ValueError

    return confidences, confidence_type


def load_candidates(candidates_str):
    """
    Deserializes a JSON string containing MetaMap candidates information into a
    Set of preferred organism names. The Set contains all the keys of the JSON
    string (except "Bacteria" and "Virus",) as those are the preferred organism
    names returned from MetaMap.
    :param candidates_str: a JSON string containing MetaMap candidates
    information
    :return: a Set containing the preferred organism names
    """
    candidates_str = candidates_str
    candidates_dict = json.loads(candidates_str)
    candidates = set(candidates_dict.keys())

    banned = {"Bacteria", "Virus"}
    candidates.difference_update(banned)

    return candidates
