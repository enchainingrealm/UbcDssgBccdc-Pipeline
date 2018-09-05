import numpy as np

from math import inf

from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, cohen_kappa_score, confusion_matrix


def compute_stats(y_true, y_pred, labels=None):
    """
    Computes and returns performance stats (accuracy, precision, recall, F1
    score, Cohen Kappa score, and confusion matrix).
    :param y_true: the List of true labels
    :param y_pred: the List of predicted labels
    :param labels: the preferred order of the labels (for precision, recall, F1
    score, and confusion matrix); leave this parameter default to use sklearn's
    default label ordering
    :return: a Dict with keys {"accuracy", "precision", "recall", "f1", "kappa",
    "confusion", "labels"}
    """
    if labels is None:
        labels = sorted(set(y_true + y_pred))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, labels=labels, average=None),
        "recall": recall_score(y_true, y_pred, labels=labels, average=None),
        "f1": f1_score(y_true, y_pred, labels=labels, average=None),
        "kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
        "labels": labels
    }


def stats_to_str(stats):
    """
    Returns a string representation of the given performance stats Dict. All
    floating point numbers are rounded to at most 4 decimal places.
    :param stats: a performance stats Dict obtained from compute_stats
    :return: a string representation of the given performance stats Dict
    """
    np.set_printoptions(linewidth=inf, formatter={
        "float": lambda x: "{0:0.4f}".format(x)
    })

    return "Accuracy: " + str(stats["accuracy"]) + "\n"\
           + "Precision: " + np.array2string(stats["precision"]) + "\n"\
           + "Recall: " + np.array2string(stats["recall"]) + "\n"\
           + "F1 score: " + np.array2string(stats["f1"]) + "\n"\
           + "Cohen Kappa score: " + str(stats["kappa"]) + "\n"\
           + "Confusion matrix:\n" + np.array2string(stats["confusion"]) + "\n"\
           + "Labels: " + str(stats["labels"]) + "\n"


def accuracies_to_str(accuracies):
    """
    Returns a string representation of the given List of accuracies. The string
    representation contains all the accuracies, their mean, and their standard
    deviation. All floating point numbers are rounded to at most 4 decimal
    places.
    :param accuracies: a List of accuracies, most likely obtained from a
    cross-validation procedure
    :return: a string representation of the given List of accuracies
    """
    np.set_printoptions(linewidth=inf, formatter={
        "float": lambda x: "{0:0.4f}".format(x)
    })

    accuracies = np.asarray(accuracies)

    return "Accuracies: " + np.array2string(accuracies) + "\n"\
           + "Mean: " + str(np.mean(accuracies)) + "\n"\
           + "Standard deviation: " + str(np.std(accuracies)) + "\n"
