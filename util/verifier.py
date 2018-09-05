import os

from sklearn.model_selection import KFold

from io_.fs import write_text
from util.stats import compute_stats, stats_to_str, accuracies_to_str


def verify_module(module_factory, df, output, save_to):
    """
    Computes and returns the expected accuracy of the given classification
    module, using 5-fold cross validation. Saves the statistics from evaluating
    each fold to the given folder.
    :param module_factory: a 0-argument lambda that returns a new, untrained
    instance of the classification module. For Level1SymbolicModule and
    Level2Module, the returned instance should hold a reference to a trained
    instance of the helper module.
    :param df: the DataFrame to use in the cross-validation process
    :param output: the name of the DataFrame column containing the true labels
    :param save_to: the absolute path to the folder to save the results to
    :return: the expected accuracy of the given classification module
    """
    accuracies = []

    kf = KFold(n_splits=5, shuffle=True)
    for index, (train_indices, test_indices) in enumerate(kf.split(df)):
        df_train = df.iloc[train_indices, :]
        df_test = df.iloc[test_indices, :]

        stats = _evaluate_fold(module_factory, df_train, df_test, output)

        accuracies.append(stats["accuracy"])
        _save_fold_results(index, stats, save_to)

    _save_accuracies(accuracies, save_to)


def _evaluate_fold(module_factory, df_train, df_test, output):
    """
    Evaluates the performance of the given classification module on the given
    training and test sets. Saves performance statistics to the given folder.
    :param module_factory: a 0-argument lambda that returns a new, untrained
    instance of the classification module. For Level1SymbolicModule and
    Level2Module, the returned instance should hold a reference to a trained
    instance of the helper module.
    :param df_train: the DataFrame containing the training rows
    :param df_test: the DataFrame containing the test rows
    :param output: the name of the DataFrame column containing the true labels
    :return: the performance statistics
    """
    module = module_factory()
    module.retrain(df_train)
    results = module.classify(df_test)

    stats = compute_stats(list(df_test[output]), list(results.iloc[:, 2]))
    return stats


def _save_fold_results(index, stats, save_to):
    """
    Saves the performance stats from evaluating the module on the the given
    fold.
    :param index: the index of the fold (0th fold, 1st fold, etc.)
    :param stats: a performance stats Dict obtained from compute_stats
    :param save_to: the absolute path to the folder to save the results to
    :return: None
    """
    write_text(
        os.path.join(save_to, f"fold_{index + 1}.txt"),
        stats_to_str(stats)
    )


def _save_accuracies(accuracies, save_to):
    """
    Saves the fold-wise accuracies from running the verification process. Also
    saves the mean and standard deviation of the accuracies.
    :param accuracies: a List whose ith element is the accuracy of classifying
    the ith fold
    :param save_to: the absolute path to the folder to save the results to
    :return: None
    """
    write_text(
        os.path.join(save_to, "summary.txt"),
        accuracies_to_str(accuracies)
    )
