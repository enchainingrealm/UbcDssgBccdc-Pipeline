import json
import os
import pickle

import pandas as pd

from util.classifier import load_candidates
from util.get_keys import get_keys
from util.get_one import get_one
from util.preprocessor import labels_to_lowercase


class Level2Module:
    def __init__(self, l1_module):
        """
        Returns a new, untrained Level2Module.
        :param l1_module: a trained Level1MLModule or Level1SymbolicModule that
        this Level2Module refers to when classifying new data
        """
        self.l1_module = l1_module
        self.dictionary = None

    def retrain(self, raw_df):
        """
        Retrains this Level2Module on the given data. Raises a ValueError if the
        given DataFrame is empty.
        - Converts all labels in the given DataFrame to lowercase
        - Populates this Level2Module's dictionary's keys with all level_1
          labels in the given DataFrame, excluding "*not found"
        - Maps each level_1 label in this Level2Module's dictionary to a List of
          level_2 labels that have appeared in the given DataFrame along with
          the level_1 label
        :param raw_df: a DataFrame containing the raw training data extracted
        from the database
        - required columns: {"level_1", "level_2"}
        :return: None
        """
        if raw_df.empty:
            raise ValueError("Cannot retrain Level2Module on empty set.")

        print("Level2Module: Started retraining")

        df = labels_to_lowercase(raw_df)

        self.dictionary = {}
        for index, row in df.iterrows():
            raw_l1_label = row["level_1"]
            l1_labels = raw_l1_label.split(" or ")

            l2_label = row["level_2"]

            if raw_l1_label not in self.dictionary:
                self.dictionary[raw_l1_label] = set()
            self.dictionary[raw_l1_label].add(l2_label)

            for l1_label in l1_labels:
                if l1_label not in self.dictionary:
                    self.dictionary[l1_label] = set()
                self.dictionary[l1_label].add(l2_label)

        print("Level2Module: Finished retraining")

    def classify(self, raw_df, observations=False, return_all=False):
        """
        Classifies the given data. Raises a ValueError if this Level2Module has
        not been trained.
        :param raw_df: a DataFrame containing the raw test data extracted from
        the database
        - required columns: {"test_key", "result_key", "obs_seq_nbr" (if
          observations is True), "candidates"}
        :param observations: True if the data is given at the observation level,
        False if the data is given at the test level
        :param return_all: True to return all candidate organisms tagged by
        MetaMap, False to return only the most likely candidate organism
        :return: a DataFrame containing the classification results
        - columns: {"test_key", "result_key", "obs_seq_nbr" (if observations is
          True), "level_2_pred"}
        """
        if not self._is_trained():
            raise ValueError("Level2Module is not trained.")

        keys = get_keys(observations)

        l1_results = self.l1_module.classify(raw_df, observations)
        if "level_1_symbolic_pred" in l1_results:
            l1_results.rename(
                columns={"level_1_symbolic_pred": "level_1_pred"}, inplace=True)
        elif "level_1_ml_pred" in l1_results:
            l1_results.rename(
                columns={"level_1_ml_pred": "level_1_pred"}, inplace=True)

        df = pd.merge(raw_df, l1_results, how="inner", on=keys)

        df["level_2_pred"] = df.apply(
            lambda row: self._classify_row(row, return_all),
            axis=1
        )

        result = df.loc[:, keys + ["level_2_pred"]]
        return result

    def _classify_row(self, row, return_all):
        """
        Classifies the given row.
        Precondition: this Level2Module has been trained.
        :param row: the data row to classify
        - required columns: {"level_1_pred", "candidates"}
        :param return_all: True to return all candidate organisms tagged by
        MetaMap, False to return only the most likely candidate organism
        :return: the classification (the most likely organism if return_all is
        False, or a string representation of the List of all candidate organisms
        tagged by MetaMap)
        """
        # check level 1
        level_1 = row["level_1_pred"]
        if level_1 == "*not found":
            return "*not found"
        elif level_1 not in self.dictionary:
            return "*no further diff"

        # preprocess candidates
        candidates = load_candidates(row["candidates"])
        candidates = [candidate.lower() for candidate in candidates]

        # ----------------------------------------------------------------------
        # predict

        if return_all:
            return json.dumps(candidates)

        if not candidates:
            return "*not further diff"

        for candidate in candidates:
            words = candidate.split()
            for i in range(len(words), 1 - 1, -1):
                level_2 = " ".join(words[:i])
                if level_2 in self.dictionary[level_1]:
                    return level_2

        return get_one(candidates)

    def _is_trained(self):
        """
        Returns True iff the retrain method has been called on this Level2Module
        instance at least once.
        :return: whether this Level2Module has been trained
        """
        return self.dictionary is not None

    def load_from_file(self, filepath):
        """
        Loads the dictionary stored in the pickle file at the given path into
        this Level2Module, overwriting this Level2Module's current dictionary.
        :param filepath: the absolute path to the pickle file to load the
        dictionary from
        :return: this Level2Module
        """
        with open(filepath, "rb") as file:
            self.dictionary = pickle.load(file)
        return self

    def save_to_file(self, filepath):
        """
        Saves this Level2Module's dictionary to the pickle file at the given
        path, overwriting the file if it already exists.
        :param filepath: the absolute path to the pickle file to save the
        dictionary to
        :return: None
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file:
            pickle.dump(self.dictionary, file)
