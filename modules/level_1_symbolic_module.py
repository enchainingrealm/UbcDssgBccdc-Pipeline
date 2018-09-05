import json
import os
import pickle

import pandas as pd

from util.classifier import load_candidates
from util.get_keys import get_keys
from util.get_one import get_one
from util.preprocessor import labels_to_lowercase


class Level1SymbolicModule:
    def __init__(self, to_module=None):
        """
        Returns a new, untrained Level1SymbolicModule.
        :param to_module: a trained TestOutcomeModule that this
        Level1SymbolicModule refers to when classifying new data
        """
        self.to_module = to_module
        self.dictionary = None

    def retrain(self, raw_df):
        """
        Retrains this Level1SymbolicModule on the given data. Raises a
        ValueError if the given DataFrame is empty.
        - Converts all labels in the given DataFrame to lowercase
        - Populates this Level1SymbolicModule's dictionary with all level_1
          labels in the given DataFrame, excluding "*not found"
        - Splits all level_1 labels containing multiple organisms, inserting
          each organism into this Level1SymbolicModule's dictionary individually
        - Corrects "influzena' to "influenza" in this Level1SymbolicModule's
          dictionary
        :param raw_df: a DataFrame containing the raw training data extracted
        from the database
        - required columns: {"level_1"}
        :return: None
        """
        if raw_df.empty:
            raise ValueError("Cannot retrain Level1SymbolicModule on empty set.")

        print("Level1SymbolicModule: Started retraining")

        df = labels_to_lowercase(raw_df)

        self.dictionary = set()
        for raw_label in df["level_1"]:
            labels = raw_label.split(" or ")
            self.dictionary.update(labels)

        if "influzena" in self.dictionary:
            self.dictionary.remove("influzena")
            self.dictionary.add("influenza")

        if "*not found" in self.dictionary:
            self.dictionary.remove("*not found")

        print("Level1SymbolicModule: Finished retraining")

    def classify(self, raw_df, observations=False, return_all=False):
        """
        Classifies the given data. Raises a ValueError if this
        Level1SymbolicModule has not been trained.
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
          True), "level_1_symbolic_pred"}
        """
        if not self._is_trained():
            raise ValueError("Level1SymbolicModule is not trained.")

        keys = get_keys(observations)

        if self.to_module is None:
            df = raw_df
        else:
            to_results = self.to_module.classify(raw_df, observations)
            df = pd.merge(raw_df, to_results, how="inner", on=keys)

        df["level_1_symbolic_pred"] = df.apply(
            lambda row: self._classify_row(row, return_all),
            axis=1
        )

        result = df.loc[:, keys + ["level_1_symbolic_pred"]]
        return result

    def _classify_row(self, row, return_all):
        """
        Classifies the given row.
        Precondition: this Level1SymbolicModule has been trained.
        :param row: the data row to classify
        - required columns: {"test_outcome_pred" (if self.to_module is not
          None), "candidates"}
        :param return_all: True to return all candidate organisms tagged by
        MetaMap, False to return only the most likely candidate organism
        :return: the classification (the most likely organism if return_all is
        False, or a string representation of the List of all candidate organisms
        tagged by MetaMap)
        """
        # check test outcome
        if self.to_module is not None:
            if row["test_outcome_pred"] == "negative":
                return "*not found"

        # preprocess candidates
        candidates = load_candidates(row["candidates"])
        candidates = [candidate.lower() for candidate in candidates]

        # ----------------------------------------------------------------------
        # predict

        if return_all:
            return json.dumps(candidates)

        if not candidates:
            return "*not found"

        for candidate in candidates:
            words = candidate.split()
            for i in range(len(words), 1 - 1, -1):
                level_1 = " ".join(words[:i])
                if level_1 in self.dictionary:
                    return "influzena" if level_1 == "influenza" else level_1

        return get_one(candidates)

    def _is_trained(self):
        """
        Returns True iff the retrain method has been called on this
        Level1SymbolicModule instance at least once.
        :return: whether this Level1SymbolicModule has been trained
        """
        return self.dictionary is not None

    def load_from_file(self, filepath):
        """
        Loads the dictionary stored in the pickle file at the given path into
        this Level1SymbolicModule, overwriting this Level1SymbolicModule's
        current dictionary.
        :param filepath: the absolute path to the pickle file to load the
        dictionary from
        :return: this Level1SymbolicModule
        """
        with open(filepath, "rb") as file:
            self.dictionary = pickle.load(file)
        return self

    def save_to_file(self, filepath):
        """
        Saves this Level1SymbolicModule's dictionary to the pickle file at the
        given path, overwriting the file if it already exists.
        :param filepath: the absolute path to the pickle file to save the
        dictionary to
        :return: None
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file:
            pickle.dump(self.dictionary, file)
