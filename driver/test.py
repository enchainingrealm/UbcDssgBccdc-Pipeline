import logging
import sys
from datetime import datetime

from io_.db import Database
from io_.fs import write_df
from modules.level_1_ml_module import Level1MLModule
from modules.level_1_symbolic_module import Level1SymbolicModule
from modules.level_2_module import Level2Module
from modules.test_outcome_module import TestOutcomeModule
from modules.test_performed_module import TestPerformedModule
from root import from_root
from util.logger import set_params


def main():
    # ==========================================================================
    # Load the DataFrames to classify

    db = Database.get_instance()

    tp_df = db.extract(from_root("sql\\test\\test_performed.sql"))
    to_df = db.extract(from_root("sql\\test\\test_outcome.sql"))
    l1_df = db.extract(from_root("sql\\test\\level_1.sql"))
    l2_df = db.extract(from_root("sql\\test\\level_2.sql"))

    print("Finished loading the DataFrames.")

    # ==========================================================================
    # Load modules

    tp_module = TestPerformedModule.load_from_file(
        from_root("pkl\\test_performed_module.pkl"))

    to_module = TestOutcomeModule.load_from_file(
        from_root("pkl\\test_outcome_module.pkl"))

    l1ml_module = Level1MLModule.load_from_file(
        from_root("pkl\\level_1_ml_module.pkl"))

    l1s_module = Level1SymbolicModule(to_module).load_from_file(
        from_root("pkl\\level_1_symbolic_module.pkl"))

    l2_module = Level2Module(l1ml_module).load_from_file(
        from_root("pkl\\level_2_module.pkl"))

    tp_module_org_false = TestPerformedModule.load_from_file(
        from_root("pkl\\test_performed_organisms_false_module.pkl"))

    to_module_org_false = TestOutcomeModule.load_from_file(
        from_root("pkl\\test_outcome_organisms_false_module.pkl"))

    print("Finished loading modules.")

    # ==========================================================================
    # Classify the DataFrames

    tp_results = tp_module.classify(tp_df)
    to_results = to_module.classify(to_df)
    l1ml_results = l1ml_module.classify(l1_df)
    l1s_results = l1s_module.classify(l1_df)
    l2_results = l2_module.classify(l2_df)

    tp_org_false_results = tp_module_org_false.classify(tp_df)
    to_org_false_results = to_module_org_false.classify(to_df)

    l1s_retall_results = l1s_module.classify(l1_df, return_all=True)
    l2_retall_results = l2_module.classify(l2_df, return_all=True)

    print("Finished classifying the DataFrames.")

    # ==========================================================================
    # Write final prediction results to CSV and database

    results = tp_results\
        .merge(to_results, how="outer", on=["test_key", "result_key"])\
        .merge(l1ml_results, how="outer", on=["test_key", "result_key"])\
        .merge(l1s_results, how="outer", on=["test_key", "result_key"])\
        .merge(l2_results, how="outer", on=["test_key", "result_key"])

    org_false_results = tp_org_false_results\
        .merge(to_org_false_results, how="outer", on=["test_key", "result_key"])

    retall_results = l1s_retall_results\
        .merge(l2_retall_results, how="outer", on=["test_key", "result_key"])

    write_df(from_root("results\\predictions.csv"), results)
    write_df(from_root("results\\predictions_org_false.csv"), org_false_results)
    write_df(from_root("results\\predictions_retall.csv"), retall_results)

    db.insert(results, "predictions", "dbo")

    print("Finished writing results to CSV and database.")


if __name__ == "__main__":
    print("Started executing script.\n")
    start_time = datetime.now()

    logger = logging.getLogger(__name__)
    set_params(logger, from_root("log\\test.log"))

    try:
        main()
    except Exception as e:
        logger.exception("test.py: Fatal error")
        sys.exit(1)

    print(f"\nExecution time: {datetime.now() - start_time}")
    print("Finished executing script.")
