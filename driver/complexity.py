import logging
import sys

import matplotlib.pyplot as plt

from datetime import datetime

from io_.db import Database
from io_.fs import write_plot
from modules.level_1_ml_module import Level1MLModule
from modules.level_1_symbolic_module import Level1SymbolicModule
from modules.level_2_module import Level2Module
from modules.test_outcome_module import TestOutcomeModule
from modules.test_performed_module import TestPerformedModule
from root import from_root
from util.logger import set_params
from util.timer import timer


TP_SQL = from_root("sql\\train\\test_performed.sql")
TO_SQL = from_root("sql\\train\\test_outcome.sql")
L1_SQL = from_root("sql\\train\\level_1.sql")
L2_SQL = from_root("sql\\train\\level_2.sql")

SIZES = [i for i in range(2000, 100000 + 1, 2000)]
ORGANISMS = True

SAVE_TO = from_root("results\\complexity.png")


def main():
    db = Database.get_instance()

    # ==========================================================================
    # Test performed

    def tp_module_factory():
        return TestPerformedModule(organisms=ORGANISMS)

    tp_df = db.extract(TP_SQL)
    tp_times = timer(tp_module_factory, tp_df, SIZES)
    tp_line, = plt.plot(SIZES, tp_times, marker="o", label="Test performed")

    # ==========================================================================
    # Test outcome

    def to_module_factory():
        return TestOutcomeModule(organisms=ORGANISMS)

    to_df = db.extract(TO_SQL)
    to_times = timer(to_module_factory, to_df, SIZES)
    to_line, = plt.plot(SIZES, to_times, marker="o", label="Test outcome")

    # ==========================================================================
    # Level 1 (machine learning)

    def l1ml_module_factory():
        return Level1MLModule()

    l1_df = db.extract(L1_SQL)
    l1ml_times = timer(l1ml_module_factory, l1_df, SIZES)
    l1ml_line, = plt.plot(
        SIZES, l1ml_times, marker="o", label="Level 1 (machine learning)")

    # ==========================================================================
    # Level 1 (symbolic)

    def l1s_module_factory():
        return Level1SymbolicModule(None)   # we pass None into the constructor
        # because we will not be classifying results using this module instance;
        # we are retraining it just to time the retraining process.

    l1s_times = timer(l1s_module_factory, l1_df, SIZES)
    l1s_line, = plt.plot(
        SIZES, l1s_times, marker="o", label="Level 1 (symbolic)")

    # ==========================================================================
    # Level 2

    def l2_module_factory():
        return Level2Module(None)   # we pass None into the constructor because
        # we will not be classifying results using this module instance; we are
        # retraining it just to time the retraining process.

    l2_df = db.extract(L2_SQL)
    l2_times = timer(l2_module_factory, l2_df, SIZES)
    l2_line, = plt.plot(SIZES, l2_times, marker="o", label="Level 2")

    # ==========================================================================

    plt.title("Time complexity plot")
    plt.xlabel("Training data size (number of rows)")
    plt.ylabel("Retrain time (seconds)")
    plt.legend(handles=[tp_line, to_line, l1ml_line, l1s_line, l2_line])

    write_plot(SAVE_TO, plt)


if __name__ == "__main__":
    print("Started executing script.\n")
    start_time = datetime.now()

    logger = logging.getLogger(__name__)
    set_params(logger, from_root("log\\complexity.log"))

    try:
        main()
    except Exception as e:
        logger.exception("complexity.py: Fatal error")
        sys.exit(1)

    print(f"\nExecution time: {datetime.now() - start_time}")
    print("Finished executing script.")
