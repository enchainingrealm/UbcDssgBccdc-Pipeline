import logging
import os
import sys
from datetime import datetime

from modules.level_1_ml_module import Level1MLModule
from modules.level_1_symbolic_module import Level1SymbolicModule
from modules.level_2_module import Level2Module
from modules.test_outcome_module import TestOutcomeModule
from modules.test_performed_module import TestPerformedModule
from root import from_root
from io_.db import Database
from util.logger import set_params
from util.verifier import verify_module


TP_SQL = from_root("sql\\train\\test_performed.sql")
TO_SQL = from_root("sql\\train\\test_outcome.sql")
L1_SQL = from_root("sql\\train\\level_1.sql")
L2_SQL = from_root("sql\\train\\level_2.sql")

ORGANISMS = True

SAVE_TO = from_root("results\\verify")


def main():
    db = Database.get_instance()

    verify_test_performed(db)
    verify_test_outcome(db)
    verify_level_1_ml(db)
    verify_level_1_symbolic(db)
    verify_level_2(db)


def verify_test_performed(db):
    def tp_module_factory():
        return TestPerformedModule(organisms=ORGANISMS)

    tp_df = db.extract(TP_SQL)
    verify_module(tp_module_factory, tp_df, "test_performed",
                  os.path.join(SAVE_TO, "test_performed"))


def verify_test_outcome(db):
    def to_module_factory():
        return TestOutcomeModule(organisms=ORGANISMS)

    to_df = db.extract(TO_SQL)
    verify_module(to_module_factory, to_df, "test_outcome",
                  os.path.join(SAVE_TO, "test_outcome"))


def verify_level_1_ml(db):
    def l1ml_module_factory():
        return Level1MLModule()

    l1_df = db.extract(L1_SQL)
    verify_module(l1ml_module_factory, l1_df, "level_1",
                  os.path.join(SAVE_TO, "level_1_ml"))


def verify_level_1_symbolic(db):
    to_df = db.extract(TO_SQL)
    to_module = TestOutcomeModule(organisms=ORGANISMS)
    to_module.retrain(to_df)

    def l1s_module_factory():
        return Level1SymbolicModule(to_module)

    l1_df = db.extract(L1_SQL)
    verify_module(l1s_module_factory, l1_df, "level_1",
                  os.path.join(SAVE_TO, "level_1_symbolic"))


def verify_level_2(db):
    l1_df = db.extract(L1_SQL)
    l1ml_module = Level1MLModule()
    l1ml_module.retrain(l1_df)

    def l2_module_factory():
        return Level2Module(l1ml_module)

    l2_df = db.extract(L2_SQL)
    verify_module(l2_module_factory, l2_df, "level_2",
                  os.path.join(SAVE_TO, "level_2"))


if __name__ == "__main__":
    print("Started executing script.\n")
    start_time = datetime.now()

    logger = logging.getLogger(__name__)
    set_params(logger, from_root("log\\verify.log"))

    try:
        main()
    except Exception as e:
        logger.exception("verify.py: Fatal error")
        sys.exit(1)

    print(f"\nExecution time: {datetime.now() - start_time}")
    print("Finished executing script.")
