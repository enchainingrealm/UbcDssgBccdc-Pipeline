import logging
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


def main():
    db = Database.get_instance()

    # ==========================================================================
    # Test performed

    tp_df = db.extract(from_root("sql\\train\\test_performed.sql"))

    tp_module = TestPerformedModule()
    tp_module.retrain(tp_df)
    tp_module.save_to_file(from_root("pkl\\test_performed_module.pkl"))

    tp_module_org_false = TestPerformedModule(organisms=False)
    tp_module_org_false.retrain(tp_df)
    tp_module_org_false.save_to_file(
        from_root("pkl\\test_performed_organisms_false_module.pkl"))

    # ==========================================================================
    # Test outcome

    to_df = db.extract(from_root("sql\\train\\test_outcome.sql"))

    to_module = TestOutcomeModule()
    to_module.retrain(to_df)
    to_module.save_to_file(from_root("pkl\\test_outcome_module.pkl"))

    to_module_org_false = TestOutcomeModule(organisms=False)
    to_module_org_false.retrain(to_df)
    to_module_org_false.save_to_file(
        from_root("pkl\\test_outcome_organisms_false_module.pkl"))

    # ==========================================================================
    # Level 1

    l1_df = db.extract(from_root("sql\\train\\level_1.sql"))

    # Machine learning
    l1ml_module = Level1MLModule()
    l1ml_module.retrain(l1_df)
    l1ml_module.save_to_file(from_root("pkl\\level_1_ml_module.pkl"))

    # Symbolic
    l1s_module = Level1SymbolicModule(to_module)
    l1s_module.retrain(l1_df)
    l1s_module.save_to_file(from_root("pkl\\level_1_symbolic_module.pkl"))

    # ==========================================================================
    # Level 2

    l2_df = db.extract(from_root("sql\\train\\level_2.sql"))
    l2_module = Level2Module(l1s_module)
    l2_module.retrain(l2_df)
    l2_module.save_to_file(from_root("pkl\\level_2_module.pkl"))


if __name__ == "__main__":
    print("Started executing script.\n")
    start_time = datetime.now()

    logger = logging.getLogger(__name__)
    set_params(logger, from_root("log\\train.log"))

    try:
        main()
    except Exception as e:
        logger.exception("train.py: Fatal error")
        sys.exit(1)

    print(f"\nExecution time: {datetime.now() - start_time}")
    print("Finished executing script.")
