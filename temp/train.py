from datetime import datetime

from io_.db import Database
from modules.level_1_ml_module import Level1MLModule
from modules.level_1_symbolic_module import Level1SymbolicModule
from modules.level_2_module import Level2Module
from modules.test_outcome_module import TestOutcomeModule
from modules.test_performed_module import TestPerformedModule
from root import from_root


def main():
    db = Database.get_instance()

    train_test_performed(db)
    to_module = train_test_outcome(db)
    l1ml_module = train_level_1_machine_learning(db)
    train_level_1_symbolic(db, to_module)
    train_level_2(db, l1ml_module)


def train_test_performed(db):
    tp_df = db.extract(from_root("temp\\sql\\train_test_performed.sql"))
    tp_module = TestPerformedModule(organisms=True)
    tp_module.retrain(tp_df)
    tp_module.save_to_file(from_root("temp\\pkl\\test_performed_module.pkl"))


def train_test_outcome(db):
    to_df = db.extract(from_root("temp\\sql\\train_test_outcome.sql"))
    to_module = TestOutcomeModule(organisms=True)
    to_module.retrain(to_df)
    to_module.save_to_file(from_root("temp\\pkl\\test_outcome_module.pkl"))
    return to_module


def train_level_1_machine_learning(db):
    l1ml_df = db.extract(from_root("temp\\sql\\train_level_1_ml.sql"))
    l1ml_module = Level1MLModule()
    l1ml_module.retrain(l1ml_df)
    l1ml_module.save_to_file(from_root("temp\\pkl\\level_1_ml_module.pkl"))
    return l1ml_module


def train_level_1_symbolic(db, to_module):
    l1s_df = db.extract(from_root("sql\\train\\level_1.sql"))
    l1s_module = Level1SymbolicModule(to_module)
    l1s_module.retrain(l1s_df)
    l1s_module.save_to_file(from_root("temp\\pkl\\level_1_symbolic_module.pkl"))


def train_level_2(db, l1ml_module):
    l2_df = db.extract(from_root("sql\\train\\level_2.sql"))
    l2_module = Level2Module(l1ml_module)
    l2_module.retrain(l2_df)
    l2_module.save_to_file(from_root("temp\\pkl\\level_2_module.pkl"))


if __name__ == "__main__":
    print("Started executing script.")
    start_time = datetime.now()

    main()

    print(f"Execution time: {datetime.now() - start_time}")
    print("Finished executing script.")
