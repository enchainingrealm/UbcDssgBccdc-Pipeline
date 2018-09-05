from datetime import datetime

from io_.fs import write_df
from modules.level_1_ml_module import Level1MLModule
from modules.level_1_symbolic_module import Level1SymbolicModule
from modules.level_2_module import Level2Module
from modules.test_outcome_module import TestOutcomeModule
from modules.test_performed_module import TestPerformedModule
from root import from_root
from io_.db import Database


def main():
    tp_module, to_module, l1ml_module, l1s_module, l2_module = load_modules()

    db = Database.get_instance()

    classify_nih(db, tp_module, to_module, l1ml_module, l1s_module, l2_module)
    classify_random(db, to_module, l1ml_module, l1s_module, l2_module)
    classify_culture(db, to_module, l1ml_module, l1s_module, l2_module)


def load_modules():
    # Test performed
    tp_module = TestPerformedModule.load_from_file(
        from_root("temp\\pkl\\test_performed_module.pkl"))

    # Test outcome
    to_module = TestOutcomeModule.load_from_file(
        from_root("temp\\pkl\\test_outcome_module.pkl"))

    # Level 1 (machine learning)
    l1ml_module = Level1MLModule.load_from_file(
        from_root("temp\\pkl\\level_1_ml_module.pkl"))

    # Level 1 (symbolic)
    l1s_module = Level1SymbolicModule(to_module).load_from_file(
        from_root("temp\\pkl\\level_1_symbolic_module.pkl"))

    # Level 2
    l2_module = Level2Module(l1ml_module).load_from_file(
        from_root("temp\\pkl\\level_2_module.pkl"))

    print("Finished loading modules.")
    return tp_module, to_module, l1ml_module, l1s_module, l2_module


def classify_nih(db, tp_module, to_module, l1ml_module, l1s_module, l2_module):
    nih_df = db.extract(from_root("temp\\sql\\test_nih.sql"))
    keys = ["test_key", "result_key"]

    nih_tp_results = tp_module.classify(nih_df)
    nih_to_results = to_module.classify(nih_df)
    nih_l1ml_results = l1ml_module.classify(nih_df)
    nih_l1s_results = l1s_module.classify(nih_df)
    nih_l2_results = l2_module.classify(nih_df)

    nih_results = nih_tp_results\
        .merge(nih_to_results, how="inner", on=keys)\
        .merge(nih_l1ml_results, how="inner", on=keys)\
        .merge(nih_l1s_results, how="inner", on=keys)\
        .merge(nih_l2_results, how="inner", on=keys)

    db.insert(nih_results, "tmp_nih_predictions", "dbo")
    write_df(from_root("temp\\predictions\\nih.csv"), nih_results)


def classify_random(db, to_module, l1ml_module, l1s_module, l2_module):
    random_df = db.extract(from_root("temp\\sql\\test_random.sql"))
    keys = ["test_key", "result_key"]

    random_to_results = to_module.classify(random_df)
    random_l1ml_results = l1ml_module.classify(random_df)
    random_l1s_results = l1s_module.classify(random_df)
    random_l2_results = l2_module.classify(random_df)

    random_results = random_to_results\
        .merge(random_l1ml_results, how="inner", on=keys)\
        .merge(random_l1s_results, how="inner", on=keys)\
        .merge(random_l2_results, how="inner", on=keys)

    db.insert(random_results, "tmp_random_predictions", "dbo")
    write_df(from_root("temp\\predictions\\random.csv"), random_results)


def classify_culture(db, to_module, l1ml_module, l1s_module, l2_module):
    culture_df = db.extract(from_root("temp\\sql\\test_culture.sql"))
    keys = ["test_key", "result_key"]

    culture_to_results = to_module.classify(culture_df)
    culture_l1ml_results = l1ml_module.classify(culture_df)
    culture_l1s_results = l1s_module.classify(culture_df)
    culture_l2_results = l2_module.classify(culture_df)

    culture_results = culture_to_results\
        .merge(culture_l1ml_results, how="inner", on=keys)\
        .merge(culture_l1s_results, how="inner", on=keys)\
        .merge(culture_l2_results, how="inner", on=keys)

    db.insert(culture_results, "tmp_culture_predictions", "dbo")
    write_df(from_root("temp\\predictions\\culture.csv"), culture_results)


if __name__ == "__main__":
    print("Started executing script.")
    start_time = datetime.now()

    main()

    print(f"Execution time: {datetime.now() - start_time}")
    print("Finished executing script.")
