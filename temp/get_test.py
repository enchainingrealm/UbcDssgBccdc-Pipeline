from datetime import datetime

import pandas as pd

from io_.fs import write_df
from root import from_root
from io_.db import Database
from util.tagger import annotate


def main():
    db = Database.get_instance()

    get_nih(db)
    get_random(db)
    get_culture(db)


def get_nih(db):
    nih_df = pd.read_csv(from_root("temp\\test_data\\nih_cleaned.csv"))
    db.insert(nih_df, "tmp_nih", "dbo")

    nih_metamap_df = annotate(nih_df)
    db.insert(nih_metamap_df, "tmp_nih_metamap", "dbo")
    write_df(from_root("temp\\test_data\\nih_metamap.csv"), nih_metamap_df)


def get_random(db):
    random_df = db.extract(from_root("temp\\sql\\get_test_random.sql"))
    random_df = random_df.sample(n=100)

    db.insert(random_df, "tmp_random", "dbo")
    write_df(from_root("temp\\test_data\\random.csv"), random_df)


def get_culture(db):
    culture_df = db.extract(from_root("temp\\sql\\get_test_culture.sql"))
    culture_df = culture_df.sample(n=100)

    db.insert(culture_df, "tmp_culture", "dbo")
    write_df(from_root("temp\\test_data\\culture.csv"), culture_df)


if __name__ == "__main__":
    print("Started executing script.")
    start_time = datetime.now()

    main()

    print(f"Execution time: {datetime.now() - start_time}")
    print("Finished executing script.")
