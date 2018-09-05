import logging
import sys
from datetime import datetime

from io_.db import Database
from root import from_root
from util.logger import set_params
from util.tagger import annotate


SQL_FILEPATH = from_root("sql\\needs_tagging.sql")
TABLE = "metamap"
SCHEMA = "dbo"

OBSERVATIONS = False


def main():
    db = Database.get_instance()
    df = db.extract(SQL_FILEPATH)

    annotations = annotate(df, observations=OBSERVATIONS)
    db.insert(annotations, TABLE, SCHEMA)


if __name__ == "__main__":
    print("Started executing script.\n")
    start_time = datetime.now()

    logger = logging.getLogger(__name__)
    set_params(logger, from_root("log\\metamap.log"))

    try:
        main()
    except Exception as e:
        logger.exception("metamap.py: Fatal error")
        sys.exit(1)

    print(f"\nExecution time: {datetime.now() - start_time}")
    print("Finished executing script.")
