import pandas as pd
import sqlalchemy

from io_.fs import read_text


class Database:
    _instance = None

    @staticmethod
    def get_instance():
        """
        Returns the Database object representing the current database
        connection, creating a new database connection if none exists.
        :return: a Database object representing the current database connection
        """
        if Database._instance is None:
            Database._instance = Database()
        return Database._instance

    def __init__(self, server="SDDBSBI002", database="DSSG"):
        """
        Creates a new database connection to the given server and database.
        This class follows the singleton pattern. Do not manually call
        __init__; instead, call the static get_instance method.
        :param server: the name of the server to connect to
        :param database: the name of the database to connect to
        """

        # "trusted_connection=yes" tells SQL Server to use Windows
        # Authentication
        url = f"mssql+pyodbc://{server}/{database}"\
              + "?driver=ODBC+Driver+13+for+SQL+Server"\
              + "&trusted_connection=yes"
        self.engine = sqlalchemy.create_engine(url)

    def extract(self, sql_filepath):
        """
        Executes the SQL query saved at the given SQL file, returning the
        results in a DataFrame.
        :param sql_filepath: the absolute path to the SQL file containing the
        SQL query to execute
        :return: a DataFrame containing the results of executing the SQL query
        """
        sql = read_text(sql_filepath)
        df = pd.read_sql(sql, self.engine)
        return df

    def insert(self, df, table, schema):
        """
        Inserts the given DataFrame into the database table with the given name
        and schema. Rows are inserted in batches of 1000 at a time.
        Precondition: The DataFrame and database table have the same columns
        (table columns with DEFAULT constraints may optionally be missing from
        the DataFrame.) This method has undefined behaviour if this precondition
        is not met.
        :param df: the DataFrame to insert
        :param table: the name of the database table to insert to
        :param schema: the name of the database table to insert to
        :return: None
        """
        df.to_sql(table, self.engine, schema=schema,
                  if_exists="append", index=False, chunksize=1000)
