def get_keys(observations):
    """
    Gets the names of the key columns used in the test DataFrames throughout the
    Pipeline:
    - ["test_key", "result_key"] if the DataFrame rows are at the test level
    - ["test_key", "result_key", "obs_seq_nbr"] if the DataFrame rows are at the
      observation level
    :param observations: False if the rows in the DataFrame are at the test
    level; True if the rows in the DataFrame are at the observation level
    :return: a List of key column names
    """
    return ["test_key", "result_key"]\
           + (["obs_seq_nbr"] if observations else [])
