import logging
import os


def set_params(logger, filepath):
    """
    Initializes the given logger.
    Adds timestamps to the logged messages and sets the logger to print logged
    messages to console and save them to the given file.
    :param logger: a Python logger object
    :param filepath: the absolute path to the log file to save messages to
    :return: None
    """
    formatter = logging.Formatter("[%(asctime)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
