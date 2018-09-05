import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def from_root(path):
    """
    Converts a path relative to the project root to the equivalent absolute
    path.
    :param path: the path relative to the project root to convert
    :return: the absolute path equivalent to the given relative path
    """
    return os.path.join(ROOT, path)
