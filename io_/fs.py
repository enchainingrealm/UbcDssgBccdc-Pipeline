import os


def read_text(filepath):
    """
    Reads the text file at the given file path, returning its contents as a
    string.
    :param filepath: the absolute path to the text file to read
    :return: the contents of the text file
    """
    with open(filepath, "r") as file:
        return file.read()


def write_text(filepath, text):
    """
    Writes the given text to the file at the given path, overwriting the file if
    it already exists.
    :param filepath: the absolute path to the text file to write to
    :param text: the text to write
    :return: None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as file:
        file.write(text)


def write_df(filepath, df):
    """
    Writes the given DataFrame to the CSV file at the given path, overwriting
    the file if it already exists.
    :param filepath: the absolute path to the CSV file to write to
    :param df: the DataFrame to write
    :return: None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


def write_plot(filename, plt):
    """
    Writes the given matplotlib plot to the image file at the given path,
    overwriting the file if it already exists.
    :param filename: the absolute path to the image file to write to
    :param plt: the matplotlib plot to write
    :return: None
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
