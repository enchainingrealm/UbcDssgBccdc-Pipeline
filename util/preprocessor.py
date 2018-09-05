import json
import re


def preprocess(df, organisms=False):
    """
    Preprocesses the data in the given DataFrame.
    Preprocesses result_full_descriptions:
    - Converts all result_full_descriptions to lowercase
    - Removes all characters that are not letters, numbers, spaces, or pipes
    - Replaces all purely-numeric words with "_NUMBER_"
    - If organisms is True, replaces all organism names with "_ORGANISM_"
    Preprocesses labels:
    - Converts all labels in the label columns {"test_performed",
      "test_outcome", "level_1", "level_2"} in the given DataFrame to lowercase.
      Skips a label column if it does not exist in the DataFrame.
    :param df: the DataFrame to preprocess
    - required columns: {"result_full_description", "candidates" (if organisms
      is True)}
    - optional columns: {"test_performed", "test_outcome", "level_1", "level_2"}
    :param organisms: whether to replace organism names in the
    result_full_descriptions with "_ORGANISM_"
    :return: the preprocessed DataFrame
    - columns: the same as the columns of the given DataFrame
    """
    df = df.copy()   # don't mutate the original DataFrame

    df["result_full_description"] = df["result_full_description"].apply(
        lambda rfd: replace_numbers(remove_symbols(rfd.lower()))
    )

    if organisms:
        def helper(row):
            return replace_organisms(
                row["result_full_description"],
                row["candidates"]
            )

        df["result_full_description"] = df.apply(helper, axis=1)

    df = labels_to_lowercase(df)
    return df


def remove_symbols(result_full_description):
    """
    Removes all characters that are not letters, numbers, spaces, or pipes from
    the given string.
    :param result_full_description: the string to remove symbols from
    :return: the string after removing symbols
    """
    return re.sub(r"[^a-zA-Z0-9 |]", "", result_full_description)


def replace_numbers(result_full_description):
    """
    Replaces all purely-numeric words in the given string with "_NUMBER_".
    :param result_full_description: the string to replace numeric words in
    :return: the string after replacing numeric words
    """
    raw_tokens = result_full_description.split()
    tokens = [
        "_NUMBER_" if all(char.isdigit() for char in token) else token
        for token in raw_tokens
    ]
    return " ".join(tokens)


def replace_organisms(result_full_description, candidates_str):
    """
    Replaces all organism names in the given result_full_description string with
    "_ORGANISM_".
    :param result_full_description: the string to replace organism names in
    :param candidates_str: a JSON string containing MetaMap candidates
    information
    :return: the result_full_description string after replacing organism names
    """
    result = result_full_description

    candidates_dict = json.loads(candidates_str)
    matchings = [text.lower() for _, value in candidates_dict.items()
                 for text in value["matched"]]
    matchings.sort(key=len, reverse=True)

    for text in matchings:
        result = result.replace(text.lower(), "_ORGANISM_")

    return result


def labels_to_lowercase(df):
    """
    Converts all labels in the label columns {"test_performed", "test_outcome",
    "level_1", "level_2"} in the given DataFrame to lowercase. Skips a label
    column if it does not exist in the DataFrame.
    :param df: the DataFrame to convert all labels to lowercase
    - optional columns: {"test_performed", "test_outcome", "level_1", "level_2"}
    :return: the DataFrame after converting all labels to lowercase
    - columns: the same as the columns of the given DataFrame
    """
    df = df.copy()   # don't mutate the original DataFrame

    outputs = ["test_performed", "test_outcome", "level_1", "level_2"]
    for output in outputs:
        if output in df.columns:
            df[output] = df[output].apply(str.lower)

    return df
