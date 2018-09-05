def ind_max(_list):
    """
    Gets the index of the maximum element in the given List. Returns -1 if the
    given List is empty.
    :param _list: the List to return the index of the maximum element of
    :return: the index of the maximum element in the given List, or -1 if the
    given List is empty
    """
    if not _list:
        return -1

    max_value = _list[0]
    max_index = 0
    for index, value in enumerate(_list):
        if value > max_value:
            max_value = value
            max_index = index
    return max_index
