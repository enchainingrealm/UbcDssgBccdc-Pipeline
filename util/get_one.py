def get_one(_set):
    """
    Gets an arbitrary element from the given Set.
    Precondition: the given Set is not empty.
    :param _set: the Set to get an arbitrary element from
    :return: an arbitrary element from the given Set
    """
    assert _set   # _set is not empty
    return next(iter(_set))
