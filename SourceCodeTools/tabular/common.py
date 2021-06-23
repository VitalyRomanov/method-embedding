import numpy


def compact_property(values, return_order=False, index_from_one=False):
    """
    Returns a map from the original value to the compact index
    :param values:
    :param return_order:
    :param index_from_one:
    :return:
    """
    uniq = numpy.unique(values)
    if index_from_one:
        index = range(1, uniq.size + 1)
    else:
        index = range(uniq.size)
    prop2pid = dict(zip(uniq, index))
    if return_order:
        inv_index = uniq.tolist()
        if index_from_one:
            inv_index.insert(0, "NA")
        return prop2pid, inv_index
    return prop2pid