import numpy


def compact_property(values, return_order=False):
    uniq = numpy.unique(values)
    prop2pid = dict(zip(uniq, range(uniq.size)))
    if return_order:
        return prop2pid, uniq.tolist()
    return prop2pid