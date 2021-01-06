import numpy


def compact_property(values):
    uniq = numpy.unique(values)
    prop2pid = dict(zip(uniq, range(uniq.size)))
    return prop2pid