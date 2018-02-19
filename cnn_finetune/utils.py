import functools
import operator


default = object()


def product(iterable):
    return functools.reduce(operator.mul, iterable)
