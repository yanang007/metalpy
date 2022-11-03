from typing import Iterable


def stringify(e):
    return [str(i) for i in e]


def ensure_as_iterable(arg, excludes=None):
    if excludes is None:
        excludes = []
    elif not isinstance(excludes, Iterable):
        excludes = [excludes]

    if arg is None:
        return []
    for exclude in excludes:
        if isinstance(arg, exclude):
            return [arg]
    if not isinstance(arg, Iterable):
        return [arg]
