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

    return arg


def get_or_default(dictionary, key, _default=None, remove=False):
    if key in dictionary:
        ret = dictionary[key]
        if remove:
            del dictionary[key]
    else:
        ret = _default

    return ret


def pop_or_default(dictionary, key, _default=None):
    return get_or_default(
        dictionary=dictionary,
        key=key,
        _default=_default,
        remove=True
    )


def not_none_or_default(val, _default):
    if val is None:
        return _default
    else:
        return val
