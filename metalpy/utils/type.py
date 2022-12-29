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


def get_or_default(dictionary, key, _default=None, supplier=None, remove=False):
    if key in dictionary:
        ret = dictionary[key]
        if remove:
            del dictionary[key]
    else:
        if supplier is not None:
            ret = supplier()
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


def not_none_or_default(val, _default=None, supplier=None):
    if val is None:
        if _default is None:
            _default = supplier()
        return _default
    else:
        return val


def get_params_dict(**kwargs):
    return dict(kwargs)


def get_params(*args, **kwargs):
    return list(args), dict(kwargs)


def get_full_qualified_class_name(type_or_instance):
    if not isinstance(type_or_instance, type):
        type_or_instance = type(type_or_instance)

    return '.'.join((type_or_instance.__module__, get_class_name(type_or_instance)))


def get_class_name(type_or_instance):
    if not isinstance(type_or_instance, type):
        type_or_instance = type(type_or_instance)

    return type_or_instance.__name__
