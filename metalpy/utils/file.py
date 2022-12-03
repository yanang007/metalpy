import os

from .type import ensure_as_iterable


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def ensure_filepath(path):
    dir, filename = os.path.split(path)
    ensure_dir(dir)
    return path


def locate_file(path, *predicates, mode='first'):
    """快速查找特定的文件
    :param path: 查找路径
    :param predicates: 判断谓词
    :param mode: 返回模式
        all - 返回全部符合条件的文件
        first - 返回第一个符合条件的文件
        last - 返回最后一个符合条件的文件
    :return: mode为all时，返回一个符合条件的完整文件路径列表（可能为空），否则返回一个完整文件路径或None
    """
    immediate_return = True
    ret = []
    if mode == 'all':
        immediate_return = False

    iterable = os.listdir(path)
    if mode == 'last':
        iterable = reversed(iterable)

    for file in iterable:
        flag = True
        for predicate in predicates:
            flag = flag and predicate(file)
            if not flag:
                break

        if flag:
            ret.append(file)
            if immediate_return:
                break

    ret = [os.path.join(path, file) for file in ret]
    if immediate_return:
        return ret[0] if len(ret) > 0 else None
    else:
        return ret


def locate_prefixed_file(path, prefix):
    return locate_file(path, lambda file: file.startswith(prefix))


def locate_file_by(paths, *, prefix=None, suffix=None, ext=None, contains=None, filename=None, mode='all'):
    preds = []

    if prefix is not None:
        prefix = ensure_as_iterable(prefix, str)
        preds.append(lambda f: any(map(lambda s: f.startswith(s), prefix)))

    if suffix is not None:
        suffix = ensure_as_iterable(suffix, str)
        preds.append(lambda f: any(map(lambda s: f.endswith(s), suffix)))

    if ext is not None:
        ext = ensure_as_iterable(ext, str)
        preds.append(lambda f: os.path.splitext(f)[1] in ext)

    if contains is not None:
        contains = ensure_as_iterable(contains, str)
        preds.append(lambda f: any(map(lambda s: s in f, contains)))

    if filename is not None:
        filename = ensure_as_iterable(filename, str)
        preds.append(lambda f: f in filename)

    ret = []

    immediate_return = True
    if mode == 'all':
        immediate_return = False

    paths = ensure_as_iterable(paths, str)
    filepaths = None
    for path in paths:
        filepaths = locate_file(path, *preds, mode=mode)
        if immediate_return:
            if filepaths is not None:
                break
        else:
            ret.extend(filepaths)

    if immediate_return:
        return filepaths
    else:
        return ret


def locate_first_file_by(paths, *, prefix=None, suffix=None, ext=None, contains=None, filename=None):
    return locate_file_by(paths, prefix=prefix, suffix=suffix, ext=ext, contains=contains, filename=filename,
                          mode='first')


def locate_last_file_by(paths, *, prefix=None, suffix=None, ext=None, contains=None, filename=None):
    return locate_file_by(paths, prefix=prefix, suffix=suffix, ext=ext, contains=contains, filename=filename,
                          mode='last')


def locate_files_by(paths, *, prefix=None, suffix=None, ext=None, contains=None, filename=None):
    return locate_file_by(paths, prefix=prefix, suffix=suffix, ext=ext, contains=contains, filename=filename,
                          mode='all')


def make_cache_file(name):
    cache_dir = './.cache'
    ret = os.path.join(cache_dir, name)
    ret = os.path.abspath(ret)
    ensure_filepath(ret)
    return ret


def make_cache_directory(name):
    cache_dir = './.cache'
    ret = os.path.join(cache_dir, name)
    ret = os.path.abspath(ret)
    ensure_dir(ret)
    return ret
