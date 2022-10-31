import os


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def ensure_filepath(path):
    dir, filename = os.path.split(path)
    ensure_dir(dir)
    return path


def locate_file(path, predicate):
    for file in os.listdir(path):
        if predicate(file):
            return os.path.join(path, file)


def locate_prefixed_file(path, prefix):
    return locate_file(path, lambda file: file.startswith(prefix))


def make_cache_file(name):
    cache_dir = './.cache'
    ret = os.path.join(cache_dir, name)
    ensure_filepath(ret)
    return ret


def make_cache_directory(name):
    cache_dir = './.cache'
    ret = os.path.join(cache_dir, name)
    ensure_dir(ret)
    return ret
