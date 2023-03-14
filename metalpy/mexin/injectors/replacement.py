import functools

from metalpy.utils.object_path import reassign_object_name, reassign_object_module


def create_replacement(func, orig, executor, qualname=None, module_name=None):
    if isinstance(func, property):
        # property不可以被构造新成员
        # 因此继承一个子类来实现引入原成员
        ret = PropertyReplacement(func, orig, executor)
    elif isinstance(executor.nest, type):
        # 类的成员函数必须是函数类型function才能在实例化时被识别并绑定self参数
        # 因此采用这个workaround来引入原函数
        func.repl_orig = orig
        func.repl_executor = executor
        ret = func
    else:
        ret = Replacement(func, orig, executor)

    if orig is None:
        if qualname is not None:
            reassign_object_name(ret, new_qualname=qualname)
        if module_name is not None:
            reassign_object_module(ret, module_name)
    else:
        functools.wraps(orig)(ret)

    return ret


class TemporaryReversion:
    def __init__(self, *repls):
        self.repls = repls
        self.replacements = []

    def __enter__(self):
        for repl in self.repls:
            self.replacements.append(repl.func)
            repl.repl_executor.rollback()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for repl, replacement in zip(self.repls, self.replacements):
            repl.repl_executor(replacement)


def reverted(*repls):
    """临时还原某些替换
    例如类名劫持，有些类在调用基类构造函数时会采用类似于super(<class-name>, self).__init__(...)的形式，
    导致在被劫持状态下无法正常构造，这时需要进行临时的revert操作
    """
    return TemporaryReversion(*repls)


def get_orig(repl):
    if hasattr(repl, 'repl_orig'):
        return repl.repl_orig
    else:
        return None


def get_ancestor(repl):
    prev = repl
    orig = get_orig(repl)
    while orig is not None:
        prev = orig
        orig = get_orig(orig)
    return prev


def is_or_is_replacement(obj, other):
    return get_ancestor(obj) == get_ancestor(other)


class Replacement:
    def __init__(self, func, orig, executor):
        self.func = func
        self.repl_orig = orig
        self.repl_executor = executor

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class PropertyReplacement(property):
    def __init__(self, prop: property, orig, executor):
        super().__init__(prop.fget, prop.fset, prop.fdel, prop.__doc__)
        self.repl_orig = orig
        self.repl_executor = executor

