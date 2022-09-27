import types

from abc import ABC, abstractmethod


class Replacement:
    def __init__(self, func, orig, executor):
        self.func = func
        self.repl_orig = orig
        self.repl_executor = executor
        self.__name__ = None if orig is None else orig.__name__

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class PropertyReplacement(property):
    def __init__(self, prop: property, orig, executor):
        super().__init__(prop.fget, prop.fset, prop.fdel, prop.__doc__)
        self.repl_orig = orig
        self.repl_executor = executor
        self.__name__ = None if orig is None else orig.__name__


def create_replacement(func, orig, executor):
    if isinstance(func, property):
        # property不可以被构造新成员
        # 因此继承一个子类来实现引入原成员
        ret = PropertyReplacement(func, orig, executor)
    elif orig is not None and is_init(orig):
        # 构造函数必须是函数类型function才能被super(...)识别并绑定self参数
        # 因此采用这个workaround来引入原函数
        func.repl_orig = orig
        func.repl_executor = executor
        func.__name__ = orig.__name__
        ret = func
    else:
        ret = Replacement(func, orig, executor)

    return ret


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


def is_init(repl):
    return repl.__name__ == '__init__'


def is_or_is_replacement(obj, other):
    obj = obj if not hasattr(obj, 'repl_orig') else getattr(obj, 'repl_orig')
    return obj == other


def wrap_method_with_target(target, func):
    if not isinstance(target, type):
        wrapper = types.MethodType(func, target)
    else:
        wrapper = func

    return wrapper


class RecoverableInjector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, func):
        pass

    @abstractmethod
    def rollback(self):
        pass


class extends(RecoverableInjector):
    def __init__(self, target, name):
        super().__init__()
        self.target = target
        self.name = name

    def __call__(self, func):
        wrapper = wrap_method_with_target(self.target, func)
        wrapper = create_replacement(wrapper, None, self)
        cmd = f'self.target.{self.name} = wrapper'
        exec(cmd)
        return wrapper

    def rollback(self):
        cmd = f'del self.target.{self.name}'
        exec(cmd)


class hijacks(RecoverableInjector):
    def __init__(self, prefix, target):
        super().__init__()
        self.prefix = prefix
        self.target = target
        self.backup = eval(f'self.prefix.{self.target}')

    def __call__(self, func):
        wrapper = create_replacement(func, self.backup, self)
        cmd = f'self.prefix.{self.target} = wrapper'
        exec(cmd)
        return func

    def rollback(self):
        cmd = f'self.prefix.{self.target} = self.backup'
        exec(cmd)


class replaces(RecoverableInjector):
    def __init__(self, target, name, keep_orig=True):
        super().__init__()
        self.target = target
        self.name = name
        self.backup = None
        self.keep_orig = keep_orig

    def __call__(self, func):
        self.backup = orig = getattr(self.target, self.name)
        if self.keep_orig:
            wrapper = lambda *args, **kwargs: func(*args, **kwargs, orig_fn=orig)
        else:
            wrapper = lambda *args, **kwargs: func(*args, **kwargs)

        wrapper = wrap_method_with_target(self.target, wrapper)
        wrapper = create_replacement(wrapper, orig, self)
        cmd = f'self.target.{self.name} = wrapper'
        exec(cmd)

        return wrapper

    def rollback(self):
        cmd = f'self.target.{self.name} = self.backup'
        exec(cmd)


class before(replaces):
    def __init__(self, target, name):
        super().__init__(target, name)

    def __call__(self, func):
        is_method = not isinstance(self.target, type)
        func = wrap_method_with_target(self.target, func)

        def wrapper(*args, orig_fn=None, **kwargs):
            if is_method:
                _self = args[0]
                args = args[1:]
            func(*args, **kwargs)
            return orig_fn(*args, **kwargs)

        wrapper = wrap_method_with_target(self.target, wrapper)
        return super().__call__(wrapper)


class after(replaces):
    def __init__(self, target, name):
        super().__init__(target, name)

    def __call__(self, func):
        is_method = not isinstance(self.target, type)
        func = wrap_method_with_target(self.target, func)

        def wrapper(*args, orig_fn=None, **kwargs):
            if is_method:
                _self = args[0]
                args = args[1:]
            result = orig_fn(*args, **kwargs)
            func(*args, **kwargs)
            return result

        return super().__call__(wrapper)
