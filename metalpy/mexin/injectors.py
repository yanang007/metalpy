import importlib
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
    elif isinstance(executor.target, type):
        # 类的成员函数必须是函数类型function才能在实例化时被识别并绑定self参数
        # 因此采用这个workaround来引入原函数
        func.repl_orig = orig
        func.repl_executor = executor
        func.__name__ = orig.__name__
        ret = func
    else:
        ret = Replacement(func, orig, executor)

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
    obj = obj if not hasattr(obj, 'repl_orig') else getattr(obj, 'repl_orig')
    return obj == other


def wrap_method_with_target(target, func):
    if not isinstance(target, type):  # 目标是实例，直接绑定
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
        wrapper = None

        if hasattr(self.target, '__init__'):
            # TODO (low-priority): 也许可以通过修改super函数实现
            def reverted_wrapper(*args, **kwargs):
                # 用于保证类名在被劫持状态下仍然能进行构造
                with reverted(wrapper):
                    ret = func(*args, **kwargs)
                return ret
            repl = reverted_wrapper
        else:
            repl = func

        wrapper = create_replacement(repl, self.backup, self)
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


class FunctionTermination:
    """指示提前终止函数，目前仅在before注解中使用，指示提前终止函数
    """
    def __init__(self, *rets):
        if len(rets) == 0:
            self.ret = None
        elif len(rets) == 1:
            self.ret = rets[0]
        else:
            self.ret = rets


def terminate():
    return FunctionTermination()


def terminate_with(*rets):
    return FunctionTermination(*rets)


class before(replaces):
    """将代码注入到目标函数开始前执行
    return terminate() 或 return terminate_with(*args) 可以指示提前终止函数
    """
    def __init__(self, target, name):
        super().__init__(target, name)

    def __call__(self, func):
        is_method = not isinstance(self.target, type)
        func = wrap_method_with_target(self.target, func)

        def wrapper(*args, orig_fn=None, **kwargs):
            if is_method:
                _self = args[0]
                args = args[1:]
            ret = func(*args, **kwargs)
            if isinstance(ret, FunctionTermination):
                return ret.ret
            return orig_fn(*args, **kwargs)

        wrapper = wrap_method_with_target(self.target, wrapper)
        return super().__call__(wrapper)


class after(replaces):
    """将代码注入到目标函数返回后执行
    返回非空的值可以替换原函数的返回值
    """
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
            ret = func(*args, **kwargs)
            if ret is not None:
                result = ret
            return result

        return super().__call__(wrapper)


def split_object_path(module_path: str):
    """ 分割模块成员的全限定路径
    Parameters
    ----------
    module_path
        对象的全限定路径
    Returns
    -------
        (对象所在模块路径，对象名)

    Examples
    --------
    >>> split_object_path('xxx.yyy.zzz.Target')
    ('xxx.yyy.zzz', 'Target')
    """
    i = module_path.rfind('.')
    return module_path[:i], module_path[i+1:]


def get_class_path(cls):
    """从类目标获取目标的全限定路径
    Parameters
    ----------
    cls
        类目标
    Returns
    -------
        类目标的全限定路径
    Examples
    --------
    >>> get_class_path(xxx.yyy.zzz.Target)
    'xxx.yyy.zzz.Target'
    """
    return str(cls)[8:-2]


def get_object_by_path(path):
    """通过全限定路径获取目标
    Parameters
    ----------
    path
        目标的全限定路径
    Returns
    -------
        全限定路径所标识的目标
    Examples
    --------
    >>> get_object_by_path('xxx.yyy.zzz.Target')
    <class 'xxx.yyy.zzz.Target'>
    """
    module_path, object_name = split_object_path(path)
    return getattr(importlib.import_module(module_path), object_name)
