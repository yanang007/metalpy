import importlib
import types

from .property_replacement import PropertyReplacement
from .replacement import Replacement


def create_replacement(func, orig, executor, name=None):
    if isinstance(func, property):
        # property不可以被构造新成员
        # 因此继承一个子类来实现引入原成员
        ret = PropertyReplacement(func, orig, executor)
    elif isinstance(executor.nest, type):
        # 类的成员函数必须是函数类型function才能在实例化时被识别并绑定self参数
        # 因此采用这个workaround来引入原函数
        func.repl_orig = orig
        func.repl_executor = executor
        if name is not None:
            func.__name__ = name
        else:
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
    if target is not None and not isinstance(target, (type, type(types))):
        # 目标不是None、类型与模块，则是实例，func是实例方法，直接绑定
        wrapper = types.MethodType(func, target)
        is_target_method = True
    else:
        wrapper = func
        is_target_method = False

    return wrapper, is_target_method


def update_params(func, args, kwargs, new_kwargs):
    """使用new_kwargs更新func的args和kwargs

    Parameters
    ----------
    func
        目标函数签名
    args
        目标函数原有的位置参数
    kwargs
        目标函数原有的关键字参数
    new_kwargs
        新的参数列表，使用str修改关键字或非varargs的位置参数，使用int修改varargs中指定位置的参数

    Returns
    -------
        tuple(updated_args, updated_kwargs)
        更新后的args和kwargs
    """
    import inspect
    from metalpy.utils.type import get_or_default, not_none_or_default

    # func可能被替换过，导致函数签名和原函数不一致，因此尝试回溯原函数从而获取真实函数签名
    # get_orig在不存在原函数时会返回None，因此需要判断
    func = not_none_or_default(get_orig(func), _default=func)
    arg_spec = inspect.getfullargspec(func)
    updated_args = None
    updated_kwargs = None

    arg_spec_args = arg_spec.args
    if isinstance(func, types.MethodType):
        # func是实例方法，需要排除掉self参数
        arg_spec_args = arg_spec_args[1:]

    for i, arg_name in enumerate(arg_spec_args):
        new_arg = get_or_default(new_kwargs, arg_name, _default=None)
        if new_arg is not None:
            if i < len(args):
                # 如果下标在args范围内，则一定是对应位置的参数
                if updated_args is None:
                    updated_args = list(args)
                updated_args[i] = new_arg
            else:
                # 如果下标不在args范围内，则原参数没有被指定或在kwargs中指定，无论哪种情况都应放到kwargs中
                if updated_kwargs is None:
                    updated_kwargs = dict(kwargs)
                updated_kwargs[arg_name] = new_arg

    for i in range(len(args)):
        new_arg = get_or_default(new_kwargs, i, _default=None)
        if new_arg is not None:
            # 通过下标替换args中的参数，主要用于varargs参数
            if updated_args is None:
                updated_args = list(args)
            updated_args[i] = new_arg

    for arg_name in arg_spec.kwonlyargs:
        new_arg = get_or_default(new_kwargs, arg_name, _default=None)
        if new_arg is not None:
            # 通过关键字替换kwargs中的kwonlyargs
            if updated_kwargs is None:
                updated_kwargs = dict(kwargs)
            updated_kwargs[arg_name] = new_arg

    if updated_args is None:
        updated_args = args
    if updated_kwargs is None:
        updated_kwargs = kwargs

    return updated_args, updated_kwargs


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
    if i == -1:
        return '', module_path
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
    if module_path == '':
        return importlib.import_module(object_name)
    elif object_name == '':
        return importlib.import_module(module_path)
    else:
        return getattr(importlib.import_module(module_path), object_name)


def get_parent(obj):
    """获取对象所在的类或模块

    Parameters
    ----------
    obj
        对象

    Returns
    -------
        方法所在的类
    """
    if isinstance(obj, types.MethodType):
        return obj.__self__
    else:
        cls_name = split_object_path(get_ancestor(obj).__qualname__)[0]
        module_path = obj.__module__
        return get_object_by_path('.'.join((module_path, cls_name)))
