import importlib
import inspect
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


def get_object_from(source, path: str):
    """用于进行从module或者其他对象中提取类似 'xxx.yyy.zzz' 路径的对象

    Parameters
    ----------
    source
        源，比如module或class
    path
        待获取的对象路径

    Returns
    -------
        获取到的对象，或None
    """
    ret = source

    if path == '':
        return ret

    for seg in path.split('.'):
        ret = getattr(ret, seg, None)
        if ret is None:
            break

    return ret


def get_nest_path_by_qualname(obj):
    """通过解析__qualname__提取对象在模块中的嵌套路径，如果是模块下直接定义的对象，则返回空字符串

    Parameters
    ----------
    obj
        待获取嵌套路径的对象，可以为type、类实例或方法以及其他具有__qualname__的对象

    Returns
    -------
        如果目标属于某嵌套定义下，则返回__qualname__中提取的嵌套路径（不包含对象本身）
        如果是模块下直接定义的对象，则返回空字符串
    """
    segments = obj.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)
    return '.'.join(segments[:-1])


def get_class_that_defined_method(meth):
    """获取定义指定方法的类

    Parameters
    ----------
    meth
        方法实例

    Returns
    -------
        定义meth的类，或者None，代表该方法无法找到

    Notes
    -----
        由于Python3移除了所有类和类中定义的方法的直接关联，因此只能通过一些方法来“猜测”，所以理论上存在无法获取到正确结果的情况

        比如如果随意修改__qualname__或者__module__的值会导致结果不可预测

    References
    ----------
        By @Yoel and with assistance from comments

        "Get class from meth.__globals__" patch by @Alexander_McFarlane

        https://stackoverflow.com/questions/3589311/get-defining-class-of-unbound-method-object-in-python-3/

        Modified to support statically nested class using 'get_object_from'
    """
    import functools
    if isinstance(meth, functools.partial):
        return get_class_that_defined_method(meth.func)
    if inspect.ismethod(meth) or (inspect.isbuiltin(meth) and
                                  getattr(meth, '__self__', None) is not None and
                                  getattr(meth.__self__, '__class__', None)):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, '__func__', meth)  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        class_path = get_nest_path_by_qualname(meth)
        cls = get_object_from(inspect.getmodule(meth), class_path)
        if cls is None:
            path_segs = class_path.split('.')
            cls = get_object_from(meth.__globals__.get(path_segs[0]), '.'.join(path_segs))
        if isinstance(cls, type):
            return cls
    return getattr(meth, '__objclass__', None)  # handle special descriptor objects


def get_nest(obj):
    """获取对象所在的类或模块

    Parameters
    ----------
    obj
        对象

    Returns
    -------
        方法所在的类

    Notes
    -----
        由于Python3移除了所有类和类中定义的方法以及和模块的直接关联，因此只能通过一些方法来“猜测”，所以理论上存在无法获取到正确结果的情况

        比如如果随意修改__qualname__或者__module__的值会导致结果不可预测
    """
    obj = get_ancestor(obj)
    if inspect.ismethod(obj):
        return obj.__self__
    elif inspect.isclass(obj):
        module = inspect.getmodule(obj)
        path = get_nest_path_by_qualname(obj)
        return get_object_from(module, path)
    else:
        ret = get_class_that_defined_method(obj)
        if ret is None:
            if obj.__qualname__ == obj.__name__:
                # 猜测定义在模块中
                ret = inspect.getmodule(obj)

        return ret
