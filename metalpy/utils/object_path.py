import importlib
import inspect
import sys
import warnings
from typing import Optional, Union

from metalpy.utils.dhash import dhash

module_type = type(warnings)


class DottedName:
    def __init__(self, *paths):
        """用于表示用'.'分割的Python对象路径

        Parameters
        ----------
        paths
            一组路径，会被'.'连接
        """
        self.parts = [seg for path in paths for seg in DottedName._split_path(path)]

    def without_prefix(self, *prefixes):
        """获取当前路径移除指定前缀后的路径

        Parameters
        ----------
        prefixes
            路径

        Returns
        -------
        ret
            按顺序依次匹配prefixes中前缀，如果匹配，则移除，否则返回当前结果

        Examples
        --------
        >>> DottedName('x.y.z').without_prefix('x.y', 'z')
        <<< DottedName('')

        >>> DottedName('x.y.z').without_prefix('x', 'z.y')
        <<< DottedName('y.z')  # only 'x' matches the prefix, and is removed
        """
        trimed = 0
        if len(prefixes) == 0:
            trimed = 1
        else:
            for prefix in prefixes:
                prefix = DottedName(prefix)
                matched = True
                for i, rhs in enumerate(prefix.parts):
                    if self.parts[trimed + i] != rhs:
                        matched = False
                        break
                if matched:
                    trimed += len(prefix)
                else:
                    break

        ret = DottedName()
        ret.parts = self.parts[trimed:]
        return ret

    def suffix(self, length=1):
        """获取当前路径中指定长度的后缀

        Parameters
        ----------
        length
            需要获取的后缀长度

        Returns
        -------
        ret
            当前路径的后缀数组，长度小于等于length

        Examples
        --------
        >>> DottedName('x.y.z').suffix()
        <<< DottedName('z')
        """
        ret = DottedName()
        length = min(self.__len__(), length)
        ret.parts = self.parts[-length:]

        return ret

    @property
    def empty(self):
        return self.__len__() == 0

    def __getitem__(self, item):
        return self.parts[item]

    def __setitem__(self, key, value):
        if key < 0:
            key = len(self.parts) + key

        if isinstance(value, str):
            self.__setitem__(key, DottedName(value))
        elif isinstance(value, (list, tuple, DottedName)):
            self.parts = [*self.parts[0:key], *value, *self.parts[key + 1:]]
        else:
            warnings.warn('Result of assigning a non-str object to be part of DottedName may be unexpected.')
            self.__setitem__(key, DottedName(str(value)))

    def __iter__(self):
        for part in self.parts:
            yield part

    def __len__(self):
        return len(self.parts)

    def __eq__(self, other):
        if not isinstance(other, DottedName):
            return False
        return self.parts == other.parts

    def __objpath__(self):
        return '.'.join(self.parts)

    def __str__(self):
        return self.__objpath__()

    def __repr__(self):
        return f"{DottedName.__name__}('{self}')"

    def __hash__(self):
        return hash(objpath(self))

    def __dhash__(self):
        return dhash(objpath(self))

    @staticmethod
    def _split_path(path):
        if isinstance(path, DottedName):
            return path.parts.copy()
        else:
            if path == '':
                return []
            else:
                return objpath(path).split('.')


class ObjectPath:
    def __init__(self, path, nested_path=None):
        """用于表示一个Python对象路径

        支持的形式包括：
            * '.'分割路径，例如xxx.yyy.zzz.Target
            * '.'分割路径但用':'分割模块和模块内路径，例如xxx.yyy.zzz:Target

        使用resolve后

        Parameters
        ----------
        path
            模块名或目标完整路径
        nested_path
            目标的模块内路径，如果未定义则认为path为完整名，会自动识别path的路径形式
        """
        self.module = None
        if isinstance(path, module_type):
            mod = path
            self.module = mod
            self.set_module_name_by_module(mod)
            self.nested_path = nested_path
        else:
            if nested_path is None:
                self.module_name, self.nested_path = split_colon_seperated_path(path)
            else:
                self.module_name = path
                self.nested_path = nested_path

    @staticmethod
    def of(obj_or_path):
        if isinstance(obj_or_path, str):
            ret = ObjectPath(obj_or_path)
        else:
            module = inspect.getmodule(obj_or_path)
            if module is None:
                raise ValueError(f'Unable to locate which module contains "{obj_or_path}".')
            nested_path = getattr(obj_or_path, '__qualname__', None)
            if nested_path is None:
                raise ValueError(f'"{obj_or_path}" does not contain "__qualname__". '
                                 f'It looks like an instance instead of valid targets like type or function.')
            ret = ObjectPath(path=module, nested_path=nested_path)
        return ret

    def resolve(self):
        ret = None

        if self.is_colon_seperated:
            if not self.is_module_loaded:
                # 此时nest path是正常状态，不需要重新修改
                self.module = importlib.import_module(objpath(self.module_name))
        else:
            ret = resolve_name_compat(objpath(self.nested_path))
            if isinstance(ret, module_type):
                module = ret
            else:
                import inspect
                module = inspect.getmodule(ret)

            self.module = module
            self.set_module_name_by_module(module)
            self.trim_nest_path_by_module_name()

        if ret is None and self.is_module_loaded:
            if self.is_module:
                ret = self.module
            else:
                ret = get_object_from(self.module, objpath(self.nested_path))

        return ret

    @property
    def module(self):
        return self._module

    @module.setter
    def module(self, mod):
        self._module = mod

    @property
    def module_name(self):
        return self._module_name

    @module_name.setter
    def module_name(self, val):
        if val is None:
            val = ''
        self._module_name = DottedName(val)

    @property
    def nested_path(self):
        return self._nested_path

    @nested_path.setter
    def nested_path(self, val):
        if val is None:
            val = ''
        self._nested_path = DottedName(val)

    def set_module_name_by_module(self, mod):
        self.module_name = mod.__name__

    def trim_nest_path_by_module_name(self):
        self.nested_path = self.nested_path.without_prefix(self.module_name)

    @property
    def is_colon_seperated(self):
        return not self.module_name.empty

    @property
    def is_module(self) -> Optional[bool]:
        if self.is_colon_seperated:
            return self.nested_path.empty
        else:
            return None

    @property
    def is_module_loaded(self):
        if self.module is not None:
            return True

        if self.is_colon_seperated:
            module = sys.modules.get(objpath(self.module_name), None)
            if module is not None:
                self.module = module
                self.set_module_name_by_module(module)
                self.trim_nest_path_by_module_name()
                return True

        return False

    def __iter__(self):
        for part in self.module_name:
            yield part
        for part in self.nested_path:
            yield part

    def __eq__(self, other):
        if not isinstance(other, ObjectPath):
            return False

        if self.is_module_loaded and other.is_module_loaded:
            return self.module == other.module and self.nested_path == other.nested_path

        for lhs, rhs in zip(self, other):
            if lhs != rhs:
                return False
        return True

    def __objpath__(self):
        if self.is_colon_seperated:
            return ':'.join((objpath(self.module_name), objpath(self.nested_path)))
        else:
            return objpath(self.nested_path)

    def __str__(self):
        return self.__objpath__()

    def __repr__(self):
        if self.is_colon_seperated:
            return f"{ObjectPath.__name__}(path='{self.module_name}', nested_path='{self.nested_path}')"
        else:
            return f"{ObjectPath.__name__}(path='{self.nested_path}')"

    def to_dotted(self):
        if self.is_colon_seperated:
            return DottedName(self.module_name, self.nested_path)
        else:
            return DottedName(self.nested_path)

    def __hash__(self):
        return hash(self.to_dotted())

    def __dhash__(self):
        return dhash(self.to_dotted())


objpath_like = Union[DottedName, ObjectPath, str]


def objpath(obj) -> str:
    """获取obj所代表的Python对象路径，类似os.fspath

    Parameters
    ----------
    obj
        代表Python对象路径的对象

    Returns
    -------
    ret
        obj所代表的对象路径

    Notes
    -----
        如果obj实现了__objpath__，则会调用该方法获取返回结果
        否则会使用str尝试将其转换为字符串，并触发警告

    Warnings
    --------
        objpath并非获取obj的对象路径
    """
    ret = getattr(obj, DottedName.__objpath__.__name__, None)
    if ret is not None:
        ret = ret()
    elif isinstance(obj, str):
        ret = obj
    else:
        warnings.warn(f'Object {type(obj).__name__} is not convertible to dotted name. Will try using str(), '
                      f'which may lead to unexpected result.')
        ret = str(obj)

    return ret


def split_colon_seperated_path(path: str):
    """用于分割使用':'定义的Python对象路径，':'分割了模块和模块内路径

    Parameters
    ----------
    path
        路径

    Returns
    -------
    ret: (str or None, str)
        如果对象路径包含冒号':'，则从冒号处分割并返回 (str_before_colon, str_after_colon)，否则返回 (None, path)
    """
    segs = path.split(':', maxsplit=1)
    if len(segs) == 2:
        return tuple(segs)
    else:
        return None, path


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
    ret
        获取到的对象，或None
    """
    ret = source

    if path == '':
        return ret

    for part in path.split('.'):
        ret = getattr(ret, part, None)
        if ret is None:
            break

    return ret


def get_qualname(obj):
    """获取目标的模块内路径，如果不存在__qualname__属性则返回__name__属性

    Parameters
    ----------
    obj
        目标对象

    Returns
    -------
    ret
        目标的模块内路径
    """
    ret = getattr(obj, '__qualname__', None)
    if ret is None:
        ret = getattr(obj, '__name__', None)

    return ret


def get_module_name(obj):
    """获取目标关联的模块名，一般是__module__

    Parameters
    ----------
    obj
        目标对象

    Returns
    -------
    ret
        目标关联的模块名
    """
    return getattr(obj, '__module__', None)


def get_full_qualified_path(obj):
    """从目标对象本身获取目标的包括模块名在内的全限定路径，使用 : 分割包和包内路径

    Parameters
    ----------
    obj
        目标对象，包括类或函数

    Returns
    -------
    ret
        目标对象的全限定路径字符串

    Examples
    --------
    >>> get_full_qualified_path(xxx.yyy.zzz.Target)
    <<< 'xxx.yyy.zzz:Target'

    Warnings
    --------
        该函数只接受类或函数等具有__module__和__qualname__的目标，常规类的实例无法以此方法获得路径
    """
    return objpath(ObjectPath.of(obj))


def reassign_object_name(obj, new_name: objpath_like = None, new_qualname: objpath_like = None):
    """重新给obj赋予名字，同时修改__name__和__qualname__，如果不存在则添加

    Parameters
    ----------
    obj
        待改名的对象
    new_name
        新的名字
    new_qualname
        新的模块内全限定名
    """
    if new_name is not None:
        new_name = DottedName(new_name)
        obj.__name__ = objpath(new_name)
        qualname = getattr(obj, '__qualname__', None)
        if qualname is not None:
            qualname = DottedName(qualname)
            qualname[-1] = new_name
            obj.__qualname__ = objpath(qualname)
    elif new_qualname is not None:
        new_qualname = DottedName(new_qualname)
        obj.__qualname__ = objpath(new_qualname)
        obj.__name__ = objpath(new_qualname.suffix(1))
    else:
        warnings.warn(f'{reassign_object_name.__name__} called without giving new names.')


def reassign_object_module(obj, new_module: objpath_like):
    """重新给obj赋予所关联的模块，如果不存在__module__则会直接添加

    Parameters
    ----------
    obj
        待修改关联模块的对象
    new_module
        新的所属模块名
    """
    obj.__module__ = objpath(new_module)


def mock_object(obj, obj_to_be_mocked):
    """修改obj的__module__，__name__和__qualname__属性，使之在ObjectPath的视角中和obj_to_be_mocked一样

    Parameters
    ----------
    obj
        需要进行伪装的对象
    obj_to_be_mocked
        目标对象
    """
    for prop in ['__module__', '__name__', '__qualname__']:
        val = getattr(obj_to_be_mocked, prop, None)
        if val is not None:
            setattr(obj, prop, val)


def get_nest_prefix_by_qualname(obj) -> str:
    """通过解析__qualname__提取对象在模块中的嵌套路径，如果是模块下直接定义的对象，则返回空字符串

    Parameters
    ----------
    obj
        待获取嵌套路径的对象，可以为type、类实例或方法以及其他具有__qualname__的对象

    Returns
    -------
    ret
        如果目标属于某嵌套定义下，则返回__qualname__中提取的嵌套路径（不包含对象本身的名字），
        如果是模块下直接定义的对象，则返回空字符串
    """
    segments = obj.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)
    return '.'.join(segments[:-1])


def get_object_by_path(path):
    """通过全限定路径获取目标

    Parameters
    ----------
    path
        目标的全限定路径

    Returns
    -------
    ret
        全限定路径所标识的目标

    Examples
    --------
    >>> get_object_by_path('xxx.yyy.zzz.Target')
    <class 'xxx.yyy.zzz.Target'>
    """
    return ObjectPath(path).resolve()


def get_class_that_defined_method(meth):
    """获取定义指定方法的类

    Parameters
    ----------
    meth
        方法实例

    Returns
    -------
    ret
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
        class_prefix = get_nest_prefix_by_qualname(meth)
        cls = get_object_from(inspect.getmodule(meth), class_prefix)
        if cls is None:
            path_segs = class_prefix.split('.')
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
    ret
        方法所在的类

    Notes
    -----
        由于Python3移除了所有类和类中定义的方法以及和模块的直接关联，因此只能通过一些方法来“猜测”，所以理论上存在无法获取到正确结果的情况

        比如如果随意修改__qualname__或者__module__的值会导致结果不可预测
    """
    if inspect.ismethod(obj):
        return obj.__self__
    elif inspect.isclass(obj):
        module = inspect.getmodule(obj)
        path = get_nest_prefix_by_qualname(obj)
        return get_object_from(module, path)
    else:
        ret = get_class_that_defined_method(obj)
        if ret is None:
            if obj.__qualname__ == obj.__name__:
                # 猜测定义在模块中
                ret = inspect.getmodule(obj)

        return ret


_NAME_PATTERN = None


def _resolve_name_compat(name):
    """解析path定义的对象路径，并返回路径

    实现来自Python 3.9的pkgutil.resolve_name

    Parameters
    ----------
    name
        Python对象路径

    Returns
    -------
    ret
        目标对象
    """
    global _NAME_PATTERN
    if _NAME_PATTERN is None:
        import re
        _DOTTED_WORDS = r'(?!\d)(\w+)(\.(?!\d)(\w+))*'
        _NAME_PATTERN = re.compile(f'^(?P<pkg>{_DOTTED_WORDS})(?P<cln>:(?P<obj>{_DOTTED_WORDS})?)?$', re.U)

    m = _NAME_PATTERN.match(name)
    if not m:
        raise ValueError(f'invalid format: {name!r}')
    gd = m.groupdict()
    if gd.get('cln'):
        # there is a colon - a one-step import is all that's needed
        mod = importlib.import_module(gd['pkg'])
        parts = gd.get('obj')
        parts = parts.split('.') if parts else []
    else:
        # no colon - have to iterate to find the package boundary
        parts = name.split('.')
        modname = parts.pop(0)
        # first part *must* be a module/package.
        mod = importlib.import_module(modname)
        while parts:
            p = parts[0]
            s = f'{modname}.{p}'
            try:
                mod = importlib.import_module(s)
                parts.pop(0)
                modname = s
            except ImportError:
                break
    # if we reach this point, mod is the module, already imported, and
    # parts is the list of parts in the object hierarchy to be traversed, or
    # an empty list if just the module is wanted.
    result = mod
    for p in parts:
        result = getattr(result, p)
    return result


if sys.version_info >= (3, 9):
    from pkgutil import resolve_name
    resolve_name_compat = resolve_name
else:
    resolve_name_compat = _resolve_name_compat
