import inspect
from typing import Union, Callable, Iterable

from metalpy.utils.object_path import ObjectPath
from metalpy.utils.type import get_or_default

type_or_str = Union[type, str]
type_desc = Union[type_or_str, Callable]


class TypeMap:
    def __init__(self, allow_match_parent=False):
        """用于实现类型间的非直接关联

        使用字符串来关联源类型和目标类型，实现在不import源类型和目标类型的情况下进行关联

        支持扩展对象路径语法，可以使用 '/' 分割模块和模块内路径，从而消除歧义
        
        Parameters
        ----------
        allow_match_parent
            是否禁止在找不到直接匹配时递归寻找最近父类的匹配

        See Also
        --------
            get_object_by_path, split_object_path: 扩展对象路径语法

        Notes
        -----
            TypeMap为了规避不必要的import，选择支持使用字符串来表示类型，但这会导致维护困难

            因此一切使用TypeMap的代码应添加相应的测试，请添加测试用例使用assert_all_types_exists验证TypeMap中所定义的类型均有效且存在
        """
        self.registries: dict[ObjectPath, type_desc] = {}
        self.allow_match_parent = allow_match_parent

    def map(self, type_key: type_or_str, type_val: type_desc):
        """注册一个类型间的映射

        Parameters
        ----------
        type_key
            作为key的类型，可以是类型或类型路径
        type_val
            作为val的类型，如果为字符串，则会在需要时动态载入，如果为函数，则会在获取时调用并返回结果
        """
        if isinstance(type_val, str):
            type_val = ObjectPath.of(type_val)
        self.registries[ObjectPath.of(type_key)] = type_val

    def strict_get(self, type_key, default=None) -> type_desc:
        """严格模式下获取key类型对应的原始类型值，不会匹配父类，不会将字符串或函数转换为具体类

        Parameters
        ----------
        type_key
            键类型，如果为类型且无匹配，则会获取对应路径再次尝试匹配
        default
            如果无匹配的返回值

        Returns
        -------
        ret
            获取的类型
        """
        ret = get_or_default(self.registries, ObjectPath.of(type_key), _default=None)
        if ret is None:
            ret = default

        return ret

    def get(self, type_key, default=None) -> type:
        """获取key类型对应的类型，如果strict_mode为False则会匹配父类

        Parameters
        ----------
        type_key
            键类型，如果为类型且无匹配，则会获取对应路径再次尝试匹配
        default
            如果无匹配的返回值

        Returns
        -------
        ret
            获取的类型
        """
        ret = self.strict_get(type_key, default=default)

        if ret is None:
            if self.allow_match_parent and isinstance(type_key, type):
                for parent_type in inspect.getmro(type_key)[1:]:
                    ret = self.strict_get(parent_type, default=default)
            else:
                ret = default

        if isinstance(ret, ObjectPath):
            ret = ret.resolve()
        elif callable(ret):
            ret = ret()

        return ret

    def keys(self) -> Iterable[ObjectPath]:
        """获取TypeMap的所有作为键的类型

        Returns
        -------
        ret
            获取TypeMap的所有原始键
        """
        return self.registries.keys()

    def values(self) -> Iterable[type_desc]:
        """获取TypeMap的所有作为值的类型，不会转换为实际的值

        Returns
        -------
        ret
            获取TypeMap的所有原始值
        """
        return self.registries.values()


def assert_all_types_exists(type_map: TypeMap):
    for key in type_map.keys():
        cls = key.resolve()
        assert isinstance(cls, type), f'Key type "{key}" in TypeMap not found.'
        assert ObjectPath.of(cls) == key, f'ObjectPath to type "{cls.__name__}" is not unique, ' \
                                          f'please use "{ObjectPath.of(cls)}" as the key to this type.'

        value = type_map.get(cls, None)
        assert isinstance(value, type), f'Key "{key}" does not map to a valid type.'
