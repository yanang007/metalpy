import inspect

from .injectors import extends, replaces
from .patch import Patch
from .sentinal import NO_MIXIN


class MixinManager:
    """会被注册为目标类的mixins属性，由其他mixin调用进行mixin的注入或访问
    """
    def __init__(self, target):
        self._mixins = {}
        self.target = target

    def add(self, mixin_type, *args, **kwargs):
        target = self.target
        obj = mixin_type(target, *args, **kwargs)

        # 获取mixin的所有方法
        methods = inspect.getmembers(obj, predicate=lambda x: inspect.ismethod(x) or inspect.isfunction(x))
        for name, method in methods:
            if name.startswith('__') or name in ['post_apply']:
                # 跳过私有函数和mixin类构造过程的函数
                continue

            obj.__dict__[name] = self.bind_method(method, name=name)

        self._mixins[mixin_type] = obj
        obj.post_apply(target)

    def get(self, mixin_type, sentinel=False):
        if mixin_type in self._mixins:
            return self._mixins[mixin_type]
        else:
            if sentinel:
                return NO_MIXIN
            else:
                return None

    def get_mixin_or_sentinel(self, mixin_type):
        return self.get(mixin_type, sentinel=True)

    def bind_method(self, method, name=None):
        # TODO: 引入注解来标记是否需要替换，或者标记替换别的目标
        # TODO: 标记是否需要保留原函数
        if name is None:
            name = method.__name__
        target = self.target
        if hasattr(target, name):
            # 如果在目标存在同名方法则替换
            method = replaces(getattr(target, name), keep_orig=False)(method)
        else:
            method = extends(target, name)(method)

        return method


class Mixed(Patch):
    def __init__(self, target):
        super().__init__()
        self.target = target

    def apply(self):
        extends(self.target, 'mixins')(self.__mixins)  # 防止离开上下文时，mixin属性被删除

    @staticmethod
    @property
    def __mixins(self):
        if hasattr(self, '_mixins'):
            return self._mixins
        else:
            self._mixins = MixinManager(self)
            return self._mixins

    @property
    def priority(self):
        return -1
