from abc import ABC, abstractmethod

from .injectors import RecoverableInjector, after
from .mixin import Mixin


class Patch(ABC):
    def __init__(self):
        """用于在全局上下文中劫持与注入python-based方法，继承类必须重写apply方法

        Notes
        -----
            所有方法必须保持可重入，即多次调用apply方法，必须保证结果一致
            rollback会消除commit的影响
            post_rollback会消除pre_apply和apply的影响


            执行时PatchContext会按如下顺序调用方法：

            **bind_context**: 绑定上下文

            **pre_apply**: 在apply之前调用，用于初始化一些变量

            **apply**: 在这里添加patch，子类必须实现

            **commit**: 在这里进行patch的应用，如果要改写，应该需要调用基类方法

            **rollback**: 在这里进行一些rollback操作

            **post_rollback**: 在rollback之后调用，用于清理一些变量

            **unbind_context**: 解绑上下文

        See Also
        --------
            PatchContext: patch上下文
        """
        self.recoverable_injs: list[tuple[RecoverableInjector, tuple, dict]] = None
        self.__mixin_injs: list[tuple[RecoverableInjector, tuple, dict]] = None
        self.mixins: dict[str, list[tuple[Mixin, tuple, dict]]] = None
        self._context = None

    def get_mixed_classes(self):
        return self.mixins.keys()

    def pre_apply(self):
        self.recoverable_injs = []
        self.__mixin_injs = []
        self.mixins = {}

    @abstractmethod
    def apply(self):
        pass

    def commit(self):
        if len(self.__mixin_injs) == 0:
            # 保证Patch可重入
            for target_type, mixins in self.mixins.items():  # 将混入mixin的代码插入到对应类的构造函数
                self.__mixin_injs.append((after(target_type.__init__), (self.__apply_mixins(mixins),), {}))

        for inj, args, kwargs in self.__get_injs():
            inj(*args, **kwargs)

    @staticmethod
    def __apply_mixins(mixins):
        def apply_mixins(this, *_, **__):
            for mixin_type, args, kwargs in mixins:
                this.mixins.add(mixin_type, *args, **kwargs)
        return apply_mixins

    def rollback(self):
        for inj, args, kwargs in self.__get_injs(reverse=True):
            inj.rollback()

    def post_rollback(self):
        # 防止其中的lambda函数和类指针被pickle
        self.recoverable_injs = None
        self.__mixin_injs = None
        self.mixins = None

    def __get_injs(self, reverse=False):
        if reverse:
            return reversed(self.recoverable_injs + self.__mixin_injs)
        else:
            return self.recoverable_injs + self.__mixin_injs

    @property
    def priority(self):
        # -1 专属于Mixed的优先级，用于注入Mixin系统
        # TODO: 可能可以改为require/after类型的依赖图？
        return 225

    @property
    def context(self):
        return self._context

    def bind_context(self, ctx):
        self._context = ctx

    def unbind_context(self):
        self._context = None

    def add_injector(self, inj: RecoverableInjector, *args, **kwargs):
        self.recoverable_injs.append((inj, args, kwargs))

    def add_mixin(self, target_type, mixin_type: [Mixin], *args, **kwargs):
        self.mixins.setdefault(target_type, []).append((mixin_type, args, kwargs))
