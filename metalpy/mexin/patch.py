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
        self.mixins: dict[type, list[tuple[Mixin, tuple, dict]]] = None
        self._context = None

    def get_mixed_classes(self):
        return self.mixins.keys()

    def pre_apply(self):
        self.recoverable_injs = []
        self.mixins = {}

    @abstractmethod
    def apply(self):
        pass

    def commit(self):
        for inj, args, kwargs in self.__get_injs():
            inj(*args, **kwargs)

    def rollback(self):
        for inj, args, kwargs in self.__get_injs(reverse=True):
            inj.rollback()

    def post_rollback(self):
        # 防止其中的lambda函数和类指针被pickle
        self.recoverable_injs = None
        self.mixins = None

    def __get_injs(self, reverse=False):
        if reverse:
            return reversed(self.recoverable_injs)
        else:
            return self.recoverable_injs

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
        """追加injector，执行全局的替换或劫持等操作

        Parameters
        ----------
        inj
            待添加的injector对象
        args
            注入操作（injector.__call__）的位置参数
        kwargs
            注入操作（injector.__call__）的关键字参数

        See Also
        --------
            PatchContext.apply: 应用injector
        """
        self.recoverable_injs.append((inj, args, kwargs))

    def add_mixin(self, target_type, mixin_type: [Mixin], *args, **kwargs):
        """指定向target_type添加局限于类实例的Mixin，在作例用范围内创建的所有target_type实例都会对应创建并绑定一份该Mixin实例

        Parameters
        ----------
        target_type
            需要添加Mixin的目标类
        mixin_type
            待添加的Mixin类
        args
            Mixin类的位置参数
        kwargs
            Mixin类的关键字参数

        Notes
        -----
            Patch本身不会进行任何关于Mixin的操作，Mixin的创建与绑定由PatchContext完成

        See Also
        --------
            PatchContext.apply: 执行MixinManager的创建与绑定
            Mixed.with_mixin: 执行Mixin的定义
            Mixed.apply_mixin or MixinManager.add: 执行Mixin的创建与绑定
        """
        self.mixins.setdefault(target_type, []).append((mixin_type, args, kwargs))
