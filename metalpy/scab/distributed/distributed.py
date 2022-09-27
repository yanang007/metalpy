import sys

from metalpy.mepa import Executor
from metalpy.mexin.injectors import hijacks
from metalpy.mexin.lazy_class_delegate import LazyClassFactory
from .distributed_simulation import DistributedSimulation
from metalpy.mexin.patch import Patch


class Distributed(Patch):
    def __init__(self, executor: Executor = None):
        super().__init__()
        self.executor = executor
        self.persisted_context = None

    def apply(self):
        if 'SimPEG.potential_fields.magnetics.simulation' in sys.modules:
            import SimPEG.potential_fields.magnetics
            self.hijack_cls_and_relative(SimPEG.potential_fields.magnetics.simulation.Simulation3DIntegral,
                                         self._parallel_wrapper, executor=self.executor)
            self.hijack_cls_and_relative(SimPEG.potential_fields.magnetics.sources.SourceField,
                                         self._lazy_wrapper)
            # Survey被人工重定向到了magnetics下，需要特殊处理
            self.hijack_cls_and_relative(SimPEG.potential_fields.magnetics.survey.Survey,
                                         self._lazy_wrapper, parent=False, child='survey')

    def hijack_cls_and_relative(self, cls, wrapper, parent=True, child=None, pass_cls=True, **extra_kwargs):
        """
        重定向类名字，使用wrapper函数替代
        :param cls: 目标类名
        :param wrapper: 替代的函数或者类

        :param parent: 是否在父级路径中重定向指定名字
        :param child: 是否在指定的子级路径中重定向指定名字
            这两个参数用于处理类的定义路径被修改的情形
            例如Survey定义在SimPEG.potential_fields.magnetics.survey，但被重定向到了SimPEG.potential_fields.magnetics
            需要同时重定向两个地方的定义
        :param pass_cls: 是否将cls作为第一个参数传递给wrapper
        :param extra_kwargs: 需要额外传给wrapper的参数
        """
        if pass_cls:
            extra_kwargs['cls'] = cls

        module_path = cls.__module__

        self.add_injector(hijacks(sys.modules[module_path], cls.__name__),
                          lambda *args, **kwargs: wrapper(*args, **kwargs, **extra_kwargs))
        if parent:
            module_parent_path = '.'.join(module_path.split('.')[:-1])
            self.add_injector(hijacks(sys.modules[module_parent_path], cls.__name__),
                              lambda *args, **kwargs: wrapper(*args, **kwargs, **extra_kwargs))
        if child is not None:
            module_child_path = '.'.join((module_path, child))
            self.add_injector(hijacks(sys.modules[module_child_path], cls.__name__),
                              lambda *args, **kwargs: wrapper(*args, **kwargs, **extra_kwargs))

    def hijack_class(self, cls, wrapper, pass_cls=True, **extra_kwargs):
        if pass_cls:
            extra_kwargs['cls'] = cls

        self.add_injector(hijacks(sys.modules[cls.__module__], cls.__name__),
                          lambda *args, **kwargs: wrapper(*args, **kwargs, **extra_kwargs))

    def unbind_context(self):
        """重写unbind_context来延长PatchContext的生命周期
        """
        self.persisted_context = self.context
        super(Distributed, self).unbind_context()

    @property
    def priority(self):
        return 0

    def _parallel_wrapper(self, *args, cls, executor=None, **kwargs):
        return DistributedSimulation(self, cls, executor=executor, *args, **kwargs)

    def _lazy_wrapper(self, *args, cls, **kwargs):
        return LazyClassFactory(cls, *args, **kwargs)