import sys

from SimPEG.simulation import BaseSimulation

from metalpy.mepa import Executor
from metalpy.mexin import Mixin
from metalpy.mexin.patch import Patch
from metalpy.mexin.injectors import reverted, replaces
from metalpy.mexin.lazy_class_delegate import LazyClassFactory
from .distributed_simulation import DistributedSimulation
from .utils import reget_class


class Distributed(Patch):
    def __init__(self, executor: Executor = None):
        super().__init__()
        self.executor = executor
        self.persisted_context = None

    def apply(self):
        self.add_mixin(BaseSimulation, Mixin)  # 保证BaseSimulation绑定mixin系统供DistributedSimulation使用

        if 'SimPEG.data' in sys.modules:
            from SimPEG import data
            self.hijack_cls_and_relative(data.Data, self._SimPEG_data_Data_wrapper)

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

        self.add_injector(replaces(cls, nest=sys.modules[module_path]),
                          lambda *args, **kwargs: wrapper(*args, **kwargs, **extra_kwargs))
        if parent:
            module_parent_path = '.'.join(module_path.split('.')[:-1])
            self.add_injector(replaces(cls, nest=sys.modules[module_parent_path]),
                              lambda *args, **kwargs: wrapper(*args, **kwargs, **extra_kwargs))
        if child is not None:
            module_child_path = '.'.join((module_path, child))
            self.add_injector(replaces(cls, nest=sys.modules[module_child_path]),
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

    @staticmethod
    def _lazy_wrapper(*args, cls, **kwargs):
        return LazyClassFactory(cls, *args, **kwargs)

    @staticmethod
    def _SimPEG_data_Data_wrapper(survey, dobs=None, cls=None, **kwargs):
        source_field = survey.find_param_by_type(LazyClassFactory, remove=False)

        # TODO: 可改进的HACK
        # 因为这些类被劫持了，得到的会是LazyClassFactory，
        # 因此需要通过获取Replacement对象临时还原才能正常构造(LazyClassFactory中的是原本的cls)
        repl_survey = reget_class(survey.cls)
        repl_sourcefield = reget_class(source_field.cls)
        with reverted(repl_survey, repl_sourcefield):  # 临时还原这两个被替换的类才能正常完成构造
            return Distributed._lazy_wrapper(survey, cls=cls, dobs=dobs, **kwargs).construct()
