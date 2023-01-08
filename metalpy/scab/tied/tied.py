import os

from SimPEG.simulation import BaseSimulation

from metalpy.mexin import Mixin
from metalpy.mexin import Patch
from metalpy.utils.taichi import ti_prepare, ti_arch, ti_reset
from metalpy.utils.type import get_params_dict

from metalpy.utils.object_path import get_full_qualified_path
from metalpy.mexin.utils import TypeMap
from metalpy.scab.distributed.policies import Distributable


class TaichiContext(Mixin):
    _implementations = TypeMap()

    def __init__(self, this, arch=None, max_cpu_threads=None, **kwargs):
        """初始化taichi上下文

        Parameters
        ----------
        this
            Simulation类指针，由mixin manager传递
        arch
            taichi架构，例如ti.cpu或ti.gpu，默认ti.cpu
        max_cpu_threads
            cpu arch下的最大线程数，taichi默认为cpu核心数
        """
        super().__init__(this)
        params = {}
        if arch is not None:
            params['arch'] = ti_arch(arch)
        if max_cpu_threads is not None:
            if max_cpu_threads < 0:
                # 需要注意taichi目前使用的为虚拟核心数 (taichi/program/compile_config.cpp)
                # cpu_max_num_threads = std::thread::hardware_concurrency();
                # 因此这里使用python自带的cpu_count()，而不是psutil的物理核心数，和taichi保证一致
                # 考虑到超线程的存在，-1可能仍然会导致计算机满载
                max_cpu_threads = os.cpu_count() + max_cpu_threads
            params['cpu_max_num_threads'] = max_cpu_threads
        ti_prepare(**params, **kwargs)

    def post_apply(self, this):
        impl = TaichiContext._implementations.get(type(this))

        if impl is None:
            print(f'Taichi support for {get_full_qualified_path(this)} is not implemented. Ignoring it.')
            return

        this.mixins.add(impl)


class Tied(Patch, Distributable):
    def __init__(self, arch=None, max_cpu_threads=None, **kwargs):
        super().__init__()
        ti_reset()
        self.params = get_params_dict(arch=arch, max_cpu_threads=max_cpu_threads, **kwargs)

    def apply(self):
        self.add_mixin(BaseSimulation, TaichiContext, **self.params)


def __implements(target):
    def decorator(func):
        TaichiContext._implementations.map(target, func)
        return func
    return decorator


@__implements('SimPEG.potential_fields.magnetics.simulation.Simulation3DIntegral')
def _():
    from .potential_fields.magnetics.simulation import TiedSimulation3DIntegralMixin
    return TiedSimulation3DIntegralMixin
