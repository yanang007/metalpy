from SimPEG.simulation import BaseSimulation

from metalpy.mexin import Mixin
from metalpy.mexin import Patch
from metalpy.utils.taichi import ti_prepare, ti_arch
from metalpy.utils.type import get_params_dict, get_full_qualified_class_name, get_or_default, get_class_name

from .potential_fields.magnetics.simulation import TiedSimulation3DIntegralMixin
from ..distributed.policies import Distributable

implementations = {
    'SimPEG.potential_fields.magnetics.simulation.Simulation3DIntegral': TiedSimulation3DIntegralMixin,
}


class TaichiContext(Mixin):
    def __init__(self, this, arch=None, max_cpu_threads=None):
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
            params['cpu_max_num_threads'] = max_cpu_threads
        ti_prepare(**params)

    def post_apply(self, this):
        type_name = get_class_name(this)
        path = get_full_qualified_class_name(this)
        impl = get_or_default(implementations, path, None)

        if impl is None:
            print(f'Taichi support for {type_name} is not implemented. Ignoring it.')
            return

        this.mixins.add(impl)


class Tied(Patch, Distributable):
    def __init__(self, arch=None, max_cpu_threads=None):
        super().__init__()
        self.params = get_params_dict(arch=arch, max_cpu_threads=max_cpu_threads)

    def apply(self):
        self.add_mixin(BaseSimulation, TaichiContext, **self.params)
