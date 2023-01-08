import copy
import os

import SimPEG
import blosc2
import numpy as np
from SimPEG.potential_fields import magnetics
from SimPEG.simulation import BaseSimulation
from discretize import TensorMesh
from properties.utils import undefined

from metalpy.mepa import Executor
from metalpy.mexin import LazyClassFactory, PatchContext
from metalpy.mexin.injectors import is_or_is_replacement, reverted
from metalpy.utils.type import pop_or_default, not_none_or_default, get_or_default
from metalpy.scab.simpeg_patch_context import simpeg_patched
from .policies import Distributable, NotDistributable
from .utils import reget_class


class DistributedSimulation(LazyClassFactory, BaseSimulation):
    def __init__(self,
                 patch,
                 simulation_class: BaseSimulation,
                 survey: LazyClassFactory,
                 executor: Executor = None,
                 *args, **kwargs):
        """
        Parameters
        ----------
        simulation_class
            正演类，继承自BaseSimulation
        survey
            观测网格和背景条件
        executor
            执行器，默认为None，此时会使用LinearExecutor
        args
            simulation的位置参数
        kwargs
            simulation的关键字参数

        Notes
        -----
            目前只用于magnetics.simulation.Simulation3DIntegral，对于其它Simulation兼容性未知
        """

        super().__init__(simulation_class, *args, **kwargs)

        if not isinstance(survey, LazyClassFactory):
            raise ValueError('Error: survey must be a LazyClassFactory. '
                             'Please check if the patch system is working correctly.')

        source_field = survey.find_param_by_type(LazyClassFactory, remove=True)

        if source_field is None:
            raise ValueError('Error: source field must be a LazyClassFactory. '
                             'Please check if the patch system is working correctly.')

        if not is_or_is_replacement(SimPEG.potential_fields.magnetics.simulation.Simulation3DIntegral,
                                    simulation_class):
            print('Warning: Distributed patch is only tested for magnetics.simulation.Simulation3DIntegral')

        # 针对magnetics.simulation.Simulation3DIntegral实现
        # TODO: 引入来支持其它Simulation，TaskDividePolicy?
        receiver_list = source_field.find_param(lambda k, v: k.startswith('receiver_'), remove=True)

        # 和外部共用receiver_list，防止被后文修改
        self.simulation = self.build_simulation(survey, source_field, receiver_list)

        # 复制receiver_list，防止影响到外部代码
        receiver_list = [copy.deepcopy(receiver) for receiver in receiver_list]
        receiver_locations = []
        for receiver in receiver_list:
            receiver_locations.append(receiver.locations)
            receiver.locations = receiver.locations[0]  # 占位符，用来越过receiver的类型检测

        self.simulation_class = simulation_class
        self.survey_factory = survey
        self.source_field = source_field
        self.receiver_list = receiver_list
        self.locations_list = receiver_locations

        self.executor = executor

        self.parallelized_patch = patch

        # 在构造完本地代理类后劫持分派类的store_sensitivities参数，如果为disk，则改为ram，防止在远程端产生不必要的cache和路径冲突
        self.store_sensitivities = get_or_default(self.kwargs, 'store_sensitivities', 'ram')
        if self.store_sensitivities == 'disk':
            self.kwargs['store_sensitivities'] = 'ram'
        else:
            self.kwargs['store_sensitivities'] = self.store_sensitivities
        self.sensitivity_path = pop_or_default(self.kwargs, 'sensitivity_path', "./sensitivity/")

    @staticmethod
    def auto_decompress_simulation(sim, ind_future, mesh_future):
        sim['ind_active'] = blosc2.unpack_array2(ind_future)
        sim['mesh'] = mesh_future
        return sim

    @staticmethod
    def auto_decompress_mesh_h(compressed_mesh_defs):
        return [blosc2.unpack_array2(ch) for ch in compressed_mesh_defs]

    def compress_model(self, sim_delegate, model):
        """用于压缩模型，以减少传输量

        主要包含model和simulation里的ind_active

        Parameters
        ----------
        sim_delegate
            仿真类构造器

        model
            模型

        Returns
        -------
        ret
            压缩后的仿真类构造器和模型
        """
        from pympler.asizeof import asizeof

        if asizeof(model) > 1024 * 1024:
            compressed_model = blosc2.pack_array2(model)
            compressed_model = self.executor.scatter(compressed_model)
            model = self.executor.submit(blosc2.unpack_array2, compressed_model)

        if asizeof(sim_delegate) > 1024 * 1024:
            compressed_act_ind = blosc2.pack_array2(sim_delegate['ind_active'])
            compressed_act_ind = self.executor.scatter(compressed_act_ind)
            sim_delegate['ind_active'] = 0

            mesh = sim_delegate['mesh']
            del sim_delegate['mesh']
            if isinstance(mesh, TensorMesh):
                if asizeof(mesh.h) > 1024 * 1024:
                    compressed_mesh_h = [blosc2.pack_array2(h) for h in mesh.h]
                    mesh_h = self.executor.submit(self.auto_decompress_mesh_h, compressed_mesh_h)
                else:
                    mesh_h = mesh.h

                origin = mesh.origin
                mesh_future = self.executor.submit(TensorMesh, h=mesh_h, origin=origin)
            else:
                mesh_future = self.executor.scatter(mesh)

            sim_delegate = self.executor.submit(self.auto_decompress_simulation,
                                                sim_delegate,
                                                compressed_act_ind,
                                                mesh_future)

        return sim_delegate, model

    @staticmethod
    def linear_operator_worker(simulation_delegate,
                               survey_delegate,
                               source_field_delegate,
                               receiver_list_container,
                               locations_list,
                               model,
                               patches):
        # model property强制要求为undefined或numpy数组，不可为None
        model = not_none_or_default(model, undefined)

        for receiver in receiver_list_container:
            receiver.locations = locations_list.pop(0)

        with simpeg_patched(*patches):
            survey = survey_delegate.construct(
                source_field=source_field_delegate.construct(
                    receiver_list=receiver_list_container
                ))

            simulation = simulation_delegate.construct(survey=survey)
            simulation.model = model

        return simulation.linear_operator()

    def linear_operator(self, this):
        """LinearSimulation
        """
        n_cells = self.nC
        if getattr(self, "model_type", None) == "vector":
            n_cells *= 3
        if self.store_sensitivities == "disk":
            sens_name = os.path.join(self.sensitivity_path, "sensitivity.npy")
            if os.path.exists(sens_name):
                # do not pull array completely into ram, just need to check the size
                kernel = np.load(sens_name, mmap_mode="r")
                if kernel.shape == (self.survey.nD, n_cells):
                    print(f"Found sensitivity file at {sens_name} with expected shape")
                    kernel = np.asarray(kernel)
                    return kernel

        if PatchContext.lock.locked():
            # 需要等PatchContext退出来让加载的Patch解除对上下文的绑定，从而防止序列化方面的问题
            raise AssertionError('Error: linear_operator must be called outside of PatchContext.')

        simulation_delegate = self.clone()  # 使用clone创建一个lazy构造器
        model = this.model
        if not self.executor.is_local():
            simulation_delegate, model = self.compress_model(simulation_delegate, model)

        futures = []
        receiver_tasks = self.executor.arrange(*self.locations_list)
        for dest_worker in self.executor.get_workers():
            # 使用get_patch_policy来判断上下文中应用的patch哪些需要在worker中启用
            patches = [patch for patch in self.get_patches()
                       if self.get_patch_policy(patch).should_distribute_to(dest_worker)]

            future = self.executor.submit(
                self.linear_operator_worker, worker=dest_worker,
                simulation_delegate=simulation_delegate,
                survey_delegate=self.survey_factory,
                source_field_delegate=self.source_field,
                receiver_list_container=self.receiver_list,
                locations_list=receiver_tasks.assign(dest_worker),
                model=model,
                patches=patches
            )
            futures.append(future)

        segments = self.executor.gather(futures)
        kernel = np.concatenate(segments)

        if self.store_sensitivities == "disk":
            print(f"writing sensitivity to {sens_name}")
            os.makedirs(self.sensitivity_path, exist_ok=True)
            np.save(sens_name, kernel)

        return kernel

    def dpred(self, model=None, f=None):
        """BaseSimulation
        """
        return self.simulation.dpred(model, f)

    @property
    def model(self):
        """LinearSimulation
        """
        return self.simulation.model

    @property
    def survey(self):
        """BaseSimulation
        """
        return self.simulation.survey

    def getJtJdiag(self, m, W=None):
        """Simulation3DIntegral / Simulations from electromagnetics
        """
        return self.simulation.getJtJdiag(m, W)

    def fields(self, model):
        """BaseSimulation
        """
        return self.simulation.fields(model)

    def Jtvec(self, m, v, f=None):
        """BaseSimulation
        """
        return self.simulation.Jtvec(m, v, f=f)

    def Jvec(self, m, v, f=None):
        """BaseSimulation
        """
        return self.simulation.Jvec(m, v, f=f)

    def Jtvec_approx(self, m, v, f=None):
        """BaseSimulation
        """
        return self.simulation.Jtvec_approx(m, v, f=f)

    def Jvec_approx(self, m, v, f=None):
        """BaseSimulation
        """
        return self.simulation.Jvec_approx(m, v, f=f)

    def residual(self, m, dobs, f=None):
        """BaseSimulation
        """
        return self.simulation.residual(m, dobs, f=f)

    @property
    def G(self):
        """LinearSimulation
        """
        return self.simulation.G

    @G.setter
    def G(self, value):
        """LinearSimulation
        """
        self.simulation.G = value

    @property
    def nC(self):
        """BasePFSimulation
        """
        return self.simulation.nC

    @staticmethod
    def get_patch_policy(patch):
        if isinstance(patch, Distributable):
            return patch
        else:
            return NotDistributable()

    def get_patches(self):
        return self.parallelized_patch.persisted_context.get_patches()

    def build_simulation(self, survey, source_field, receiver_list):
        r = reget_class(survey.cls), reget_class(source_field.cls),
        with reverted(*r):
            ret = self.construct(
                survey=survey.construct(
                    source_field=source_field.construct(
                        receiver_list=receiver_list)))
        ret.mixins.bind_method(self.linear_operator)
        return ret
