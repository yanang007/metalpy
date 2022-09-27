import sys

import SimPEG
import numpy as np
from SimPEG.potential_fields import magnetics
from SimPEG.simulation import BaseSimulation

from metalpy.mepa import Executor, LinearExecutor
from metalpy.mexin import LazyClassFactory, PatchContext
from metalpy.mexin.injectors import is_or_is_replacement

from ..simpeg_patch_context import simpeg_patched
from .policies import AlwaysFalse


class DistributedSimulation(LazyClassFactory):
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
            目前只用于len(receiver_list)==1的magnetics.simulation.Simulation3DIntegral，对于其它Simulation兼容性未知

        Known Issues
        ------------
            如果在调用dpred之前未退出PatchContext，且executor为本地的LinearExecutor，那么会导致死锁

        """

        super().__init__(simulation_class, *args, **kwargs)

        if not isinstance(survey, LazyClassFactory):
            raise ValueError('Error: survey must be a LazyClassFactory. '
                             'Please check if the patch system is working correctly.')

        source_field = survey.find_param_by_type(LazyClassFactory, remove=True)

        if source_field is None:
            raise ValueError('Error: source field must be a LazyClassFactory. '
                             'Please check if the patch system is working correctly.')

        if not is_or_is_replacement(SimPEG.potential_fields.magnetics.simulation.Simulation3DIntegral, simulation_class):
            print('Warning: Distributed patch is only tested for magnetics.simulation.Simulation3DIntegral')

        # 针对magnetics.simulation.Simulation3DIntegral实现
        # TODO: 引入来支持其它Simulation，TaskDividePolicy?
        receiver_list = source_field.find_param(lambda k, v: k.startswith('receiver_'), remove=True)
        receiver_locations = []

        for receiver in receiver_list:
            receiver_locations.append(receiver.locations)
            receiver.locations = receiver.locations[0]  # 占位符，用来越过receiver的类型检测

        self.simulation_class = simulation_class
        self.survey = survey
        self.source_field = source_field
        self.receiver_list = receiver_list
        self.locations_list = receiver_locations

        self.executor = executor

        self.parallelized_patch = patch

        self.patch_policies = {}
        self.default_patch_policy = AlwaysFalse()

        if 'metalpy.scab.progressed' in sys.modules:
            from ..progressed import Progressed
            from .policies import ProgressedPolicy
            self.define_patch_policy(Progressed, ProgressedPolicy())

    @staticmethod
    def worker(simulation_delegate,
               survey_delegate,
               source_field_delegate,
               receiver_list_container,
               locations_list,
               _model,
               patches):
        for receiver in receiver_list_container:
            receiver.locations = locations_list.pop(0)

        with simpeg_patched(*patches):
            survey = survey_delegate.construct(
                source_field=source_field_delegate.construct(
                    receiver_list=receiver_list_container
                ))

            simulation = simulation_delegate.construct(survey=survey)

        return simulation.dpred(_model)

    def dpred(self, model):
        if PatchContext.lock.locked():
            raise AssertionError('Error: dpred must be called outside of PatchContext.')

        futures = []
        receiver_tasks = self.executor.arrange(*self.locations_list)
        for dest_worker in self.executor.get_workers():
            # 使用get_patch_policy来判断上下文中应用的patch哪些需要在worker中启用
            patches = [patch for patch in self.get_patches() if self.get_patch_policy(patch)(patch, dest_worker)]

            future = self.executor.submit(
                self.worker, worker=dest_worker,
                simulation_delegate=self.clone(),  # 使用clone创建一个lazy构造器
                survey_delegate=self.survey,
                source_field_delegate=self.source_field,
                receiver_list_container=self.receiver_list,
                locations_list=receiver_tasks.assign(dest_worker),
                _model=model,
                patches=patches
            )
            futures.append(future)

        ret = self.executor.gather(futures)
        ret = np.concatenate(ret)
        return ret

    def define_patch_policy(self, patch_type, policy):
        self.patch_policies[patch_type] = policy

    def get_patch_policy(self, patch):
        policy = self.patch_policies.get(patch.__class__, self.default_patch_policy)
        return policy

    def get_patches(self):
        return self.parallelized_patch.persisted_context.get_patches()
