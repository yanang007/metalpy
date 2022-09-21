import numpy as np
from SimPEG.potential_fields import magnetics
from SimPEG.simulation import BaseSimulation
from SimPEG.survey import BaseSurvey, BaseSrc

from META.mepa import Executor
from .injectors import extends
from .lazy_class_delegate import LazyClassFactory


class ParallelizedSimulation(LazyClassFactory):
    def __init__(self,
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
        """

        wrong_usage_error_msg = '''Parallelized simulation takes specially constructed BaseSurvey and BaseSrc.
    Bad:
        source_field = SourceField(receiver_list=receiver_list, parameters=H)
        survey = Survey(source_field)
        simulation = Simulation3DIntegral.parallel(
            survey=survey,
            // ...
        )
        
    Expected:
        source_field = SourceField.parallel(receiver_list=receiver_list, parameters=H)
        survey = Survey.parallel(source_field)
        simulation = Simulation3DIntegral.parallel(
            survey=survey,
            // ...
        )
'''
        super().__init__(simulation_class, *args, **kwargs)

        if not isinstance(survey, LazyClassFactory):
            raise ValueError(wrong_usage_error_msg)

        source_field = survey.find_param_by_type(LazyClassFactory, remove=True)

        if source_field is None:
            raise ValueError(wrong_usage_error_msg)

        if magnetics.simulation.Simulation3DIntegral != simulation_class:
            print('Warning: ParallelizedSimulation is only tested for magnetics.simulation.Simulation3DIntegral')
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

    @staticmethod
    def worker(simulation_delegate,
               survey_delegate,
               source_field_delegate,
               receiver_list_container,
               locations_list,
               _model):
        for receiver in receiver_list_container:
            receiver.locations = locations_list.pop(0)

        survey = survey_delegate.construct(
            source_field=source_field_delegate.construct(
                receiver_list=receiver_list_container
            ))
        simulation = simulation_delegate.construct(survey=survey)
        return simulation.dpred(_model)

    def dpred(self, model):
        futures = []
        receiver_tasks = self.executor.arrange(*self.locations_list)
        for dest_worker in self.executor.get_workers():
            future = self.executor.submit(
                self.worker, worker=dest_worker,
                simulation_delegate=self.clone(),  # 使用clone创建一个lazy构造器
                survey_delegate=self.survey,
                source_field_delegate=self.source_field,
                receiver_list_container=self.receiver_list,
                locations_list=receiver_tasks.assign(dest_worker),
                _model=model
            )
            futures.append(future)

        ret = self.executor.gather(futures)
        ret = np.concatenate(ret)
        return ret


@extends(BaseSimulation, 'parallel')
@classmethod
def __BaseSimulation_ext_parallel(cls, executor=None, *args, **kwargs):
    return ParallelizedSimulation(cls, executor=executor, *args, **kwargs)


@extends(BaseSurvey, 'parallel')
@classmethod
def __BaseSurvey_ext_parallel(cls, *args, **kwargs):
    return LazyClassFactory(cls, *args, **kwargs)


@extends(BaseSrc, 'parallel')
@classmethod
def __BaseSrc_ext_parallel(cls, *args, **kwargs):
    return LazyClassFactory(cls, *args, **kwargs)

# BaseSimulation.parallel = parallelized_simulation
