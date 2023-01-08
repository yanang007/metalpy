from abc import ABC, abstractmethod

from .task_allocator import TaskAllocator, SingleTaskAllocator
from .utils import structured_traverse


class Executor(ABC):
    """
    Note:
    1. 工作单元 n_units 以资源池中的基础算力作为单位。
        例：假设存在worker a和b，a的算力是b的两倍，则以b为基本单位，返回1+2=3
    """
    def __init__(self):
        pass

    def submit(self, func, *args,
               worker=None, workers=None, **kwargs):
        if workers is None and worker is not None:
            workers = [worker]
        return self.do_submit(func, workers=workers, *args, **kwargs)

    @abstractmethod
    def do_submit(self, func, *args, workers=None, **kwargs):
        pass

    def arrange_single(self, data):
        return SingleTaskAllocator(self.get_n_units(), data)

    def arrange(self, *data_list):
        return TaskAllocator(self.get_n_units(), *data_list)

    @abstractmethod
    def get_workers(self):
        pass

    @abstractmethod
    def get_n_units(self):
        """
        返回当前集群的基本工作单位数
        :return: 当前集群的基本工作单位数
        :rtype: int
        """
        pass

    @abstractmethod
    def is_local(self):
        return True

    def gather(self, futures):
        """收集futures的结果

        Parameters
        ----------
        futures
            待收集的futures，可以为单个future也可以是future的Iterable或dict

        Returns
        -------
        ret
            以原本结构构造的future结果

        Notes
        -----
            子类需从gather和gather_single二选一实现

        Examples
        --------
        >>> from operator import add
        >>> from metalpy.mepa.linear_executor import LinearExecutor
        >>> executor = LinearExecutor()
        >>> f = executor.submit(add, 1, 2)
        >>> executor.gather(f)
        3
        >>> executor.gather([f, [f], f])  # 支持列表
        [3, [3], 3]
        >>> executor.gather({'First': f, 'Second': [f, [f]]})  # 支持字典
        {'First': 3, 'Second': [3, [3]]}
        """
        return structured_traverse(futures, lambda f: self.gather_single(f))

    def gather_single(self, future):
        """收集单个的结果

        Parameters
        ----------
        future
            待收集的future

        Returns
        -------
        ret
            future的结果

        Notes
        -----
            子类需从gather和gather_single二选一实现
        """
        return self.gather(future)

    def scatter(self, data):
        """
        用于预先分发大规模数据（一般指内存占用大于1MB）
        :param data: 大规模数据

        see also
        distributed.Client.scatter
        """
        return data


def traverse_args(args, kwargs, func):
    # 对所有的参数进行处理
    _args = []
    for arg in args:
        _args.append(func(arg))

    _kwargs = {}
    for k, v in kwargs.items():
        _kwargs[k] = func(v)

    return _args, _kwargs
