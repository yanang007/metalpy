from distributed import Client

from .executor import Executor
from .dask_helper import configure_dask_client
from .worker import Worker


class DaskExecutor(Executor):
    def __init__(self, scheduler_addr, n_units=None, extra_paths=None, excludes=None):
        """
        构造基于dask.distributed的执行器
        :param scheduler_addr: 调度器地址，包括协议和端口号
        :param n_units: 工作单位数，实际为 min(n_units, n_actual_working_units)
        """
        super().__init__()
        self.client = Client(scheduler_addr)
        configure_dask_client(self.client, extra_paths=extra_paths, excludes=excludes)

        worker_groups = {}
        for addr, worker in self.client.scheduler_info()['workers'].items():
            host = worker['host']
            if host not in worker_groups:
                worker_groups[host] = []
            worker_groups[host].append(worker['name'])

        self.workers = []
        n_actual_working_units = 0
        load_weight = {}

        for host, workers in worker_groups.items():
            for worker in workers:
                worker_instance = Worker(worker, weight=1, verbose=False)
                worker_name = worker_instance.name

                if worker_name in load_weight:
                    weight = load_weight[worker_name]
                else:
                    weight = 1

                worker_instance.weight = weight
                self.workers.append(worker_instance)
                n_actual_working_units += weight

        if n_units is None:
            self.n_units = n_actual_working_units
        else:
            self.n_units = n_units

    def do_submit(self, func, *args, workers=None, **kwargs):
        if isinstance(workers, Worker):
            workers = [workers]

        if workers is not None:
            targets = [worker.get_name() for worker in workers]
            kwargs['workers'] = targets
            kwargs.setdefault('allow_other_workers', True)

        return self.client.submit(func, *args, **kwargs)

    def get_workers(self):
        return self.workers

    def get_n_units(self):
        return self.n_units

    def is_local(self):
        return False

    def gather(self, futures):
        return self.client.gather(futures)

    def scatter(self, data):
        return self.client.scatter(data)
