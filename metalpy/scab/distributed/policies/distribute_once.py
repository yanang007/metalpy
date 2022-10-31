from metalpy.mepa import Worker
from .distributable import Distributable


class DistributeOnce(Distributable):
    def __init__(self):
        super().__init__()

    def should_distribute_to(self, worker: Worker):
        return worker.get_in_group_id() == 0
