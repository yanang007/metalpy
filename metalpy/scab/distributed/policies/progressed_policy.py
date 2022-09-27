from metalpy.mepa import Worker
from .patch_policy import PatchPolicy


class ProgressedPolicy(PatchPolicy):
    def __int__(self):
        pass

    def __call__(self, patch, worker: Worker):
        return worker.get_in_group_id() == 0
