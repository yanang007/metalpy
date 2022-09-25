import torch
from SimPEG.simulation import BaseSimulation

from metalpy.scab.injectors import extends

from .extensions.potential_fields.magnetics import ext_torch
from .utils.patch import use_patch

use_patch(ext_torch)


@extends(BaseSimulation, 'torch_on')
def __BaseSimulation_ext_torch_on(self, device='cpu'):
    self.torch_device = torch.device(device)

    extends(self, 'estimate_memory_cost')(__BaseSimulation_ext_estimate_memory_cost)
    extends(self, 'estimate_batch_size')(__BaseSimulation_ext_estimate_batch_size)

    self.torch_on_impl()


def __BaseSimulation_ext_estimate_memory_cost(self, size=None):
    # size = self.survey.receiver_locations.shape[0] if size is None else size
    # return 100 * size * self.modelMap.shape[0]

    return None


def __BaseSimulation_ext_estimate_batch_size(self, memory):
    return int(memory / self.estimate_memory_cost(1))
