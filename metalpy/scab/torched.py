import torch
from SimPEG.simulation import BaseSimulation

from metalpy.scab.injectors import extends

from .extensions.potential_fields.magnetics import ext_torch
from .utils.patch import use_patch

use_patch(ext_torch)


@extends(BaseSimulation, 'torch_on')
def __BaseSimulation_ext_torch_on(self, device='cpu'):
    self.torch_device = torch.device(device)
    self.torch_on_impl()

