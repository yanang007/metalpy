from SimPEG.simulation import BaseSimulation

from metalpy.scab.injectors import extends, replaces

from .extensions.potential_fields.magnetics import ext_torch

@extends(BaseSimulation, 'torch_on')
def __BaseSimulation_ext_progress_on(self, device='cpu'):
    self.torch_device = device

