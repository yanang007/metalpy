import torch
from SimPEG.simulation import BaseSimulation

from metalpy.mexin.injectors import extends
from metalpy.mexin import Mixin
from metalpy.mexin import Patch


class TorchContext(Mixin):
    def __init__(self, this, device='cuda'):
        super().__init__(this)
        self._device = device

    def post_apply(self, this):
        path = type(this).__module__
        dispatch = path.replace('SimPEG.', '').replace('.simulation', '')
        dispatch = f'metalpy.scab.extensions.{dispatch}.ext_torch'

        try:
            impl: Mixin = __import__(dispatch, fromlist=['TorchedImpl']).TorchedImpl
        except:
            print(f'Torch support for {path} is not implemented. Ignoring it.')
            return

        this.mixins.add(impl)

    def get_torch_device(self, this):
        return self._device

    @staticmethod
    def estimate_memory_cost(this, size=None):
        # size = self.survey.receiver_locations.shape[0] if size is None else size
        # return 100 * size * self.modelMap.shape[0]

        return None

    @staticmethod
    def estimate_batch_size(this, memory):
        return int(memory / this.estimate_memory_cost(1))


class Torched(Patch):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

    def apply(self):
        self.add_mixin(BaseSimulation, TorchContext, self.device)
