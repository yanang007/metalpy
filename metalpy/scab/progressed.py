import tqdm
from SimPEG.simulation import BaseSimulation

from metalpy.mexin import Mixin
from metalpy.mexin.patch import Patch
from metalpy.mexin.injectors import after
from .distributed.policies import DistributeOnce


class Progress(Mixin):
    def __init__(self, this: BaseSimulation):
        super().__init__(this)
        self.progressbar = tqdm.tqdm(total=len(this.survey.receiver_locations))
        self.progressbar.clear()
        self.manual_update = True  # 指示是否有其它插件在手动更新进度条

    @property
    def progressbar(self):
        return self._progressbar

    @progressbar.setter
    def progressbar(self, value):
        self._progressbar = value

    def get_progressbar(self, this):
        return self.progressbar

    def set_manual_update(self, this, manual_update: bool = True):
        self.manual_update = manual_update

    def update(self, this, count):
        if self.progressbar is not None:
            if not self.manual_update:
                self.progressbar.reset()
                self.manual_update = True  # 切换为用户更新模式，此前可能已被自动更新过，因此需要重置
            self.progressbar.update(count)


class Progressed(Patch, DistributeOnce):
    def __init__(self):
        super().__init__()

    def apply(self):
        if len(self.context.get_patches()) == 1:
            raise RuntimeError("Progressed patch cannot be used alone due to SimPEG's design")

        self.add_mixin(BaseSimulation, Progress)
