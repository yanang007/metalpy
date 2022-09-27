import tqdm
from SimPEG.simulation import BaseSimulation

from metalpy.mexin import Mixin
from metalpy.mexin.patch import Patch
from metalpy.mexin.injectors import after


class Progress(Mixin):
    def __init__(self, this: BaseSimulation):
        super().__init__(this)
        self.progressbar = tqdm.tqdm(total=len(this.survey.receiver_locations))
        self.manual_update = False  # 指示是否有其它插件在手动更新进度条

        # TODO: 引入一套新的注解来支持不同的mixin方法注入
        @after(this, 'evaluate_integral')
        def wrapper(_, *args, **kwargs):
            if not self.manual_update:
                self.progressbar.update(1)

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


class Progressed(Patch):
    def __init__(self):
        super().__init__()

    def apply(self):
        self.add_mixin(BaseSimulation, Progress)
