import sys

import tqdm
from SimPEG.simulation import BaseSimulation

from .injectors import extends, after


@extends(BaseSimulation, 'progressbar')
@property
def __BaseSimulation_ext_progressbar(self):
    if not hasattr(self, '_progressbar'):
        self._progressbar = None
    return self._progressbar


@extends(BaseSimulation, 'progressbar')
@__BaseSimulation_ext_progressbar.setter
def __BaseSimulation_ext_progressbar(self, p):
    if hasattr(self, '_progressbar'):
        ret = self._progressbar
    else:
        ret = None
    self._progressbar = p
    return ret


@extends(BaseSimulation, 'progress_on')
def __BaseSimulation_ext_progress_on(self):
    self.progressbar = tqdm.tqdm(total=len(self.survey.receiver_locations))
    self.manual_update = False  # 指示是否有其它插件在手动更新进度条

    @after(self, 'evaluate_integral')
    def wrapper(*args, **kwargs):
        if not self.manual_update:
            self.progressbar.update(1)


@extends(BaseSimulation, 'set_manual_update')
def __BaseSimulation_ext_update_progress(self):
    self.manual_update = True


@extends(BaseSimulation, 'update_progress')
def __BaseSimulation_ext_update_progress(self, count):
    if self.progressbar is not None:
        if not self.manual_update:
            self.progressbar.reset()
            self.manual_update = True  # 切换为用户更新模式，此前可能已被自动更新过，因此需要重置
        self.progressbar.update(count)

