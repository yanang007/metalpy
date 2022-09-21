import sys

import tqdm
from SimPEG.simulation import BaseSimulation

from .injectors import extends, before


@extends(BaseSimulation, 'progress_on')
def __BaseSimulation_ext_progress_on(self):
    progressbar = tqdm.tqdm(total=len(self.survey.receiver_locations))

    @before(self, 'evaluate_integral')
    def wrapper(*args, **kwargs):
        progressbar.update(1)

