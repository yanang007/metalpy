from __future__ import annotations

import os

from SimPEG.potential_fields.base import BasePFSimulation

from metalpy.utils.type import undefined
from metalpy.scab.modelling.modelled_mesh import ModelledMesh
from .. import SimulationBuilder


class BasePFSimulationBuilder(SimulationBuilder):
    def __init__(self, sim_cls: BasePFSimulation):
        super().__init__(sim_cls)
        self._model_mesh: ModelledMesh | None = None

    @SimulationBuilder._supplies('mesh', ['ind_active', 'actInd'])
    def active_mesh(self, mesh, ind_active=undefined, **_):
        if not isinstance(mesh, ModelledMesh):
            ind = ind_active if ind_active != undefined else None
            self._model_mesh = ModelledMesh(mesh, ind_active=ind)
        else:
            self._model_mesh = mesh
            if ind_active == undefined:
                ind_active = mesh.active_cells
            mesh = mesh.base_mesh

        return mesh, ind_active

    @SimulationBuilder._supplies('store_sensitivities', 'sensitivity_path')
    def store_sensitivities(self, target_or_path: bool | str):
        if target_or_path is True:
            return 'ram', undefined
        elif target_or_path is False:
            return 'forward_only', undefined
        else:
            return 'disk', os.fspath(target_or_path)
