from __future__ import annotations

from typing import Iterable

from SimPEG import maps
from SimPEG.potential_fields import magnetics

from metalpy.scab.utils.misc import define_inducing_field
from .. import BasePFSimulationBuilder
from ... import SimulationBuilder


class Simulation3DIntegralBuilder(BasePFSimulationBuilder):
    def __init__(self, sim_cls: magnetics.Simulation3DIntegral):
        super().__init__(sim_cls)
        self._receiver_list = []
        self._source_field = None

        self.source_field(50000, 90, 0)

    def build(self) -> magnetics.Simulation3DIntegral:
        return super().build()

    def receivers(self, receiver_points, components: str | Iterable[str] = 'tmi'):
        self._receiver_list.append(
            magnetics.receivers.Point(
                receiver_points, components=components
            ))

    def source_field(self, strength, inc, dec):
        self._source_field = define_inducing_field(strength, inc, dec)

    @SimulationBuilder._assembles('survey')
    def _survey(self):
        field = magnetics.sources.SourceField(
            receiver_list=self._receiver_list, parameters=self._source_field
        )
        survey = magnetics.survey.Survey(field)

        return survey

    @SimulationBuilder._supplies('model_type')
    def model_type(self, model_type, **_):
        return model_type

    def scalar_model(self):
        return self.model_type('scalar')

    def vector_model(self):
        return self.model_type('vector')

    @SimulationBuilder._assembles('chiMap')
    def _chi_map(self):
        return maps.IdentityMap(nP=self._model_mesh.n_active_cells)
