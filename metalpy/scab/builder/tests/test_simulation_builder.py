from metalpy.mexin.utils.type_map import assert_all_types_exists
from metalpy.scab.builder import SimulationBuilder


def test_builders_map():
    assert_all_types_exists(SimulationBuilder._registry)
