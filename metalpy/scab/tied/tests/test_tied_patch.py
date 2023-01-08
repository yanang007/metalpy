from metalpy.mexin.utils.type_map import assert_all_types_exists
from metalpy.scab.tied.tied import TaichiContext


def test_tied_impls_map():
    assert_all_types_exists(TaichiContext._implementations)
