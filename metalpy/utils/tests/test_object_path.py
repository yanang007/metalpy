import functools

from metalpy.utils.object_path import get_full_qualified_path, reassign_object_name, mock_object


class XX:
    class YY:
        def zz(self):
            pass


class Target:
    pass


def test_full_qualname():
    assert get_full_qualified_path(XX.YY.zz).endswith('XX.YY.zz')

    XX.YY.zz.__module__ = __name__
    assert get_full_qualified_path(XX.YY.zz) == f'{__name__}:XX.YY.zz'

    XX.YY.zz.__name__ = 'ZZ'
    assert get_full_qualified_path(XX.YY.zz) != f'{__name__}:XX.YY.ZZ'  # 只改了__name__但没有改__qualname__

    reassign_object_name(XX.YY.zz, new_name='ZZ')
    assert get_full_qualified_path(XX.YY.zz) == f'{__name__}:XX.YY.ZZ'

    mock = XX()
    functools.wraps(Target)(mock)
    assert get_full_qualified_path(mock) == get_full_qualified_path(Target)
