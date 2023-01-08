from metalpy.utils.object_path import get_full_qualified_path


class XX:
    class YY:
        def zz(self):
            pass


def test_full_qualname():
    assert get_full_qualified_path(XX.YY.zz).endswith('XX.YY.zz')

    XX.YY.zz.__module__ = __name__
    assert get_full_qualified_path(XX.YY.zz) == f'{__name__}:XX.YY.zz'

    XX.YY.zz.__name__ = 'ZZ'
    assert get_full_qualified_path(XX.YY.zz) == f'{__name__}:XX.YY.ZZ'

    mock = XX()
    mock.__module__ = __name__
    mock.__name__ = 'Target'
    assert get_full_qualified_path(mock) == f'{__name__}:Target'
