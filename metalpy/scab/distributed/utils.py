from metalpy.mexin.injectors import get_nest


def reget_class(cls):
    """用于获取被劫持的cls对应的Replacement"""
    return getattr(get_nest(cls), cls.__name__)
