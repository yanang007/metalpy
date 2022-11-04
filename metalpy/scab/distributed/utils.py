from metalpy.mexin.injectors import get_object_by_path, get_class_path


def reget_class(cls):
    """用于获取被劫持的cls对应的Replacement"""
    return get_object_by_path(get_class_path(cls))
