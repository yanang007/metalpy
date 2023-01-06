from metalpy.utils.object_path import ObjectPath


def reget_object(obj):
    """用于获取被劫持后的对象
    
    由于mexin中的劫持是通过替换其上层对象的成员实现的，
    因此在劫持发生前获取的引用不会被劫持影响。

    Parameters
    ----------
    obj
        劫持前获取的对象

    Returns
    -------
    ret
        同路径下的对象，若没有劫持行为，则返回值和输入指向同一个对象
    """
    return ObjectPath.of(obj).resolve()
