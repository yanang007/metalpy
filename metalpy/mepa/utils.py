import traceback


def structured_traverse(struct, func):
    if isinstance(struct, dict):
        ret = {key: structured_traverse(element, func) for key, element in struct.items()}
    elif isinstance(struct, list):
        ret = [structured_traverse(element, func) for element in struct]
    else:
        ret = func(struct)

    return ret


def exception_caught(func, *args, **kwargs):
    """用于处理Loki无法正确识别并捕获异常的问题
    TODO: 参考multiprocessing进行重新创建堆栈，从而帮助调试
    """
    try:
        return func(*args, **kwargs)
    except BaseException as e:
        traceback.print_exc()
        raise e
