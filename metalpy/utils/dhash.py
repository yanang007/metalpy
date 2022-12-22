import sys

import numpy as np

from metalpy.utils.type import get_or_default


def dhash(*objs):
    """deterministic hash

    Parameters
    ----------
    objs

    Returns
    -------

    """
    return DHash(*objs)


def register_dhasher(hashed_type):
    def wrapper(func):
        DHash.hashers[hashed_type] = func
        return func
    return wrapper


class DHash:
    hashers = {}

    def __init__(self, *objs, convert=True):
        """

        Parameters
        ----------
        objs
            待哈希的对象

        Warnings
        --------
            DHash.__init__本身不会对输入的对象进行任何转换，默认均为dhash可接受对象
            请优先使用dhash()函数


        """
        if convert:
            self.objs = tuple(DHash.convert_to_dhashable(obj) for obj in objs)
        else:
            self.objs = objs
        self.result = None

    def __iter__(self):
        for obj in self.objs:
            if isinstance(obj, DHash):
                for o in obj:
                    yield o
            else:
                yield obj

    def digest(self):
        if self.result is None:
            self.result = hash(self.objs)
        return self.result

    def hexdigest(self, digits=32):
        hash_value = self.digest()
        char_table = '0123456789abcdef'
        tablesize = len(char_table)
        hash_string = ''

        while hash_value != 0 and len(hash_string) < digits:
            c = hash_value % tablesize
            hash_string = char_table[c] + hash_string
            hash_value = hash_value // tablesize

        return hash_string

    def __str__(self):
        return self.hexdigest(6)

    def __dhash__(self):
        return self

    def __hash__(self):
        return self.digest()

    @staticmethod
    def convert_to_dhashable(obj):
        t = type(obj)
        hasher = get_or_default(DHash.hashers, t, None)

        if hasher is None:
            hasher = getattr(t, '__dhash__', lambda x: DHash(x, convert=False))

        return hasher(obj)


@register_dhasher(list)
def _hash_list(obj: list):
    return dhash(*obj)


@register_dhasher(tuple)
def _hash_list(obj: tuple):
    return dhash(*obj)


@register_dhasher(str)
def _hash_str(obj: str):
    from functools import reduce
    with np.errstate(over='ignore'):  # 消除上溢警告
        order = np.uint64(5)  # 解决uint64左移运算符的类型错误 https://github.com/numpy/numpy/issues/2524
        ret = reduce(lambda h, c: np.uint64(ord(c)) + ((h << order) + h), obj, np.uint64(5381))
    return ret


@register_dhasher(bytes)
def _hash_bytes(obj: bytes):
    buf = np.frombuffer(obj, dtype=np.uint8)

    from functools import reduce
    with np.errstate(over='ignore'):
        order = np.uint64(5)
        ret = reduce(lambda h, c: c + ((h << order) + h), buf, np.uint64(5381))
    return ret


@register_dhasher(dict)
def _hash_dict(obj: dict):
    return dhash(*sorted(obj.items(), key=lambda x: x[0]))


@register_dhasher(np.ndarray)
def _hash_array(arr: np.ndarray, n_samples=10, sparse=False):
    if not sparse:
        arr = arr.ravel()
        rand = np.random.RandomState(int(arr[len(arr) // 2]))
        n_samples = min(n_samples, len(arr))

        # 加入shape作为参数防止类似np.ones(100)和np.ones(1000)的冲突
        return dhash((arr.shape, *rand.choice(arr, min(len(arr), n_samples), replace=False),))
    else:
        # 稀疏的数组可能会在采样时碰撞概率较大
        # 但是涉及压缩操作可能会导致效率一定程度上降低
        import blosc2
        packed = blosc2.pack_array2(arr)

        return dhash(packed)
