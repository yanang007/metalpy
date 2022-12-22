import numpy as np


# return the hash of the given string in string format
def hash_str(*objs):
    flattened = []
    for obj in objs:
        if isinstance(obj, list):
            flattened.extend(obj)
        else:
            flattened.append(obj)

    flattened = tuple(flattened)
    hash_value = hash(flattened)

    return create_hash_str(hash_value)


def create_hash_str(hash_value, max_width=6):
    char_table = '0123456789abcdef'
    tablesize = len(char_table)
    hash_string = ''

    while hash_value != 0 and len(hash_string) < max_width:
        c = hash_value % tablesize
        hash_string = char_table[c] + hash_string
        hash_value = hash_value // tablesize

    return hash_string


def hash_numpy_array(arr: np.ndarray, n_samples=10, sparse=False):
    if not sparse:
        arr = arr.ravel()
        rand = np.random.RandomState(int(arr[len(arr) // 2]))

        # 加入shape作为参数防止类似np.ones(100)和np.ones(1000)的冲突
        return hash((arr.shape, *rand.choice(arr, min(len(arr), n_samples), replace=False),))
    else:
        # 稀疏的数组可能会在采样时碰撞概率较大
        # 但是涉及压缩操作可能会导致效率一定程度上降低
        import blosc2
        packed = blosc2.pack_array2(arr)

        return hash(packed)
