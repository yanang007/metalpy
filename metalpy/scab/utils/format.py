from typing import Sequence

import numpy as np
import pandas as pd


def check_components(components):
    if isinstance(components, str):
        return components,
    elif not isinstance(components, Sequence):
        return tuple(components)
    else:
        return components


def format_numpy(data: np.ndarray, components, receiver_locations=None):
    components = check_components(components)
    ret = data.reshape(-1, len(components))
    if receiver_locations is not None:
        ret = np.c_[receiver_locations, ret]

    return ret


def format_pandas(data, components, receiver_locations=None):
    components = check_components(components)
    ret = format_numpy(data, components, receiver_locations)

    headers = components
    if receiver_locations is not None:
        headers = ('x', 'y', 'z', *headers)

    return pd.DataFrame(ret, columns=headers)
