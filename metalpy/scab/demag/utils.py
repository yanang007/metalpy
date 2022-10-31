from typing import Union

import numpy as np


def get_prolate_spheroid_demag_factor(eps, polar_axis=2):
    """
    获取长旋转椭球体的退磁系数（a, a, c, c > a）

    Parameters
    ----------
    eps : float
        椭球长短轴比 (c / a)

    polar_axis : int
        长轴所在轴方向，0、1、2代表x、y、z，默认为2，即z轴

    Returns
    -------
    长旋转椭球体的退磁系数
    """
    ratio = eps
    ratio2m1 = ratio * ratio - 1
    sratio2m1 = np.sqrt(ratio2m1)
    Na = (ratio / sratio2m1 * np.log(ratio + sratio2m1) - 1) / ratio2m1
    Nc = (1 - Na) / 2
    N = np.asarray([Nc, Nc, Nc])
    N[polar_axis] = Na

    return N


def get_demag_apparent_k(k: Union[float, np.ndarray], N):
    """
    计算退磁作用下的视磁化率

    Parameters
    ----------
    k : float | array-like(3,) | array-like(3,n)
        均匀介质磁化率
    N : array-like(3,)
        三个方向上的退磁系数

    Returns
    -------
    out : ndarray(3,n)
        视磁化率
    """
    N = np.asarray(N)
    k = np.asarray(k)

    if k.ndim < 2:
        k = k[..., None]

    return k / (1 + k * N)
