from random import random

import numpy as np


def define_inducing_field(strength, inclination, declination):
    """
    :param strength: 场强
    :param inclination: 磁倾角
    :param declination: 磁偏角
    :return:
    """
    inducing_field = (strength, inclination, declination)

    return inducing_field


def plot_opaque_cube(ax, x=10, y=20, z=30, dx=40, dy=50, dz=60, alpha=1, **kwargs):
    xx = np.linspace(x, x + dx, 2)
    yy = np.linspace(y, y + dy, 2)
    zz = np.linspace(z, z + dz, 2)

    xx2, yy2 = np.meshgrid(xx, yy)

    ax.plot_surface(xx2, yy2, np.full_like(xx2, z), alpha=alpha, **kwargs)
    ax.plot_surface(xx2, yy2, np.full_like(xx2, z + dz), alpha=alpha, **kwargs)

    yy2, zz2 = np.meshgrid(yy, zz)
    ax.plot_surface(np.full_like(yy2, x), yy2, zz2, alpha=alpha, **kwargs)
    ax.plot_surface(np.full_like(yy2, x + dx), yy2, zz2, alpha=alpha, **kwargs)

    xx2, zz2 = np.meshgrid(xx, zz)
    ax.plot_surface(xx2, np.full_like(yy2, y), zz2, alpha=alpha, **kwargs)
    ax.plot_surface(xx2, np.full_like(yy2, y + dy), zz2, alpha=alpha, **kwargs)

def plot_linear_cube(ax, x, y, z, dx, dy, dz, color='red'):
    xx = [x, x, x + dx, x + dx, x]
    yy = [y, y + dy, y + dy, y, y]
    kwargs = {'alpha': 1, 'color': color}
    ax.plot3D(xx, yy, [z] * 5, **kwargs)
    ax.plot3D(xx, yy, [z + dz] * 5, **kwargs)
    ax.plot3D([x, x], [y, y], [z, z + dz], **kwargs)
    ax.plot3D([x, x], [y + dy, y + dy], [z, z + dz], **kwargs)
    ax.plot3D([x + dx, x + dx], [y + dy, y + dy], [z, z + dz], **kwargs)
    ax.plot3D([x + dx, x + dx], [y, y], [z, z + dz], **kwargs)


def sin2pi(a):
    return np.sin(2 * np.pi * a)


def cos2pi(a):
    return np.cos(2 * np.pi * a)


def rand_between(vmin, vmax):
    return (vmax - vmin) * random() + vmin
