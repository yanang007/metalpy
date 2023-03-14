import numpy as np
from metalpy.utils.time import Timer

from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Prism


def main(resolution):
    h0, h1 = 11.5, 27

    hh = h0 + 2.6
    ddx, ddy = 2, -28
    r0 = 14
    r1 = r0 + 0.68
    resolution //= 2
    theta2 = np.linspace(-np.pi / 6, -np.pi * 5 / 6, resolution)
    txy = np.c_[np.cos(theta2), np.sin(theta2)]
    p0 = txy * r0
    p1 = (txy * r1)[::-1]

    ret = [
        Prism(np.r_[p0, p1], h0, hh).translate(ddx, ddy, 0),
    ]

    _ = Scene.of(*ret).to_multiblock()


if __name__ == '__main__':
    t = Timer()
    x = [20, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000]
    y = []
    for res in x:
        with t:
            main(res)
        print(f'{res}: {t}')
        y.append(t.elapsed / 1e6)

    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.xlabel('Num Vertices')
    plt.ylabel('Running Time (ms)')
    plt.show()
