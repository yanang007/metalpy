from metalpy.scab.modelling.shapes.bounds import Bounds


class Bounded(Bounds):
    def __init__(self, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None):
        """指示可选的边界

        Parameters
        ----------
        xmin, xmax, ymin, ymax, zmin, zmax
            指定边界或为None
        """
        super().__init__(xmin, xmax, ymin, ymax, zmin, zmax)

    def __str__(self):
        desc = []
        b_desc = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
        for i in range(6):
            b = self.bounds[i]
            if b is not None:
                desc.append(f'{b_desc[i]}={b}')
        return f'Bounded({", ".join(desc)})'
