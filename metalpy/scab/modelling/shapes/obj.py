import numpy as np

from . import Shape3D


class Obj(Shape3D):
    def __init__(self, model, dx, dy, dz):
        """
        :param model: 底面多边形顶点列表
        :param dx, dy, dz: 模型网格尺寸
        """
        super().__init__()
        self.model = model
        self.grid_size = np.asarray([dx, dy, dz])

    def do_place(self, mesh_cell_centers, worker_id):
        grids = mesh_cell_centers / self.grid_size
        grids = grids.astype(int)

        indices = np.zeros(mesh_cell_centers.shape[0])

        for i, coord in enumerate(grids):
            if np.any(coord >= self.model.shape):
                continue
            indices[i] = self.model[tuple(coord)]

        return indices

    def do_hash(self):
        return hash((*self.model.flatten(), *self.model.shape, *self.grid_size))

    def do_clone(self):
        return Obj(self.model.clone(), *self.grid_size)
