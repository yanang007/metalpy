import pyvista as pv

from . import Shape3D


class Obj2(Shape3D):
    def __init__(self, model, dx, dy, dz):
        """
        :param model: 底面多边形顶点列表
        :param dx, dy, dz: 模型网格尺寸
        """
        super().__init__()
        if model is str:
            model = pv.fro
        self.model = model
        self.grid_size = np.asarray([dx, dy, dz])

    def place(self, mesh_cell_centers, worker_id):
        grids = mesh_cell_centers / self.grid_size
        grids = grids.astype(int)

        indices = np.zeros(mesh_cell_centers.shape[0])

        for i, coord in enumerate(grids):
            if np.any(coord >= self.model.shape):
                continue
            indices[i] = self.model[tuple(coord)]

        return indices

    def __hash__(self):
        return hash((*self.model.flatten(), *self.model.shape, *self.grid_size))

    def clone(self):
        return Obj(self.model.clone(), self.dx, self.dy, self.dz)

    def plot(self, ax, color):
        pass
