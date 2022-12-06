import copy
from typing import Any, Union, Iterable

import numpy as np
import tqdm
from discretize import TensorMesh

from metalpy.mepa import LinearExecutor
from .object import Object
from .shapes import Shape3D
from .shapes.shape3d import bounding_box_of


class Scene:
    def __init__(self):
        self.objects: list[Object] = []

    @staticmethod
    def of(*shapes: Shape3D, models: Union[Any, dict, list[Any], list[dict], None] = None):
        ret = Scene()
        if not isinstance(models, list):
            models = [models] * len(shapes)
        for shape, value in zip(shapes, models):
            ret.append(shape, value)

        return ret

    def append(self, shape: Shape3D, models: Union[dict[str, Any], Any]) -> Object:
        """添加三维几何体

        Parameters
        ----------
        shape
            三维几何体
        models
            三维几何体的参数

        Returns
        -------
            返回构造的三维几何体
        """
        obj = Object(shape, models)
        self.objects.append(obj)

        return obj

    @property
    def bounds(self):
        return bounding_box_of(self.shapes())

    @staticmethod
    def build_mesh_worker(objects: list[Object], mesh, show_modeling_progress, worker_id):
        models = {}
        for obj in tqdm.tqdm(objects, position=0, leave=False, ncols=80) if show_modeling_progress else objects:
            shape = obj.shape

            # place的结果应该为布尔数组或范围为[0, 1]的数组，指示对应网格位置是否有效或有效程度
            ind = shape.place(mesh, worker_id)
            current_mask = ind if ind.dtype == bool else ind != 0  # TODO: 考察是否有必要视0为非活动网格

            for key, current_value in obj.items():
                if ind.dtype == bool:
                    current_layer = np.zeros_like(ind, dtype=type(current_value))
                    current_layer[ind] = current_value
                else:
                    current_layer = ind * current_value

                if key not in models:
                    models[key] = current_layer
                else:
                    prev_layer = models[key]
                    filled_ind = prev_layer != 0  # TODO: 同上
                    overlapping_mask = filled_ind & current_mask
                    non_overlapping_mask = current_mask ^ overlapping_mask

                    prev_layer[overlapping_mask] = obj.mix(
                        prev_layer[overlapping_mask],
                        current_layer[overlapping_mask]
                    )
                    prev_layer[non_overlapping_mask] = current_layer[non_overlapping_mask]

        return models

    def build_model(self, mesh,
                    executor=None,
                    progress=False) -> Union[np.ndarray, dict[str, np.ndarray]]:
        """在给定网格上构建模型

        Parameters
        ----------
        mesh
            用于构建模型的网格
        executor
            并行执行器
        progress
            是否显示进度条

        Returns
        -------
            若所有模型只包含默认键，则返回一个数组，为默认键下的值的构建结果

            否则返回字典，包含所有键以及该键下的值的构建结果

        Notes
        -----
            模型假设为非0值
        """
        if executor is None:
            executor = LinearExecutor(1)

        mesh_centers = np.asarray(mesh.cell_centers)
        input_mesh = executor.arrange_single(mesh_centers)

        futures = []
        for i, worker in enumerate(executor.get_workers()):
            futures.append(
                executor.submit(self.build_mesh_worker, self.objects, input_mesh.assign(worker),
                                worker_id=i,
                                show_modeling_progress=i == 0 if progress else False,
                                worker=worker, )
            )
        executor.gather(futures)

        models_dict = {}
        for key in futures[0].result().keys():
            models_dict[key] = np.concatenate([future.result()[key] for future in futures])

        if len(models_dict) == 1 and Object.DEFAULT_KEY in models_dict:
            return models_dict[Object.DEFAULT_KEY]
        else:
            return models_dict

    def create_mesh(self, cell_size=None, n_cells=None) -> Union[TensorMesh]:
        """根据场景边界构建网格

        Parameters
        ----------
        cell_size : number or array(3,)
            网格长宽高相等或分别定义网格x, y, z方向宽度大小
        n_cells : number or array(3,)
            总网格数或x, y, z方向网格数

        Returns
        -------
            构建的网格

        Notes
        -----
            原点是场景的边界的最小值。
            若指定cell_size，则边界点保证大于场景边界；
            若指定n_cells，则边界点是场景的边界的最大值。

            若n_cells为单个值，则用于指定总网格数，保证生成的网格数小于等于该值。
        """
        if cell_size is not None:
            if not isinstance(cell_size, Iterable):
                cell_size = [cell_size] * 3
            cell_size = np.asarray(cell_size)
            bounds = self.bounds
            n_cells = np.ceil((bounds[1::2] - bounds[::2]) / cell_size).astype(int)
        else:
            bounds = self.bounds
            sizes = bounds[1::2] - bounds[::2]
            if not isinstance(n_cells, Iterable):
                avg_grids = (n_cells / np.prod(sizes)) ** (1 / 3)
                n_cells = (avg_grids * sizes).astype(int)
                cell_size = [1 / avg_grids] * 3
            else:
                n_cells = np.asarray(n_cells)
                cell_size = sizes / n_cells

        return TensorMesh([[(d, n)] for d, n in zip(cell_size, n_cells)], origin=bounds[::2])

    def build(self, cell_size=None, n_cells=None, executor=None, progress=False):
        """根据给定的网格尺寸，构建场景的网格和模型，是create_mesh和build_model的组合

        Parameters
        ----------
        cell_size : number or array(3,)
            网格长宽高相等或分别定义网格x, y, z方向宽度大小
        n_cells : number or array(3,)
            总网格数或x, y, z方向网格数
        executor
            并行执行器
        progress
            是否显示进度条

        Returns
        -------
            (网格，模型)
            网格输出同create_mesh
            模型输出同build_model

        See Also
        --------
            Scene.create_mesh
            Scene.build_model
        """
        mesh = self.create_mesh(cell_size=cell_size, n_cells=n_cells)
        model = self.build_model(mesh, executor=executor, progress=progress)

        return mesh, model

    def __iter__(self) -> Iterable[Object]:
        for obj in self.objects:
            yield obj

    def __getitem__(self, item) -> Object:
        return self.objects[item]

    def shapes(self) -> Iterable[Shape3D]:
        for obj in self.objects:
            yield obj.shape

    def to_multiblock(self):
        import pyvista as pv

        ret = pv.MultiBlock()
        for shape in self.shapes():
            ret.append(shape.to_polydata())

        return ret

    @staticmethod
    def mesh_to_polydata(mesh, models: Union[np.ndarray, dict[str, np.ndarray]]):
        if not isinstance(models, dict):
            models = {'active': models}
        else:
            models = copy.copy(models)

        for key in models:
            if models[key].dtype == bool:
                models[key] = models[key].astype(np.int8)

        grids = mesh.to_vtk(models=models)

        for key in models:
            # 将第一个model设为active
            grids.set_active_scalars(key)
            break

        return grids
