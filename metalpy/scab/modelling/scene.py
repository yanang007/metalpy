import copy
import os.path
from typing import Any, Union, Iterable

import numpy as np
import tqdm
from discretize import TensorMesh

from metalpy.mepa import LinearExecutor
from metalpy.utils.file import ensure_dir, make_cache_directory
from metalpy.utils.dhash import dhash
from metalpy.scab.utils.hash import dhash_discretize_mesh
from .layer import Layer
from .mix_modes import MixMode
from .object import Object
from .modelled_mesh import ModelledMesh
from .shapes import Shape3D
from .shapes.bounds import Bounds
from .shapes.full_space import FullSpace
from .shapes.shape3d import bounding_box_of


class Scene:
    default_cache_dir = make_cache_directory('models')
    INACTIVE_BOOL = False
    INACTIVE_INT = 0
    INACTIVE_FLOAT = np.nan

    def __init__(self):
        self.objects_layer = Layer()
        self.backgrounds_layer = Layer(mix_mode=MixMode.KeepOriginal)

    @staticmethod
    def of(*shapes: Shape3D, models: Union[Any, dict, list[Any], list[dict], None] = None):
        ret = Scene()
        if not isinstance(models, list):
            models = [models] * len(shapes)
        for shape, value in zip(shapes, models):
            ret.append(shape, value)

        return ret

    def with_background(self, value, region_shape=None):
        self.append_background(value, region_shape)
        return self

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
        ret
            返回构造的三维几何体
        """
        obj = Object(shape, models)
        self.objects_layer.append(obj)

        return obj

    def append_background(self, value, region_shape=None):
        if region_shape is None:
            region_shape = FullSpace()
        obj = Object(region_shape, value)
        self.backgrounds_layer.append(obj)

        return obj

    @property
    def bounds(self):
        return bounding_box_of(self.shapes)

    def build_model(self, mesh,
                    executor=None,
                    progress=False,
                    cache=None,
                    cache_dir=None,
                    ) -> ModelledMesh:
        """在给定网格上构建模型

        Parameters
        ----------
        mesh
            用于构建模型的网格
        executor
            并行执行器
        progress
            是否显示进度条
        cache
            控制缓存文件，若为bool值，则指示是否启用缓存；若为str，则指示缓存文件的路径
        cache_dir
            控制缓存文件夹，指示放置缓存文件的文件夹，文件名会使用默认规则生成

        Returns
        -------
        ret
            若所有模型只包含默认键，则返回一个数组，为默认键下的值的构建结果

            否则返回字典，包含所有键以及该键下的值的构建结果

        Notes
        -----
            模型假设为非0值

            cache和cache_dir任意一个不为空时则会启动缓存，但如果cache和cache_dir同时被指定，则

            如果cache为True，则使用cache_dir与默认规则文件名

            如果cache为路径，则使用cache指定的文件路径
        """
        cache_filepath = self._determine_cache_filepath(mesh, cache, cache_dir)

        if cache_filepath is not None and os.path.exists(cache_filepath):
            import pickle
            with open(cache_filepath, 'rb') as f:
                models_dict = pickle.load(f)
        else:
            if executor is None:
                executor = LinearExecutor(1)

            mesh_centers = np.asarray(mesh.cell_centers)
            input_mesh = executor.arrange_single(mesh_centers)

            futures = []
            for i, worker in enumerate(executor.get_workers()):
                futures.append(
                    executor.submit(self._build_mesh_worker, self.layers, input_mesh.assign(worker),
                                    worker_id=i,
                                    show_modeling_progress=i == 0 if progress else False,
                                    worker=worker, )
                )
            executor.gather(futures)

            models_dict = {}
            for key in futures[0].result().keys():
                models_dict[key] = np.concatenate([future.result()[key] for future in futures])

            if cache_filepath is not None:
                import pickle
                with open(cache_filepath, 'wb') as f:
                    pickle.dump(models_dict, f)

        return ModelledMesh(mesh, models_dict)

    def create_mesh(self, cell_size=None, n_cells=None, bounds=None,) -> Union[TensorMesh]:
        """根据场景边界构建网格

        Parameters
        ----------
        cell_size : number or array(3,)
            网格长宽高相等或分别定义网格x, y, z方向宽度大小
        n_cells : number or array(3,)
            总网格数或x, y, z方向网格数
        bounds : array(6,) or Bounds
            网格范围，为array(6,)[xmin, xmax, ymin, ymax, zmin, zmax]或Bounds实例，为None的位置会使用默认值

        Returns
        -------
        ret
            构建的网格

        Notes
        -----
            原点是场景的边界的最小值。
            若指定cell_size，则边界点保证大于场景边界；
            若指定n_cells，则边界点是场景的边界的最大值。

            若n_cells为单个值，则用于指定总网格数，保证生成的网格数小于等于该值。
        """
        actual_bounds: Bounds = self.bounds
        if bounds is not None:
            bounds = np.asarray(bounds)
            actual_bounds.bounds[bounds != None] = bounds[bounds != None]

        bounds = actual_bounds
        sizes = bounds[1::2] - bounds[::2]

        if cell_size is not None:
            if not isinstance(cell_size, Iterable):
                cell_size = [cell_size] * 3
            cell_size = np.asarray(cell_size)
            n_cells = np.ceil(sizes / cell_size).astype(int)
        else:
            if not isinstance(n_cells, Iterable):
                avg_grids = (n_cells / np.prod(sizes)) ** (1 / 3)
                n_cells = (avg_grids * sizes).astype(int)
                cell_size = [1 / avg_grids] * 3
            else:
                n_cells = np.asarray(n_cells)
                cell_size = sizes / n_cells

        return TensorMesh([[(d, n)] for d, n in zip(cell_size, n_cells)], origin=bounds[::2])

    def build(self, cell_size=None, n_cells=None, bounds=None,
              executor=None, progress=False,
              cache=None, cache_dir=None,
              ) -> ModelledMesh:
        """根据给定的网格尺寸，构建场景的网格和模型，是create_mesh和build_model的组合

        Parameters
        ----------
        cell_size : number or array(3,)
            网格长宽高相等或分别定义网格x, y, z方向宽度大小
        n_cells : number or array(3,)
            总网格数或x, y, z方向网格数
        bounds : array(6,) or Bounds
            网格范围，为array(6,)[xmin, xmax, ymin, ymax, zmin, zmax]或Bounds实例，为None的位置会使用默认值
        executor
            并行执行器
        progress
            是否显示进度条
        cache
            指示缓存行为，详见Scene.build_model。单独使用cache=True来使用默认缓存路径或cache='...'来指定缓存文件路径
        cache_dir
            指示缓存行为，详见Scene.build_model。单独使用cache_dir='...'来指定缓存所在的文件夹路径

        Returns
        -------
        ret
            (mesh, model)，网格输出同create_mesh，模型输出同build_model

        See Also
        --------
            Scene.create_mesh : 构造网格
            Scene.build_model : 构造模型
        """
        mesh = self.create_mesh(cell_size=cell_size, n_cells=n_cells, bounds=bounds)
        pruned = self.build_model(mesh, executor=executor, progress=progress, cache=cache, cache_dir=cache_dir)

        return pruned

    @property
    def layers(self) -> Iterable[Layer]:
        yield self.objects_layer
        yield self.backgrounds_layer

    @property
    def objects(self):
        for layer in self.layers:
            for obj in layer:
                yield obj

    @property
    def shapes(self) -> Iterable[Shape3D]:
        for obj in self.objects:
            yield obj.shape

    def __len__(self):
        return sum((len(layer) for layer in self.layers))

    def __iter__(self) -> Iterable[Object]:
        for obj in self.objects:
            yield obj

    def __getitem__(self, item) -> Object:
        if item < 0:
            item = len(self) + item

        for layer in self.layers:
            size = len(layer)
            if item < size:
                return layer[item]
            else:
                item -= size

    def to_multiblock(self):
        import pyvista as pv

        ret = pv.MultiBlock()
        for shape in self.shapes:
            poly = shape.to_polydata()
            if poly is not None:
                ret.append(poly)

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

    def __dhash__(self):
        return dhash(*self.layers)

    @staticmethod
    def get_inactive_value_for_type(dtype):
        if dtype == bool:
            return Scene.INACTIVE_BOOL
        elif dtype == np.issubdtype(dtype, np.integer):
            return Scene.INACTIVE_INT
        else:
            return Scene.INACTIVE_FLOAT

    @staticmethod
    def is_active(model):
        # TODO: 考察是否有必要视0为非活动网格
        return model if model.dtype == bool else model != 0

    @staticmethod
    def compute_active(models: Iterable[np.ndarray]):
        ind_active = None
        for model in models:
            if ind_active is None:
                ind_active = Scene.is_active(model)
            else:
                ind_active |= Scene.is_active(model)

        return ind_active

    @staticmethod
    def _merge_models(models, new_models: Iterable[tuple[str, np.ndarray]], mask, mixer):
        """将new_models合并到models中，inplace

        Parameters
        ----------
        models
            已有的model
        new_models
            将要合并的model，以迭代器形式提供
        mask
            新model的有效格mask，如果指定为None，则会为每个待合并model单独计算mask
        mixer
            混合器

        Notes
        -----
            会直接修改models以及其内的数组
        """
        for key, current_layer in new_models:
            if mask is not None:
                current_mask = mask
            else:
                current_mask = Scene.is_active(current_layer)

            if key not in models:
                models[key] = current_layer
            else:
                prev_layer = models[key]
                filled_ind = Scene.is_active(prev_layer)
                overlapping_mask = filled_ind & current_mask
                non_overlapping_mask = current_mask ^ overlapping_mask

                prev_layer[overlapping_mask] = mixer(
                    prev_layer[overlapping_mask],
                    current_layer[overlapping_mask]
                )
                prev_layer[non_overlapping_mask] = current_layer[non_overlapping_mask]

    @staticmethod
    def _build_layer(layer: Layer, mesh, worker_id, progress=None):
        objects = layer.objects
        models = {}

        for obj in objects:
            shape = obj.shape

            # place的结果应该为布尔数组或范围为[0, 1]的数组，指示对应网格位置是否有效或有效程度
            ind: np.ndarray = shape.place(mesh, worker_id)
            mask = Scene.is_active(ind)

            def model_generator():
                for key, current_value in obj.items():
                    if ind.dtype == bool:
                        current_layer = np.zeros_like(ind, dtype=type(current_value))
                        current_layer[ind] = current_value
                    else:
                        current_layer = ind * current_value
                    yield key, current_layer

            Scene._merge_models(models, model_generator(), mask, obj.mixer)

            if progress is not None:
                progress.update(1)

        return models

    @staticmethod
    def _build_mesh_worker(layers: list[Layer], mesh, show_modeling_progress, worker_id):
        models = {}

        if show_modeling_progress:
            progress = tqdm.tqdm(total=sum((len(layer) for layer in layers)),
                                 position=0, leave=False, ncols=80)
        else:
            progress = None

        for layer in layers:
            layer_models = Scene._build_layer(layer, mesh, worker_id, progress)
            Scene._merge_models(models, layer_models.items(), None, layer.mixer)

        if progress is not None:
            progress.close()

        return models

    def _determine_cache_filepath(self, mesh, cache, cache_dir):
        cache_filepath = None
        if isinstance(cache, bool) and cache:
            cache_filename = self._generate_model_filename(mesh)
            if cache_dir is not None:
                cache_filepath = os.path.join(cache_dir, cache_filename)
            else:
                ensure_dir(Scene.default_cache_dir)
                cache_filepath = os.path.join(os.path.abspath(Scene.default_cache_dir), cache_filename)
        elif isinstance(cache, str):
            cache_filepath = cache
        elif cache is None:
            if cache_dir is not None:
                cache_filename = self._generate_model_filename(mesh)
                cache_filepath = os.path.join(cache_dir, cache_filename)

        return cache_filepath

    def _generate_model_filename(self, mesh: TensorMesh):
        mesh_hash = dhash_discretize_mesh(mesh).hexdigest(digits=6)
        model_hash = dhash(self).hexdigest(digits=6)

        origin = mesh.origin.astype(float)
        lengths = np.asarray([h.sum() for h in mesh.h])

        components = []

        for axis, x0, length in zip('xyz', origin, lengths):
            components.append(f'{axis}({x0:.2}~{x0 + length:.2})')

        avg_node_size = lengths / [h.shape[0] for h in mesh.h]
        if np.all(avg_node_size - avg_node_size[0] < 1e-7):
            avg_node_size = np.asarray(avg_node_size[0])
            components.append(f'avg_gs({avg_node_size:.2})')
        else:
            components.append(f'avg_gs{(*np.around(avg_node_size, decimals=2),)}')

        components.append(f'mesh({mesh_hash})')
        components.append(f'model({model_hash})')

        return '#'.join(components) + '.pkl'
