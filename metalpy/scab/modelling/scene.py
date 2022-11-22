from typing import Any, Union

import numpy as np
import tqdm

from metalpy.mepa import LinearExecutor
from metalpy.scab.modelling.object import Object
from metalpy.scab.modelling.shapes import Shape3D


class Scene:
    def __init__(self):
        self.objects = []

    def append(self, shape: Shape3D, values: Union[dict[str, Any], Any]):
        """添加三维几何体

        Parameters
        ----------
        shape
            三维几何体

        values
            三维几何体的参数
        """
        self.objects.append(Object(shape, values))

    @staticmethod
    def build_mesh_worker(objects: list[Object], mesh, show_modeling_progress, worker_id):
        values = {}
        for object in tqdm.tqdm(objects, position=0, leave=False, ncols=80) if show_modeling_progress else objects:
            shape = object.shape

            # place的结果应该为布尔数组或范围为[0, 1]的数组，指示对应网格位置是否有效或有效程度
            ind = shape.place(mesh, worker_id)
            current_mask = ind if ind.dtype == bool else ind != 0  # TODO: 考察是否有必要视0为非活动网格

            for key, current_value in object.items():
                if ind.dtype == bool:
                    current_layer = np.zeros_like(ind)
                    current_layer[ind] = current_value
                else:
                    current_layer = ind * current_value

                if key not in values:
                    values[key] = current_layer
                else:
                    prev_layer = values[key]
                    filled_ind = prev_layer != 0  # TODO: 同上
                    overlapping_mask = filled_ind & current_mask
                    non_overlapping_mask = current_mask ^ overlapping_mask

                    prev_layer[overlapping_mask] = object.mix(
                        prev_layer[overlapping_mask],
                        current_layer[overlapping_mask]
                    )
                    prev_layer[non_overlapping_mask] = current_layer[non_overlapping_mask]

        return values

    def build(self, mesh,
              executor=None,
              progress=False):
        """构建模型网格

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

        values_dict = {}
        for key in futures[0].result().keys():
            values_dict[key] = np.concatenate([future.result()[key] for future in futures])

        if len(values_dict) == 1 and Object.DEFAULT_KEY in values_dict:
            return values_dict[Object.DEFAULT_KEY]
        else:
            return values_dict

    def __iter__(self):
        for obj in self.objects:
            yield obj
