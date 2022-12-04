import os

import numpy as np
from discretize.utils import mkvc

from metalpy.mexin import Mixin
from metalpy.scab.progressed import Progress

from .tied_simulation3d_integral import TaichiSimulation3DIntegral, Receiver


class TiedSimulation3DIntegralMixin(Mixin):
    def __init__(self, this):
        super().__init__(this)

    def linear_operator(this, self):
        from SimPEG.potential_fields.magnetics.simulation import Simulation3DIntegral
        self: Simulation3DIntegral
        n_cells = self.nC

        if getattr(self, "model_type", None) == "vector":
            n_cells *= 3
        if self.store_sensitivities == "disk":
            sens_name = os.path.join(self.sensitivity_path, "sensitivity.npy")
            if os.path.exists(sens_name):
                # do not pull array completely into ram, just need to check the size
                kernel = np.load(sens_name, mmap_mode="r")
                if kernel.shape == (self.survey.nD, n_cells):
                    print(f"Found sensitivity file at {sens_name} with expected shape")
                    kernel = np.asarray(kernel)
                    return kernel

        forward_only = self.store_sensitivities == "forward_only"
        if forward_only:
            model = self.chi  # chi = mapping * model
        else:
            model = None

        # 计算网格在三个方向的边界位置
        cell_centers = self.mesh.cell_centers[self.ind_active]
        h_gridded = self.mesh.h_gridded[self.ind_active]
        bsw = cell_centers - h_gridded / 2.0
        tne = cell_centers + h_gridded / 2.0

        xn1, xn2 = bsw[:, 0], tne[:, 0]
        yn1, yn2 = bsw[:, 1], tne[:, 1]
        zn1, zn2 = bsw[:, 2], tne[:, 2]

        xn = np.c_[mkvc(xn1), mkvc(xn2)]
        yn = np.c_[mkvc(yn1), mkvc(yn2)]
        zn = np.c_[mkvc(zn1), mkvc(zn2)]

        base_cell_sizes = np.r_[
            self.mesh.h[0].min(),
            self.mesh.h[1].min(),
            self.mesh.h[2].min(),
        ]

        magnetization = self.M  # (nC, 3)的矩阵或None
        if magnetization is None:
            # 输入标量，TaichiSimulation3DIntegral内部会自动转换为(nC, 3)的矩阵
            magnetization = self.survey.source_field.amplitude

        if self.model_type == 'scalar':
            model_type = TaichiSimulation3DIntegral.MType_Scalar
        else:
            model_type = TaichiSimulation3DIntegral.MType_Vector
            if forward_only:
                model = model.reshape(3, -1).T

        receivers = [Receiver(rx.locations, rx.components) for rx in self.survey.source_field.receiver_list]

        progress = self.mixins.get(Progress)
        if progress is not None:
            progress.set_manual_update(True)
            progress = progress.progressbar

        kernel = TaichiSimulation3DIntegral(
            receivers=receivers,
            xn=xn, yn=yn, zn=zn,
            base_cell_sizes=base_cell_sizes,
            model_type=model_type,
            magnetization=magnetization,
            row_stype=TaichiSimulation3DIntegral.Layout_SoA,
            col_stype=TaichiSimulation3DIntegral.Layout_AoS,
            tmi_projection=self.tmi_projection,
        ).dpred(model, progress=progress)

        if self.store_sensitivities == "forward_only":
            kernel = kernel.ravel()

        if self.store_sensitivities == "disk":
            print(f"writing sensitivity to {sens_name}")
            os.makedirs(self.sensitivity_path, exist_ok=True)
            np.save(sens_name, kernel)
        return kernel
