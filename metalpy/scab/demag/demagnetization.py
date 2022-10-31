import numpy as np
import pyamg
import scipy.sparse as sp
from SimPEG import maps
from SimPEG.potential_fields import magnetics
from discretize.base import BaseTensorMesh

from ..utils.misc import Field


class Demagnetization:
    def __init__(self, mesh: BaseTensorMesh, source_field: Field, active_ind=None):
        """
        听过BiCGSTAB求解计算退磁作用下的磁化强度

        Parameters
        ----------
        mesh
            模型网格

        source_field
            外部场源

        active_ind
            有效网格下标或掩码
        """
        super().__init__()

        self.mesh = mesh
        self.source_field = source_field

        receiver_points = mesh.cell_centers
        if active_ind is not None:
            receiver_points = receiver_points[active_ind]

        nC = len(receiver_points)
        components = ('bx', 'by', 'bz')
        receiver_list = magnetics.receivers.Point(receiver_points, components=components)
        receiver_list = [receiver_list]

        inducing_field = magnetics.sources.SourceField(
            receiver_list=receiver_list, parameters=source_field.with_strength(1)  # 让磁化矩阵M为1，因为它计算的是TM
        )
        survey = magnetics.survey.Survey(inducing_field)
        model_map = maps.IdentityMap(nP=3 * nC)
        self.sim = magnetics.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            model_type="vector",
            chiMap=model_map,
            actInd=active_ind,
            store_sensitivities="ram",
        )

    def dpred(self, model):
        """
        Parameters
        ----------
        model: array-like(nC,)
            磁化率模型

        Returns
        -------
            array(nC, 3)，三轴磁化率矩阵
        """
        nC = self.sim.nC
        I = sp.identity(3 * nC)
        H0 = self.source_field.unit_vector
        H0 = np.repeat(H0, nC).ravel()

        T = self.sim.G
        del self.sim._G
        T = T.reshape(nC, 3, -1).swapaxes(0, 1).reshape(3 * nC, -1)
        X = np.repeat(model[None, :], 3, axis=0).ravel()
        X = sp.diags(X)

        A = I - X @ T
        b = X @ H0

        m, info = pyamg.krylov.bicgstab(A, b)

        # assert abs(X @ (H0 + T @ m) - m).mean() < 1e-3, 'fucked up'
        return m.reshape(3, -1).T
