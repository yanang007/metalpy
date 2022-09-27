import os

import numpy as np
import torch
import SimPEG.potential_fields.magnetics

import psutil

from metalpy.scab.injectors import replaces, extends
import metalpy.scab.progressed
from metalpy.scab.utils.patch import use_patch
from metalpy.scab.utils.sparse import scipy2torch

use_patch(metalpy.scab.progressed)

PROFILE_MEMORY = False


@extends(SimPEG.potential_fields.magnetics.Simulation3DIntegral, 'torch_on_impl')
def __magnetics_Simulation3DIntegral_ext_torch_on_impl(self):
    """
    由__BaseSimulation_ext_progress_on调用
    """
    replaces(self, 'linear_operator')(__magnetics_Simulation3DIntegral_ext_linear_operator)

    if PROFILE_MEMORY:
        global __magnetics_Simulation3DIntegral_ext_evaluate_integral
        from memory_profiler import profile
        __magnetics_Simulation3DIntegral_ext_evaluate_integral = profile(__magnetics_Simulation3DIntegral_ext_evaluate_integral)

    replaces(self, 'evaluate_integral')(__magnetics_Simulation3DIntegral_ext_evaluate_integral)

    extends(self, 'estimate_memory_cost')(__magnetics_Simulation3DIntegral_ext_estimate_memory_cost)


def __magnetics_Simulation3DIntegral_ext_estimate_memory_cost(self, size=None):
    size = self.survey.receiver_locations.shape[0] if size is None else size
    return 309.1 * 1024 * 1024 / 10 / 50250 * size * self.modelMap.shape[0]


def __magnetics_Simulation3DIntegral_ext_linear_operator(self, orig_fn):
    # evaluate_integral暂时没有转换回来，因此这个方法调用时会调用错误的evaluate_integralfan方法
    # TODO: 修复他
    # if not hasattr(self, 'torch_device'):
    #     return orig_fn()

    def create_tensor(x, dtype=None):
        # return torch.tensor(x, dtype=dtype, device=self.torch_device)
        if dtype is None:
            return torch.from_numpy(x).to(self.torch_device)
        else:
            return torch.as_tensor(x, dtype=dtype, device=self.torch_device)

    # infos = [torch.cuda.get_device_properties(i) for i in range(ng)]
    # 获取显存大小来实施计算分块

    self.nC = self.modelMap.shape[0]  # 网格数
    nD = self.survey.nD  # 数据量，观测点数 * 目标量数
    nObs = self.survey.receiver_locations.shape[0]  # self.nD  # 观测点数

    if self.store_sensitivities == "disk":
        sens_name = self.sensitivity_path + "sensitivity.npy"
        if os.path.exists(sens_name):
            # do not pull array completely into ram, just need to check the size
            kernel = np.load(sens_name, mmap_mode="r")
            if kernel.shape == (nD, self.nC):
                print(f"Found sensitivity file at {sens_name} with expected shape")
                kernel = np.asarray(kernel)
                return kernel
    # Single threaded
    mem = psutil.virtual_memory().free
    mem -= 100 * 1024 * 1024  # 100MB
    chunk_size = self.estimate_batch_size(mem)
    n_chunks = int(np.ceil(nObs / chunk_size))

    receiver_lists = np.array_split(self.survey.receiver_locations, n_chunks)
    active_components = {
        k: np.array_split(v, n_chunks) for k, v in self.survey.components.items()
    }

    kernels = []
    for i, receivers in enumerate(receiver_lists):
        components = {k: v[i] for k, v in active_components.items()}
        kernel = self.evaluate_integral(create_tensor(receivers),
                                        {k: create_tensor(v) for k, v in
                                        components.items()}, )
        kernels.append(kernel)

        self.update_progress(len(receivers))

    kernel = torch.vstack(kernels)
    model = create_tensor(self.model)

    if self.store_sensitivities == "forward_only":
        kernel = torch.einsum('ocg,g->oc', kernel, model)

    if self.store_sensitivities == "disk":
        print(f"writing sensitivity to {sens_name}")
        os.makedirs(self.sensitivity_path, exist_ok=True)
        np.save(sens_name, kernel)

    return kernel.detach().numpy()


def __magnetics_Simulation3DIntegral_ext_evaluate_integral(self, receiver_location, components, orig_fn=None):
    """
    Load in the active nodes of a tensor mesh and computes the magnetic
    forward relation between a cuboid and a given observation
    location outside the Earth [obsx, obsy, obsz]

    INPUT:
    receiver_location:  [obsx, obsy, obsz] nC x 3 Array

    components: list[str]
        List of magnetic components chosen from:
        'bx', 'by', 'bz', 'bxx', 'bxy', 'bxz', 'byy', 'byz', 'bzz'

    OUTPUT:
    Tx = [Txx Txy Txz]
    Ty = [Tyx Tyy Tyz]
    Tz = [Tzx Tzy Tzz]
    """

    def create_tensor(x, dtype=None):
        # return torch.tensor(x, dtype=dtype, device=self.torch_device)
        if dtype is None:
            return torch.from_numpy(x).to(self.torch_device)
        else:
            return torch.as_tensor(x, dtype=dtype, device=self.torch_device)

    def create_zeros(shape, dtype=None):
        return torch.zeros(shape, dtype=torch.float64 if dtype is None else dtype, device=self.torch_device)

    # TODO: This should probably be converted to C
    tol1 = 1e-10  # Tolerance 1 for numerical stability over nodes and edges
    tol2 = 1e-4  # Tolerance 2 for numerical stability over nodes and edges

    receiver_location.requires_grad = False
    # number of cells in mesh
    nC = self.Xn.shape[0]
    batch_size = receiver_location.shape[0]

    # TODO: 用component mask控制具体点上要计算的分量
    rows = {}

    # base cell dimensions
    min_hx, min_hy, min_hz = (
        self.mesh.hx.min(),
        self.mesh.hy.min(),
        self.mesh.hz.min(),
    )

    # comp. pos. differences for tne, bsw nodes. Adjust if location within
    # tolerance of a node or edge
    # 数据打包，如果没有性能问题就算了
    # boundaries = np.c_[self.Xn, self.Yn, self.Zn]
    # boundaries = create_tensor(boundaries)
    #
    # d2 = boundaries[:, ::2]

    # self.M是CSR，转置后变成CSC
    # 使用torch的CSR/CSC格式存在以下问题
    # 1. CSR计算结果有问题（核心问题）
    # 2. CSR只能左乘（不过COO也只能左乘）
    # 3. CSR的操作无法求导
    # TODO: 但是CSR理论计算效率高于COO，如果能解决上述问题，那么这里可以考虑不用转换为COO
    M = scipy2torch(self.M.T, device=self.torch_device).to_sparse_coo()

    def mul_by_m(vd, key):
        # truth = torch.as_tensor(vd[key].detach().numpy() * self.M)
        mult = M @ vd[key].T
        mult = mult.T
        vd[key] = mult

    Zn = create_tensor(self.Zn)
    Zs = receiver_location[:, 2][:, None, None].expand(-1, nC, -1)
    dz = Zn - Zs
    del Zn, Zs
    dz[torch.abs(dz) < tol2 * min_hz] = tol2 * min_hz
    dzdz = dz ** 2
    dz1, dz2 = dz[:, :, 0], dz[:, :, 1]
    dz1dz1, dz2dz2 = dzdz[:, :, 0], dzdz[:, :, 1]

    Yn = create_tensor(self.Yn)
    Ys = receiver_location[:, 1][:, None, None].expand(-1, nC, -1)
    dy = Yn - Ys
    del Yn, Ys
    dy[torch.abs(dy) < tol2 * min_hy] = tol2 * min_hy
    dydy = dy ** 2
    dy1, dy2 = dy[:, :, 0], dy[:, :, 1]
    dy1dy1, dy2dy2 = dydy[:, :, 0], dydy[:, :, 1]

    Xn = create_tensor(self.Xn)
    Xs = receiver_location[:, 0][:, None, None].expand(-1, nC, -1)
    dx = Xn - Xs
    del Xn, Xs
    dx[torch.abs(dx) < tol2 * min_hx] = tol2 * min_hx
    dxdx = dx ** 2
    dx1, dx2 = dx[:, :, 0], dx[:, :, 1]
    dx1dx1, dx2dx2 = dxdx[:, :, 0], dxdx[:, :, 1]

    # 2D radius component squared of corner nodes
    R1 = dy2dy2 + dx2dx2
    R2 = dy2dy2 + dx1dx1
    R3 = dy1dy1 + dx2dx2
    R4 = dy1dy1 + dx1dx1

    # radius to each cell node
    r1 = torch.sqrt(dz2dz2 + R2)
    r2 = torch.sqrt(dz2dz2 + R1)
    r3 = torch.sqrt(dz1dz1 + R1)
    r4 = torch.sqrt(dz1dz1 + R2)
    r5 = torch.sqrt(dz2dz2 + R3)
    r6 = torch.sqrt(dz2dz2 + R4)
    r7 = torch.sqrt(dz1dz1 + R4)
    r8 = torch.sqrt(dz1dz1 + R3)

    # compactify argument calculations
    arg1_ = dx1 + dy2 + r1
    arg1 = dy2 + dz2 + r1
    arg2 = dx1 + dz2 + r1
    arg3 = dx1 + r1
    arg4 = dy2 + r1
    arg5 = dz2 + r1

    arg6_ = dx2 + dy2 + r2
    arg6 = dy2 + dz2 + r2
    arg7 = dx2 + dz2 + r2
    arg8 = dx2 + r2
    arg9 = dy2 + r2
    arg10 = dz2 + r2

    arg11_ = dx2 + dy2 + r3
    arg11 = dy2 + dz1 + r3
    arg12 = dx2 + dz1 + r3
    arg13 = dx2 + r3
    arg14 = dy2 + r3
    arg15 = dz1 + r3

    arg16_ = dx1 + dy2 + r4
    arg16 = dy2 + dz1 + r4
    arg17 = dx1 + dz1 + r4
    arg18 = dx1 + r4
    arg19 = dy2 + r4
    arg20 = dz1 + r4

    arg21_ = dx2 + dy1 + r5
    arg21 = dy1 + dz2 + r5
    arg22 = dx2 + dz2 + r5
    arg23 = dx2 + r5
    arg24 = dy1 + r5
    arg25 = dz2 + r5

    arg26_ = dx1 + dy1 + r6
    arg26 = dy1 + dz2 + r6
    arg27 = dx1 + dz2 + r6
    arg28 = dx1 + r6
    arg29 = dy1 + r6
    arg30 = dz2 + r6

    arg31_ = dx1 + dy1 + r7
    arg31 = dy1 + dz1 + r7
    arg32 = dx1 + dz1 + r7
    arg33 = dx1 + r7
    arg34 = dy1 + r7
    arg35 = dz1 + r7

    arg36_ = dx2 + dy1 + r8
    arg36 = dy1 + dz1 + r8
    arg37 = dx2 + dz1 + r8
    arg38 = dx2 + r8
    arg39 = dy1 + r8
    arg40 = dz1 + r8

    if ("bxx" in components) or ("bzz" in components):
        rows["bxx"] = create_zeros((batch_size, 3 * nC))

        rows["bxx"][:, 0:nC] = 2 * (
                ((dx1 ** 2 - r1 * arg1) / (r1 * arg1 ** 2 + dx1 ** 2 * r1))
                - ((dx2 ** 2 - r2 * arg6) / (r2 * arg6 ** 2 + dx2 ** 2 * r2))
                + ((dx2 ** 2 - r3 * arg11) / (r3 * arg11 ** 2 + dx2 ** 2 * r3))
                - ((dx1 ** 2 - r4 * arg16) / (r4 * arg16 ** 2 + dx1 ** 2 * r4))
                + ((dx2 ** 2 - r5 * arg21) / (r5 * arg21 ** 2 + dx2 ** 2 * r5))
                - ((dx1 ** 2 - r6 * arg26) / (r6 * arg26 ** 2 + dx1 ** 2 * r6))
                + ((dx1 ** 2 - r7 * arg31) / (r7 * arg31 ** 2 + dx1 ** 2 * r7))
                - ((dx2 ** 2 - r8 * arg36) / (r8 * arg36 ** 2 + dx2 ** 2 * r8))
        )

        rows["bxx"][:, nC: 2 * nC] = (
                dx2 / (r5 * arg25)
                - dx2 / (r2 * arg10)
                + dx2 / (r3 * arg15)
                - dx2 / (r8 * arg40)
                + dx1 / (r1 * arg5)
                - dx1 / (r6 * arg30)
                + dx1 / (r7 * arg35)
                - dx1 / (r4 * arg20)
        )

        rows["bxx"][:, 2 * nC:] = (
                dx1 / (r1 * arg4)
                - dx2 / (r2 * arg9)
                + dx2 / (r3 * arg14)
                - dx1 / (r4 * arg19)
                + dx2 / (r5 * arg24)
                - dx1 / (r6 * arg29)
                + dx1 / (r7 * arg34)
                - dx2 / (r8 * arg39)
        )

        rows["bxx"] /= 4 * torch.pi
        mul_by_m(rows, "bxx")

    if ("byy" in components) or ("bzz" in components):
        rows["byy"] = create_zeros((batch_size, 3 * nC))

        rows["byy"][:, 0:nC] = (
                dy2 / (r3 * arg15)
                - dy2 / (r2 * arg10)
                + dy1 / (r5 * arg25)
                - dy1 / (r8 * arg40)
                + dy2 / (r1 * arg5)
                - dy2 / (r4 * arg20)
                + dy1 / (r7 * arg35)
                - dy1 / (r6 * arg30)
        )
        rows["byy"][:, nC: 2 * nC] = 2 * (
                ((dy2 ** 2 - r1 * arg2) / (r1 * arg2 ** 2 + dy2 ** 2 * r1))
                - ((dy2 ** 2 - r2 * arg7) / (r2 * arg7 ** 2 + dy2 ** 2 * r2))
                + ((dy2 ** 2 - r3 * arg12) / (r3 * arg12 ** 2 + dy2 ** 2 * r3))
                - ((dy2 ** 2 - r4 * arg17) / (r4 * arg17 ** 2 + dy2 ** 2 * r4))
                + ((dy1 ** 2 - r5 * arg22) / (r5 * arg22 ** 2 + dy1 ** 2 * r5))
                - ((dy1 ** 2 - r6 * arg27) / (r6 * arg27 ** 2 + dy1 ** 2 * r6))
                + ((dy1 ** 2 - r7 * arg32) / (r7 * arg32 ** 2 + dy1 ** 2 * r7))
                - ((dy1 ** 2 - r8 * arg37) / (r8 * arg37 ** 2 + dy1 ** 2 * r8))
        )
        rows["byy"][:, 2 * nC:] = (
                dy2 / (r1 * arg3)
                - dy2 / (r2 * arg8)
                + dy2 / (r3 * arg13)
                - dy2 / (r4 * arg18)
                + dy1 / (r5 * arg23)
                - dy1 / (r6 * arg28)
                + dy1 / (r7 * arg33)
                - dy1 / (r8 * arg38)
        )

        rows["byy"] /= 4 * torch.pi
        mul_by_m(rows, "byy")

    if "bzz" in components:
        rows["bzz"] = -rows["bxx"] - rows["byy"]

    if "bxy" in components:
        rows["bxy"] = create_zeros((batch_size, 3 * nC))

        rows["bxy"][:, 0:nC] = 2 * (
                ((dx1 * arg4) / (r1 * arg1 ** 2 + (dx1 ** 2) * r1))
                - ((dx2 * arg9) / (r2 * arg6 ** 2 + (dx2 ** 2) * r2))
                + ((dx2 * arg14) / (r3 * arg11 ** 2 + (dx2 ** 2) * r3))
                - ((dx1 * arg19) / (r4 * arg16 ** 2 + (dx1 ** 2) * r4))
                + ((dx2 * arg24) / (r5 * arg21 ** 2 + (dx2 ** 2) * r5))
                - ((dx1 * arg29) / (r6 * arg26 ** 2 + (dx1 ** 2) * r6))
                + ((dx1 * arg34) / (r7 * arg31 ** 2 + (dx1 ** 2) * r7))
                - ((dx2 * arg39) / (r8 * arg36 ** 2 + (dx2 ** 2) * r8))
        )
        rows["bxy"][:, nC: 2 * nC] = (
                dy2 / (r1 * arg5)
                - dy2 / (r2 * arg10)
                + dy2 / (r3 * arg15)
                - dy2 / (r4 * arg20)
                + dy1 / (r5 * arg25)
                - dy1 / (r6 * arg30)
                + dy1 / (r7 * arg35)
                - dy1 / (r8 * arg40)
        )
        rows["bxy"][:, 2 * nC:] = (
                1 / r1 - 1 / r2 + 1 / r3 - 1 / r4 + 1 / r5 - 1 / r6 + 1 / r7 - 1 / r8
        )

        rows["bxy"] /= 4 * torch.pi
        mul_by_m(rows, "bxy")

    if "bxz" in components:
        rows["bxz"] = create_zeros((batch_size, 3 * nC))

        rows["bxz"][:, 0:nC] = 2 * (
                ((dx1 * arg5) / (r1 * (arg1 ** 2) + (dx1 ** 2) * r1))
                - ((dx2 * arg10) / (r2 * (arg6 ** 2) + (dx2 ** 2) * r2))
                + ((dx2 * arg15) / (r3 * (arg11 ** 2) + (dx2 ** 2) * r3))
                - ((dx1 * arg20) / (r4 * (arg16 ** 2) + (dx1 ** 2) * r4))
                + ((dx2 * arg25) / (r5 * (arg21 ** 2) + (dx2 ** 2) * r5))
                - ((dx1 * arg30) / (r6 * (arg26 ** 2) + (dx1 ** 2) * r6))
                + ((dx1 * arg35) / (r7 * (arg31 ** 2) + (dx1 ** 2) * r7))
                - ((dx2 * arg40) / (r8 * (arg36 ** 2) + (dx2 ** 2) * r8))
        )
        rows["bxz"][:, nC: 2 * nC] = (
                1 / r1 - 1 / r2 + 1 / r3 - 1 / r4 + 1 / r5 - 1 / r6 + 1 / r7 - 1 / r8
        )
        rows["bxz"][:, 2 * nC:] = (
                dz2 / (r1 * arg4)
                - dz2 / (r2 * arg9)
                + dz1 / (r3 * arg14)
                - dz1 / (r4 * arg19)
                + dz2 / (r5 * arg24)
                - dz2 / (r6 * arg29)
                + dz1 / (r7 * arg34)
                - dz1 / (r8 * arg39)
        )

        rows["bxz"] /= 4 * torch.pi
        mul_by_m(rows, "bxz")

    if "byz" in components:
        rows["byz"] = create_zeros((batch_size, 3 * nC))

        rows["byz"][:, 0:nC] = (
                1 / r3 - 1 / r2 + 1 / r5 - 1 / r8 + 1 / r1 - 1 / r4 + 1 / r7 - 1 / r6
        )
        rows["byz"][:, nC: 2 * nC] = 2 * (
                (dy2 * arg5 / r1 * (arg2 ** 2) + (dy2 ** 2) * r1)
                - ((dy2 * arg10 / r2 * (arg7 ** 2) + (dy2 ** 2) * r2))
                + ((dy2 * arg15 / r3 * (arg12 ** 2) + (dy2 ** 2) * r3))
                - ((dy2 * arg20 / r4 * (arg17 ** 2) + (dy2 ** 2) * r4))
                + ((dy1 * arg25 / r5 * (arg22 ** 2) + (dy1 ** 2) * r5))
                - ((dy1 * arg30 / r6 * (arg27 ** 2) + (dy1 ** 2) * r6))
                + ((dy1 * arg35 / r7 * (arg32 ** 2) + (dy1 ** 2) * r7))
                - ((dy1 * arg40 / r8 * (arg37 ** 2) + (dy1 ** 2) * r8))
        )
        rows["byz"][:, 2 * nC:] = (
                dz2 / (r1 * arg3)
                - dz2 / (r2 * arg8)
                + dz1 / (r3 * arg13)
                - dz1 / (r4 * arg18)
                + dz2 / (r5 * arg23)
                - dz2 / (r6 * arg28)
                + dz1 / (r7 * arg33)
                - dz1 / (r8 * arg38)
        )

        rows["byz"] /= 4 * torch.pi
        mul_by_m(rows, "byz")

    if ("bx" in components) or ("tmi" in components):
        rows["bx"] = create_zeros((batch_size, 3 * nC))

        rows["bx"][:, 0:nC] = (
                (-2 * torch.arctan2(dx1, arg1 + tol1))
                - (-2 * torch.arctan2(dx2, arg6 + tol1))
                + (-2 * torch.arctan2(dx2, arg11 + tol1))
                - (-2 * torch.arctan2(dx1, arg16 + tol1))
                + (-2 * torch.arctan2(dx2, arg21 + tol1))
                - (-2 * torch.arctan2(dx1, arg26 + tol1))
                + (-2 * torch.arctan2(dx1, arg31 + tol1))
                - (-2 * torch.arctan2(dx2, arg36 + tol1))
        )
        rows["bx"][:, nC: 2 * nC] = (
                torch.log(arg5)
                - torch.log(arg10)
                + torch.log(arg15)
                - torch.log(arg20)
                + torch.log(arg25)
                - torch.log(arg30)
                + torch.log(arg35)
                - torch.log(arg40)
        )
        rows["bx"][:, 2 * nC:] = (
                (torch.log(arg4) - torch.log(arg9))
                + (torch.log(arg14) - torch.log(arg19))
                + (torch.log(arg24) - torch.log(arg29))
                + (torch.log(arg34) - torch.log(arg39))
        )
        rows["bx"] /= -4 * torch.pi
        mul_by_m(rows, "bx")

    if ("by" in components) or ("tmi" in components):
        rows["by"] = create_zeros((batch_size, 3 * nC))

        rows["by"][:, 0:nC] = (
                torch.log(arg5)
                - torch.log(arg10)
                + torch.log(arg15)
                - torch.log(arg20)
                + torch.log(arg25)
                - torch.log(arg30)
                + torch.log(arg35)
                - torch.log(arg40)
        )
        rows["by"][:, nC: 2 * nC] = (
                (-2 * torch.arctan2(dy2, arg2 + tol1))
                - (-2 * torch.arctan2(dy2, arg7 + tol1))
                + (-2 * torch.arctan2(dy2, arg12 + tol1))
                - (-2 * torch.arctan2(dy2, arg17 + tol1))
                + (-2 * torch.arctan2(dy1, arg22 + tol1))
                - (-2 * torch.arctan2(dy1, arg27 + tol1))
                + (-2 * torch.arctan2(dy1, arg32 + tol1))
                - (-2 * torch.arctan2(dy1, arg37 + tol1))
        )
        rows["by"][:, 2 * nC:] = (
                (torch.log(arg3) - torch.log(arg8))
                + (torch.log(arg13) - torch.log(arg18))
                + (torch.log(arg23) - torch.log(arg28))
                + (torch.log(arg33) - torch.log(arg38))
        )

        rows["by"] /= -4 * torch.pi
        mul_by_m(rows, "by")

    if ("bz" in components) or ("tmi" in components):
        rows["bz"] = create_zeros((batch_size, 3 * nC))

        rows["bz"][:, 0:nC] = (
                torch.log(arg4)
                - torch.log(arg9)
                + torch.log(arg14)
                - torch.log(arg19)
                + torch.log(arg24)
                - torch.log(arg29)
                + torch.log(arg34)
                - torch.log(arg39)
        )
        rows["bz"][:, nC: 2 * nC] = (
                (torch.log(arg3) - torch.log(arg8))
                + (torch.log(arg13) - torch.log(arg18))
                + (torch.log(arg23) - torch.log(arg28))
                + (torch.log(arg33) - torch.log(arg38))
        )
        rows["bz"][:, 2 * nC:] = (
                (-2 * torch.arctan2(dz2, arg1_ + tol1))
                - (-2 * torch.arctan2(dz2, arg6_ + tol1))
                + (-2 * torch.arctan2(dz1, arg11_ + tol1))
                - (-2 * torch.arctan2(dz1, arg16_ + tol1))
                + (-2 * torch.arctan2(dz2, arg21_ + tol1))
                - (-2 * torch.arctan2(dz2, arg26_ + tol1))
                + (-2 * torch.arctan2(dz1, arg31_ + tol1))
                - (-2 * torch.arctan2(dz1, arg36_ + tol1))
        )
        rows["bz"] /= -4 * torch.pi
        mul_by_m(rows, "bz")

    if "tmi" in components:
        # truth = orig_fn(receiver_location=receiver_location[0], components=list(components.keys()))
        mat = torch.concat([rows[c][:, :, None] for c in ['bx', 'by', 'bz']], dim=2)
        proj = create_tensor(self.tmi_projection)

        rows["tmi"] = torch.einsum('ij,bkj->bk', proj, mat)

    return torch.concat([rows[component][:, None, :] for component in components], dim=1)
