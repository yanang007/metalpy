from functools import reduce

import numpy as np
import tqdm


class QuickUnion:
    def __init__(self, n):
        """快速并查集
        """
        # TODO: 考察是否有必要使用int64作为索引类型
        self.unions = np.arange(n)
        self.array_buf = np.zeros(n, dtype=np.int32)

    def connect(self, a, b):
        rdst, da = self.find_root(a)
        rsrc, db = self.find_root(b)
        if da < db:
            rdst, rsrc = rsrc, rdst
        self.unions[rsrc] = rdst

    def find_root(self, a):
        dist = 0
        root = self.unions[a]
        while root != a:
            self.array_buf[dist] = a
            dist += 1
            a = root
            root = self.unions[a]

        self.unions[self.array_buf[:dist]] = root  # 坍缩树优化
        return root, dist

    def collapse(self):
        for i in tqdm.trange(len(self.unions)):
            self.find_root(i)


class ConnectedTriangleSurfaces:
    def __init__(self):
        """用于依据联通性（是否有相邻边）将多个三角面的顶点分组
        """
        self.point_groups: list[set[int]] = []

    def add(self, pts):
        vs = set(pts)
        diff = None

        candidates = []

        # 考虑到面在各个局部体内的连续性，一般而言新的面会邻接最后一个点集或者属于新的点集合
        for group in reversed(self.point_groups):
            diff = vs.difference(group)
            if len(diff) <= 1:
                candidates.append(group)

                if len(candidates) > 1:
                    break  # 因为针对三角面，所以一定只能同时邻接两个组

        n_adjacent = len(candidates)

        if n_adjacent == 0:
            target_group = set(pts)
            self.point_groups.append(target_group)
        elif n_adjacent == 1:
            if len(diff) == 1:
                candidates[0].add(diff.pop())
        else:
            for g in candidates[1:]:
                self.point_groups.remove(g)

            reduce(lambda x, y: x.update(y), candidates)

    def get_groups(self):
        return self.point_groups
