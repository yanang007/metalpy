import os


class ObjSplitter:
    def __init__(self, obj_path: str):
        self.obj_path = obj_path
        self.obj_lines = []
        self.groups_index = []
        self.vertices_counts = []
        self.texture_params_counts = []
        self.normals_counts = []
        self.faces_without_offset = {}

    def __obj2lines(self):
        if not os.path.exists(self.obj_path):
            print("obj file does not exist")
            return -1
        self.obj_lines = []
        with open(self.obj_path, "r", encoding="utf-8") as f:
            self.obj_lines = f.read().splitlines()

    def __get_groups_index(self):
        # g所在位置+结尾
        self.groups_index = [index for index, line in enumerate(self.obj_lines) if "g " in line]
        self.groups_index.append(len(self.obj_lines))
        print(f"{self.groups_index=}")

    def __count_groups_coord(self):
        # 各组v,vt,vn统计
        self.vertices_counts = []
        self.texture_params_counts = []
        self.normals_counts = []
        for g_start, g_end in zip(self.groups_index, self.groups_index[1:]):
            vertices_count = sum(1 for line in self.obj_lines[g_start:g_end] if "v " in line)
            texture_params_count = sum(1 for line in self.obj_lines[g_start:g_end] if "vt " in line)
            normals_count = sum(1 for line in self.obj_lines[g_start:g_end] if "vn " in line)
            self.vertices_counts.append(vertices_count)
            self.texture_params_counts.append(texture_params_count)
            self.normals_counts.append(normals_count)
        print(f"{self.vertices_counts=}")
        print(f"{self.texture_params_counts=}")
        print(f"{self.normals_counts=}")

    def __get_faces_without_offset(self):
        self.faces_without_offset = {}
        for group_id, (g_start, g_end) in enumerate(zip(self.groups_index, self.groups_index[1:])):
            for line in self.obj_lines[g_start:g_end]:
                if "f " in line:
                    f_edges_params = line.split(" ")
                    f_edges_params_without_offset = "f"
                    for f_edge_params in f_edges_params[1:]:
                        f_edge_param = f_edge_params.split("/")
                        v = int(f_edge_param[0]) - sum(self.vertices_counts[:group_id])
                        vt = int(f_edge_param[1]) - sum(self.texture_params_counts[:group_id])
                        vn = int(f_edge_param[2]) - sum(self.normals_counts[:group_id])
                        f_edge_param_without_offset = str(v) + "/" + str(vt) + "/" + str(vn)
                        f_edges_params_without_offset = " ".join(
                            [f_edges_params_without_offset, f_edge_param_without_offset])
                    if group_id in self.faces_without_offset:
                        self.faces_without_offset[group_id].append(f_edges_params_without_offset)
                    else:
                        self.faces_without_offset[group_id] = [f_edges_params_without_offset]
        print(f"{self.faces_without_offset=}")

    def __save_rebuilt_objs(self, output_dir: str):
        # 重建分组obj
        for group_id, (g_start, g_end) in enumerate(zip(self.groups_index, self.groups_index[1:])):
            f_first_index = g_start + next(
                index for index, line in enumerate(self.obj_lines[g_start:g_end]) if "f " in line)
            origin = "\n".join(self.obj_lines[g_start:f_first_index])
            faces = "\n".join(self.faces_without_offset[group_id])
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"{group_id}.obj"), "w", encoding="utf-8") as output:
                output.write("\n".join([origin, faces]))

    def split_by_group(self, output_dir: str):
        err = self.__obj2lines()
        if err:
            return
        self.__get_groups_index()
        self.__count_groups_coord()
        self.__get_faces_without_offset()
        self.__save_rebuilt_objs(output_dir)


if __name__ == "__main__":
    obj_splitter = ObjSplitter("origin.obj")
    obj_splitter.split_by_group("./output")
