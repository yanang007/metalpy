import os.path
import pickle

import numpy as np
import pyvista as pv
import tqdm

from .algos import QuickUnion, ConnectedTriangleSurfaces
from .dhash import _hash_array
from .file import make_cache_file, make_cache_directory
from .hash import hash_numpy_array
from .obj_splitter import ObjSplitter
from .time import Timer


def hash_model(model, n_samples=10):
    return hash_numpy_array(model.points, n_samples)


def dhash_model(model, n_samples=10):
    return _hash_array(model.points, n_samples)


def extract_model_list_bounds(model_list):
    bounds = None
    for m in model_list:
        if bounds is None:
            bounds = np.asarray(m.bounds)
            continue
        bounds[1::2] = np.max([bounds[1::2], m.bounds[1::2]], axis=0)
        bounds[0::2] = np.min([bounds[0::2], m.bounds[0::2]], axis=0)

    return bounds


def merge_as_multiblock(model_list):
    mb = pv.MultiBlock()
    for m in model_list:
        mb.append(m)

    return mb


def load_model_file(model_file, verbose=True):
    model_name = os.path.basename(model_file)
    cache_path = make_cache_file(model_name + '.vtk')

    if os.path.exists(cache_path):
        if verbose:
            print(f'Loading cached model from {cache_path}...')
        model_file = cache_path

    reader = pv.get_reader(model_file)

    if verbose:
        reader.show_progress()

    timer = Timer()
    with timer:
        model = reader.read()

    if timer.elapsed > 5 and cache_path != model_file:
        if verbose:
            print(f'Saving loaded model to {cache_path}...')
        model.save(cache_path, binary=True)

    return model


def load_grouped_file(model_file, verbose=True):
    if os.path.splitext(model_file)[1] != '.obj':
        return None  # 目前只支持obj文件

    splitter = ObjSplitter(model_file)
    model_name = os.path.basename(model_file)
    cache_path = make_cache_directory(os.path.splitext(model_name))
    if not os.path.exists(cache_path):
        splitter.split_by_group(cache_path)

    models = []
    for sub_model_file in os.listdir(cache_path):
        sub_model_path = os.path.join(cache_path, sub_model_file)
        models.append(load_model_file(sub_model_path))

    return models


def split_models_in_memory(model, verbose=True):
    cache = make_cache_file(f'{dhash_model(model)}.sub')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            if verbose:
                print(f'Loading split models from {cache}...')
            ret = pickle.load(f)
    else:
        models = split_models_in_memory_pointwisely(model, verbose=verbose)
        ret = []

        if verbose:
            progress = tqdm.tqdm(total=model.n_points, desc='Splitting models by edge connectivity')
        else:
            progress = None

        for m in models:
            ms = split_models_in_memory_edgewisely(m, verbose=verbose, pointwise_progress=progress)
            ret.extend(ms)

        with open(cache, 'wb') as f:
            if verbose:
                print(f'Saving split models to {cache}...')
            pickle.dump(models, f)

    return ret


def split_models_in_memory_edgewisely(model, verbose=True, pointwise_progress=None):
    points = ConnectedTriangleSurfaces()
    model_faces = model.faces
    i = 0
    faces_size = len(model_faces)

    if verbose and pointwise_progress is None:
        pointwise_progress = tqdm.tqdm(total=faces_size, desc='Splitting models by edge connectivity')

    while i < faces_size:
        nv = model_faces[i]
        pts = model_faces[i + 1:i + nv + 1]
        i = i + nv + 1
        points.add(pts)
        if verbose:
            pointwise_progress.update(nv + 1)

    models = []
    for g in points.get_groups():
        indices = np.asarray(list(g))
        models.append(model.extract_points(indices, adjacent_cells=False))

    return models


def split_models_in_memory_pointwisely(model, verbose=True):
    """
    按点连通性来划分子几何体，对一些情况无法适用
    """
    unions = QuickUnion(model.points.shape[0])
    model_faces = model.faces
    i = 0
    faces_size = len(model_faces)

    if verbose:
        progress = tqdm.tqdm(total=faces_size, desc='Splitting models by point connectivity')

    while i < faces_size:
        nv = model_faces[i]
        pts = model_faces[i + 1:i + nv + 1]
        i = i + nv + 1
        p1 = pts[0]
        for p2 in pts[1:]:
            unions.connect(p1, p2)

        if verbose:
            progress.update(nv + 1)

    if verbose:
        progress.close()

    unions.collapse(verbose)
    groups = np.unique(unions.unions)

    models = []

    if verbose:
        groups = tqdm.tqdm(groups, desc='Extracting sub-models')

    for g in groups:
        indices = np.argwhere(unions.unions == g).squeeze()
        sub_model = model.extract_points(indices, adjacent_cells=False)
        models.append(sub_model.extract_surface())

    return models
