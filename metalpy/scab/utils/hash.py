from discretize import TensorMesh

from metalpy.utils.hash import hash_numpy_array


def hash_discretize_mesh(mesh: TensorMesh):
    assert isinstance(mesh, TensorMesh), 'hash_discretize_mesh() currently supports only TensorMesh.'
    origin = mesh.origin
    hs = mesh.h

    return hash((*origin, *(hash_numpy_array(h) for h in hs)))
