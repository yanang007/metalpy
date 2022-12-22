from discretize import TensorMesh

from metalpy.utils.dhash import dhash


def dhash_discretize_mesh(mesh: TensorMesh):
    assert isinstance(mesh, TensorMesh), 'hash_discretize_mesh() currently supports only TensorMesh.'
    origin = mesh.origin
    hs = mesh.h

    return dhash(*origin, *hs)
