from numpy.testing import assert_almost_equal

from metalpy.scab.modelling.shapes import Ellipsoid


def test_shape3d_to_polydata_with_transforms():
    # 测试transforms的等价性
    sphere = Ellipsoid(10, 8, 6)
    raw = sphere.to_local_polydata()

    delta = [5, 0, 0]
    translated = sphere.translate(*delta)
    translated_model = translated.to_polydata()
    translated_model_truth = raw.translate(delta)
    assert_almost_equal(translated_model_truth.points, translated_model.points, decimal=6)

    rotated = translated.rotate(90, 90, 0, degrees=True, seq='xyz')  # vtk使用的是外旋
    rotated_model = rotated.to_polydata()
    rotated_model_truth = translated_model_truth.rotate_x(90).rotate_y(90)
    assert_almost_equal(rotated_model.points, rotated_model_truth.points, decimal=6)
