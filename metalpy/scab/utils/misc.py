from typing import Union, Iterable

import numpy as np
from SimPEG.utils import mat_utils


class Field:
    def __init__(self, strength, inclination, declination):
        self.definition = (strength, inclination, declination)

    @property
    def strength(self):
        return self.definition[0]

    @property
    def inclination(self):
        return self.definition[1]

    @property
    def declination(self):
        return self.definition[2]

    @property
    def unit_vector(self):
        return mat_utils.dip_azimuth2cartesian(
            self.inclination,
            self.declination,
        ).squeeze()

    def __iter__(self):
        for v in self.definition:
            yield v

    def __getitem__(self, index):
        return self.definition[index]

    def __len__(self):
        return 3

    def with_strength(self, strength):
        return Field(strength, self.inclination, self.declination)


def define_inducing_field(strength, inclination, declination) -> Field:
    """定义一个外部激发场

    Parameters
    ----------
    strength
        场强
    inclination
        场倾角
    declination
        场偏角

    Returns
    -------
    ret
        定义的物理场
    """
    return Field(strength, inclination, declination)


def define_magnetics_survey(receiver_points: np.ndarray,
                            components: Union[str, Iterable[str]],
                            *other_receivers: Iterable[Union[np.ndarray, Union[str, Iterable[str]]]],
                            inducing_field: Field = None):
    """定义一个磁力学的Survey

    Parameters
    ----------
    receiver_points
        接收点
    components
        接收点所需要计算的分量
    other_receivers
        其他接受点和对应分量，可以有多个，按 rp1, comp1, rp2, comp2, ... 的顺序传入
    inducing_field
        外部激发场，如果为None则使用默认值 (50000nT, 90°, 0°)

    Returns
    -------
    ret
        包含接收点和外部场信息的磁力学Survey
    """
    from SimPEG.potential_fields import magnetics

    receiver_list = magnetics.receivers.Point(receiver_points, components=components)
    receiver_list = [receiver_list]

    assert len(other_receivers) % 2 == 0, '*other_receivers* must be paris of receiver_points, components, ' \
                                          'etc. define_magnetics_survey(rp1, comp1, rp2, comp2, rp3, comp3)'
    for receiver_points, components in zip(other_receivers[::2], other_receivers[1::2]):
        receiver_list.append(magnetics.receivers.Point(receiver_points, components=components))

    source_field = magnetics.sources.SourceField(
        receiver_list=receiver_list, parameters=inducing_field
    )
    survey = magnetics.survey.Survey(source_field)

    return survey
