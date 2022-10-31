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
    """
    :param strength: 场强
    :param inclination: 磁倾角
    :param declination: 磁偏角
    :return:
    """
    return Field(strength, inclination, declination)
