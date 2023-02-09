import math

ref_area = 508164597540055.75
area_factors = [
    1.0,
    1.003741849761155,
    1.8970972739048304,
    2.7118085839548,
    3.0342500406694364,
    3.1231014735135538,
    3.1457588045774316,
    3.151449027223487,
    3.1528731677136914,
    3.1532293013524657,
    3.1533183409109418,
    3.1533406011847736,
    3.1533461662766444,
    3.1533475575519319,
    3.1533479053694733,
    3.153347992323662,
    3.1533480140543539,
    3.1533480194967369,
    3.1533480208329561,
    3.1533480211297258,
]


def quadbin_area_zy(z, y):
    area_factor = area_factors[min(len(area_factors) - 1, z)]
    area = area_factor * ref_area / (1 << (z << 1))
    centery = 0 if z == 0 else (1 << (z - 1))
    if y < centery - 1 or y > centery:

        def zfactor(y, yoffset):
            return math.pow(
                math.cos(
                    2
                    * math.pi
                    * (
                        math.atan(
                            math.exp(-(2 * (y + yoffset) / (1 << z) - 1) * math.pi)
                        )
                        / math.pi
                        - 0.25
                    )
                ),
                2,
            )

        area *= zfactor(y, 0.5) / zfactor(centery, 0.5)
    return area
