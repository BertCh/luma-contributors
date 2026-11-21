# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run: /usr/bin/python3 modules/analysis/test/analysis-core/grid-coordinate-bump.oracle.py

Independent IEEE binary32 evaluation of the published alpha.16 definition documented by
GPU_GRID_COORDINATE_WGSL: endpoint clamps; rescale only if the domain width overflows;
truncate ((value - minimum) / (maximum - minimum)) * size. No repository imports.
The overflow row describes IEEE operations; a device may flush division intermediates.
"""

import numpy as np
from fractions import Fraction


def coordinate(value, minimum, maximum, size):
    value, minimum, maximum = map(np.float32, (value, minimum, maximum))
    if maximum == minimum or value <= minimum:
        return 0
    if value >= maximum:
        return size - 1
    if minimum < 0 and maximum > np.float32(np.finfo(np.float32).max + minimum):
        scale = max(abs(minimum), abs(maximum))
        value, minimum, maximum = [np.float32(item / scale) for item in (value, minimum, maximum)]
    fraction = np.float32(np.float32(value - minimum) / np.float32(maximum - minimum))
    return min(int(np.float32(fraction * np.float32(size))), size - 1)


for value, minimum, maximum, size in [
    (-177, -180, 180, 360),
    (103, 100, 460, 360),
    (0, -180, 180, 360),
    (180, -180, 180, 360),
    (0, -3e38, 3e38, 360),
    (-177.000001, -180, 180, 360),
    (-357, -360, 0, 360),
    (-180, -180, 180, 360),
    (7, 7, 7, 360),
    (-18.5, -20.5, 20.5, 41),
    (-17.5, -20.5, 20.5, 41),
    (0, 0, 41, 41),
    (2, 0, 41, 41),
    (40, 0, 41, 41),
]:
    print((value, minimum, maximum, size), coordinate(value, minimum, maximum, size))


# WGSL permits 2.5 ULP division error and either adjacent rounding result for multiplication.
# https://www.w3.org/TR/WGSL/#floating-point-accuracy
# Enumerate the f32 quotients near exact x/41, using rational arithmetic for the error test.
# No observed GPU output is an input to this calculation.
for value in [2, 40]:
    exact_quotient = Fraction(value, 41)
    nearest = np.float32(value / 41)
    unit = Fraction.from_float(float(np.spacing(nearest)))
    cells = set()
    for steps in range(-4, 5):
        quotient = nearest
        for _ in range(abs(steps)):
            quotient = np.nextafter(quotient, np.float32(-np.inf if steps < 0 else np.inf))
        if abs(Fraction.from_float(float(quotient)) - exact_quotient) > Fraction(5, 2) * unit:
            continue
        exact_product = Fraction.from_float(float(quotient)) * 41
        product = np.float32(float(exact_product))
        products = [product]
        rounded_product = Fraction.from_float(float(product))
        if rounded_product != exact_product:
            direction = np.inf if rounded_product < exact_product else -np.inf
            products.append(np.nextafter(product, np.float32(direction)))
        cells.update(min(int(product), 40) for product in products)
    expected = {1, 2} if value == 2 else {39, 40}
    assert cells == expected, (value, cells)
    print('WGSL division/rounding cells', value, sorted(cells))

# Review R11: an allowed quotient for the old interior y=2 sample on a three-row grid.
# Error is 2/3 ULP; multiplication then truncation places (4,2) in row 1, flat cell 7.
quotient = np.float32(0.6666666269302368)
error_units = abs(Fraction.from_float(float(quotient)) - Fraction(2, 3)) / Fraction(1, 2**24)
scaled = np.float32(quotient * np.float32(3))
assert error_units <= Fraction(5, 2)
assert float(scaled) == 1.9999998807907104
assert int(scaled) * 4 + 3 == 7
print('old interior (4,2) permitted flat cell', int(scaled) * 4 + 3)
print('actual inclusive maximum (4,3) flat cell', coordinate(3, 0, 3, 3) * 4 + 3)
