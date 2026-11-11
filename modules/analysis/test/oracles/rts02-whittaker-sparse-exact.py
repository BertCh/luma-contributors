# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""RTS-02 reference for the Whittaker smoother on very sparse support.

Solves `(W + lambda D'D) z = W y` on a regular axis in exact rational arithmetic (Python
`fractions`, standard library only), so the reference is defined wherever the system is
non-singular, however badly conditioned it is in floating point. A binary64 factorisation of the
normal equations is not a usable reference here: with three observations at one end of 256 slices
and `lambda = 1e7` its condition number is beyond 1e16.

`D` is the ordinary difference of order 1 to 3, `W` is 1 on valid slices and 0 elsewhere, the
values are dyadic so they are exact in float32. Output: fixtures/rts02-whittaker-sparse-exact.json.
"""
import json
import pathlib
import platform
from fractions import Fraction

OUTPUT = pathlib.Path(__file__).resolve().parent / 'fixtures' / 'rts02-whittaker-sparse-exact.json'
SLICE_COUNT = 256
STENCILS = {1: [-1, 1], 2: [1, -2, 1], 3: [-1, 3, -3, 1]}
SMOOTHING_PARAMETERS = [1, 100, 10_000, 1_000_000, 10_000_000]
# Dyadic observations around 300 (kelvin-like offset), exact in float32.
OBSERVATIONS = [300.25, 299.5, 301.0, 300.75, 299.25]


def get_positions(count, placement):
    """Slices that hold the `count` valid observations."""
    if placement == 'start':
        return list(range(count))
    if placement == 'end':
        return list(range(SLICE_COUNT - count, SLICE_COUNT))
    if placement == 'middle':
        first = SLICE_COUNT // 2 - count // 2
        return list(range(first, first + count))
    # Spread over the whole series, both ends included.
    return [round(index * (SLICE_COUNT - 1) / (count - 1)) for index in range(count)]


def solve_exactly(positions, order, smoothing):
    """Banded LDL^T in rationals; the matrix is positive definite, so no pivoting is needed."""
    count = SLICE_COUNT
    stencil = STENCILS[order]
    # band[row][offset] = A[row][row + offset], offset 0..order.
    band = [[Fraction(0)] * (order + 1) for _ in range(count)]
    right_side = [Fraction(0)] * count
    for index, position in enumerate(positions):
        band[position][0] += 1
        right_side[position] = Fraction(OBSERVATIONS[index])
    for row in range(count - order):
        for left in range(order + 1):
            for right in range(left, order + 1):
                band[row + left][right - left] += smoothing * stencil[left] * stencil[right]
    # Factor: after the loop band[row][0] is D[row] and band[row][k] is L[row + k][row].
    for row in range(count):
        for column in range(max(0, row - order), row):
            factor = band[column][row - column]
            band[row][0] -= factor * factor * band[column][0]
        for below in range(row + 1, min(row + order, count - 1) + 1):
            coupled = band[row][below - row]
            for column in range(max(0, below - order), row):
                coupled -= band[column][below - column] * band[column][row - column] * band[column][0]
            band[row][below - row] = coupled / band[row][0]
    solution = right_side
    for row in range(count):
        for column in range(max(0, row - order), row):
            solution[row] -= band[column][row - column] * solution[column]
    for row in range(count):
        solution[row] /= band[row][0]
    for row in range(count - 1, -1, -1):
        for below in range(row + 1, min(row + order, count - 1) + 1):
            solution[row] -= band[row][below - row] * solution[below]
    return [float(value) for value in solution]


def main():
    cases = []
    for count in (2, 3, 5):
        for placement in ('start', 'middle', 'end', 'spread'):
            positions = get_positions(count, placement)
            # The kernel needs more valid observations than the difference order.
            for order in range(1, min(3, count - 1) + 1):
                for smoothing in SMOOTHING_PARAMETERS:
                    cases.append({
                        'name': f'{count}-{placement}-order-{order}-lambda-{smoothing}',
                        'order': order,
                        'smoothing': smoothing,
                        'positions': positions,
                        'observations': OBSERVATIONS[:count],
                        'expected': [float(f'{value:.12g}') for value in solve_exactly(positions, order, smoothing)],
                    })
    fixture = {
        'source': 'exact rational solve of (W + lambda D\'D) z = W y, rounded to 12 significant digits; no library',
        'versions': {'python': platform.python_version()},
        'sliceCount': SLICE_COUNT,
        'cases': cases,
    }
    OUTPUT.write_text(json.dumps(fixture))
    print('wrote', OUTPUT, OUTPUT.stat().st_size, len(cases))


if __name__ == '__main__':
    main()
