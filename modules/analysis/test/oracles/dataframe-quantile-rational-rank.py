# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NumPy and exact-rational references for the rational-probability rank of GPUColumnQuantiles.

Writes fixtures/dataframe-quantile-rational-rank.json next to this script (consumed by
test/gpu-dataframe/column-classification/gpu-column-quantiles-rational.spec.ts). Run from anywhere:
    <venv>/bin/python -I modules/analysis/test/oracles/dataframe-quantile-rational-rank.py
in a venv with numpy (the version is recorded in the fixture). Needs about 1 GB of memory for the
33,554,432-row case.

Math review 2026-11-10, finding dataframe-08. The default rank of GPUColumnQuantiles is
`h = fround(fround(n - 1) * fround(p))`, which loses the fraction from 2^23 rows and uses a float32
probability. The optional `rationalProbabilities` prop gives `p = numerator / denominator` and the
kernel computes `floor((n - 1) numerator / denominator)` and the remainder with 64-bit integers.

Two references are stored for every case, both in binary64 on the float32 data:
- `numpy`: `numpy.quantile(x.astype(float64), numerator / denominator, method=...)` for the five
  methods the class supports (`lower`, `higher`, `nearest`, `linear`, `midpoint`).
- `exact`: the definition evaluated with Python integers and `fractions.Fraction` on the sorted
  column: `lo = floor((n - 1) p)`, fraction `((n - 1) numerator mod denominator) / denominator`.
  No float product is formed, so this is what the integer WGSL must return.
NumPy forms the virtual index `(n - 1) * q` in float64 with a float64 `q`. When the exact product is
an integer `k` the float64 product can land just below it or just above it. Just below (101 rows,
`q = 0.29`: 28.999999999999996) numpy's `floor` is `k - 1`, so `lower` and `midpoint` return a
different order statistic; just above (101 rows, `q = 0.55`: 55.00000000000001) its `ceil` is
`k + 1`, so `higher` and `midpoint` do. `linear` still agrees to rounding and `nearest` is not
affected. On the grid here there are 22 such cases, 6 below and 16 above, all at 101 rows. The
script asserts that this is the only way the two references differ, stores the exact value
everywhere and lists the numpy value where it differs (`numpyDifferences`).

Sections of the fixture:
- `small`: one column of 5,000 float32 values (multiples of 1/8 with ties); for each count `n` the
  sample is its first `n` rows. 256 probabilities: every `a / b` for 15 denominators, and 13
  with large or non-reduced terms.
- `float32Path`: where the default float32 rank has another `floor(h)` than numpy on the same
  grid (the defect the finding reports; the default path keeps it), with a few cases to pin.
- `large`: step columns of 2^24, 2^24 + 1 and 2^25 rows. Row `i` holds the number of exact lower
  ranks below `i`, so each quantile is `level + fraction` and a rank that is off by one row moves
  the result by about 1. The spec rebuilds the column from `thresholds`.
- `rank`: `(count, numerator, denominator)` triples up to count 2^32 - 1 with the exact lower rank,
  the remainder and the bits of the correctly rounded float32 fraction, for a spec that runs the
  rank WGSL alone on counts no GPU buffer can hold.
"""

import json
import os
import platform
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'dataframe-quantile-rational-rank.json')
METHODS = ['lower', 'higher', 'nearest', 'linear', 'midpoint']

SMALL_ROW_COUNT = 5000
SMALL_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17, 21, 32, 33, 50, 64, 100, 101, 128,
                257, 1000, 1001, 4999, 5000]
SMALL_DENOMINATORS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 25, 100]
UINT32_MAXIMUM = 2 ** 32 - 1
EXTRA_PROBABILITIES = [
    [1, UINT32_MAXIMUM],
    [UINT32_MAXIMUM - 1, UINT32_MAXIMUM],
    [UINT32_MAXIMUM, UINT32_MAXIMUM],
    [2 ** 31, UINT32_MAXIMUM],
    [858993459, UINT32_MAXIMUM],  # exactly 1 / 5
    [123456789, 1000000007],
    [700000000, 1000000000],  # 7 / 10, not reduced
    [950000000, 1000000000],
    [333333333, 1000000000],
    [1, 1000],
    [999, 1000],
    [2147483647, 4294967294],  # 1 / 2, not reduced
    [3, 4294967292],
]
LARGE_COUNTS = [2 ** 24, 2 ** 24 + 1, 2 ** 25]
LARGE_PROBABILITIES = [
    [19, 20], [7, 10], [1, 3], [2, 3], [1, 2], [1, 4], [3, 4], [0, 1], [1, 1], [999, 1000],
    [1, 1000], [1, 7], [5, 7], [99, 100], [1, 100], [123456789, 1000000007],
    [1, UINT32_MAXIMUM], [UINT32_MAXIMUM - 1, UINT32_MAXIMUM], [UINT32_MAXIMUM, UINT32_MAXIMUM],
    [2 ** 31, UINT32_MAXIMUM], [9, 10], [3, 10], [1, 10], [95, 100],
]
RANK_COUNTS = [1, 2, 3, 10, 11, 101, 2 ** 23, 2 ** 23 + 1, 2 ** 24, 2 ** 24 + 1, 2 ** 24 + 2,
               2 ** 25, 123456789, 2 ** 31 - 1, 2 ** 31, UINT32_MAXIMUM]
RANK_PROBABILITIES = LARGE_PROBABILITIES + [
    [UINT32_MAXIMUM - 1, UINT32_MAXIMUM - 1], [1, 3], [16777215, 16777216], [1, 2 ** 31],
    [2 ** 31 - 1, 2 ** 31], [4194304, 16777217],
]


def get_exact_quantile(sorted_values, numerator, denominator, method):
    """The definition on exact rationals; `sorted_values` are Fractions."""
    last = len(sorted_values) - 1
    product = last * numerator
    lower = product // denominator
    remainder = product % denominator
    upper = lower + (1 if remainder else 0)
    if method == 'lower':
        return sorted_values[lower]
    if method == 'higher':
        return sorted_values[upper]
    if method == 'nearest':
        # Round half to even, as numpy.around.
        if 2 * remainder < denominator:
            return sorted_values[lower]
        if 2 * remainder > denominator:
            return sorted_values[lower + 1]
        return sorted_values[lower if lower % 2 == 0 else lower + 1]
    if method == 'midpoint':
        return (sorted_values[lower] + sorted_values[upper]) / 2
    high = sorted_values[min(lower + 1, last)]
    return sorted_values[lower] + (high - sorted_values[lower]) * Fraction(remainder, denominator)


def is_index_rounded_across_integer(count, numerator, denominator):
    """True when the exact (n - 1) p is an integer and numpy's float64 product is not it."""
    product = Fraction((count - 1) * numerator, denominator)
    return product.denominator == 1 and float(count - 1) * (numerator / denominator) != float(product)


def get_references(values, probabilities):
    """Per method: exact binary64 values and the numpy values that differ from them."""
    count = len(values)
    doubles = values.astype(np.float64)
    sorted_fractions = [Fraction(float(value)) for value in np.sort(doubles)]
    scale = max(1.0, float(np.max(np.abs(doubles))))
    quantiles = [numerator / denominator for numerator, denominator in probabilities]
    exact = {}
    numpy_differences = {}
    for method in METHODS:
        reference = np.quantile(doubles, quantiles, method=method)
        exact[method] = []
        numpy_differences[method] = []
        for index, (numerator, denominator) in enumerate(probabilities):
            value = float(get_exact_quantile(sorted_fractions, numerator, denominator, method))
            exact[method].append(value)
            if abs(float(reference[index]) - value) > 1e-9 * scale:
                # The only admissible cause: the float64 virtual index fell across an integer.
                assert method != 'linear', (count, numerator, denominator)
                assert is_index_rounded_across_integer(count, numerator, denominator), (
                    count, numerator, denominator, method)
                numpy_differences[method].append([index, float(reference[index])])
    return exact, numpy_differences


def round_fraction_to_float32_bits(value):
    """Bits of the float32 nearest to an exact Fraction, ties to an even significand."""
    if value == 0:
        return 0
    guess = np.float32(float(value))
    candidates = [np.nextafter(guess, np.float32(-np.inf)), guess, np.nextafter(guess, np.float32(np.inf))]
    best = min(candidates, key=lambda candidate: (abs(Fraction(float(candidate)) - value),
                                                  int(candidate.view(np.uint32)) & 1))
    return int(best.view(np.uint32))


def get_float32_path_floor(count, probability):
    """floor of the default kernel rank, fround(fround(n - 1) * fround(p))."""
    return int(np.floor(np.float32(count - 1) * np.float32(probability)))


def main():
    generator = np.random.default_rng(20261111)
    column = (generator.integers(-8000, 8001, size=SMALL_ROW_COUNT) / 8).astype(np.float32)
    probabilities = [[numerator, denominator] for denominator in SMALL_DENOMINATORS
                     for numerator in range(denominator + 1)] + EXTRA_PROBABILITIES
    assert len(probabilities) == 256

    small_cases = []
    case_count = 0
    numpy_difference_count = 0
    for count in SMALL_COUNTS:
        exact, numpy_differences = get_references(column[:count], probabilities)
        case_count += len(probabilities) * len(METHODS)
        numpy_difference_count += sum(len(entries) for entries in numpy_differences.values())
        small_cases.append({'count': count, 'exact': exact, 'numpyDifferences': numpy_differences})

    # The default float32 rank against numpy's float64 virtual index, on the same grid.
    float32_differences = []
    for count in SMALL_COUNTS:
        doubles = column[:count].astype(np.float64)
        for numerator, denominator in probabilities:
            quantile = numerator / denominator
            numpy_floor = int(np.floor((count - 1) * quantile))
            if get_float32_path_floor(count, quantile) != numpy_floor:
                float32_differences.append({
                    'count': count,
                    'probability': [numerator, denominator],
                    'float32Floor': get_float32_path_floor(count, quantile),
                    'numpyFloor': numpy_floor,
                    'numpyLower': float(np.quantile(doubles, quantile, method='lower')),
                    'numpyLinear': float(np.quantile(doubles, quantile, method='linear')),
                })
    # Keep the cases whose `lower` order statistic is far from its neighbour, so a spec can see it.
    pinned = []
    for entry in float32_differences:
        sorted_values = np.sort(column[:entry['count']].astype(np.float64))
        if abs(sorted_values[entry['float32Floor']] - sorted_values[entry['numpyFloor']]) >= 0.125:
            pinned.append(entry)
    pinned = pinned[:: max(1, len(pinned) // 8)][:8]

    large_cases = []
    for count in LARGE_COUNTS:
        thresholds = sorted({(count - 1) * numerator // denominator
                             for numerator, denominator in LARGE_PROBABILITIES})
        values = np.searchsorted(np.asarray(thresholds), np.arange(count), side='left').astype(np.float32)
        exact, numpy_differences = get_references(values, LARGE_PROBABILITIES)
        float32_linear = None
        if count <= 2 ** 24:
            # What the default path returns for 19 / 20: the float32 rank has no fraction here.
            height = np.float32(count - 1) * np.float32(19 / 20)
            lower = int(np.floor(height))
            float32_linear = float(values[lower] + (values[min(lower + 1, count - 1)] - values[lower])
                                   * np.float32(height - np.floor(height)))
        large_cases.append({
            'count': count,
            'thresholds': thresholds,
            'exact': exact,
            'numpyDifferences': numpy_differences,
            'float32PathLinearAtFirstProbability': float32_linear,
        })
        del values

    rank_cases = []
    for count in RANK_COUNTS:
        for numerator, denominator in RANK_PROBABILITIES:
            product = (count - 1) * numerator
            remainder = product % denominator
            rank_cases.append([count, numerator, denominator, product // denominator, remainder,
                               round_fraction_to_float32_bits(Fraction(remainder, denominator))])

    fixture = {
        'generator': 'test/oracles/dataframe-quantile-rational-rank.py',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__},
        'methods': METHODS,
        'small': {
            'values': [float(value) for value in column],
            'probabilities': probabilities,
            'cases': small_cases,
            'caseCount': case_count,
            'numpyDifferenceCount': numpy_difference_count,
        },
        'float32Path': {
            'comparedCount': len(SMALL_COUNTS) * len(probabilities),
            'floorDifferenceCount': len(float32_differences),
            'pinned': pinned,
        },
        'large': {'probabilities': LARGE_PROBABILITIES, 'cases': large_cases},
        'rank': {
            'columns': ['count', 'numerator', 'denominator', 'lower', 'remainder', 'fractionBits'],
            'cases': rank_cases,
        },
    }
    with open(OUTPUT, 'w', encoding='utf-8') as file:
        json.dump(fixture, file, separators=(',', ':'))
        file.write('\n')
    print(f'wrote {OUTPUT}: {case_count} small cases, {numpy_difference_count} numpy differences, '
          f'{len(float32_differences)} float32-path floor differences of '
          f'{len(SMALL_COUNTS) * len(probabilities)}, {len(rank_cases)} rank triples')


if __name__ == '__main__':
    main()
