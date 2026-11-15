# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""NumPy oracle for the requested GeoDa multivariate quantile intersection definition.

Run: /usr/bin/python3 modules/analysis/test/oracles/multivariate-quantile-lisa-numpy.py
Per-column type-5 quantiles (NumPy hazen), half-open indicators, then their intersection.
Conditional null distributions enumerate combinations of eligible rows excluding the focus,
so this oracle uses neither production quantile selection nor the production random stream.
As requested, an empty intersection stays empty (no MultiJoinCount BJC fallback).
"""
import itertools
import json
import pathlib

import numpy as np

cycle = [[(row + 1) % 8, (row + 7) % 8] for row in range(8)]
cases = [
    ('two variables overlap', [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 7, 6, 5, 4]], [[2, 2], [4, 4]], [1] * 8, cycle),
    ('ties and three variable intersection', [[0, 0, 1, 1, 1, 1, 2, 2], [2, 2, 2, 2, 1, 1, 0, 0], [9] * 8], [[4, 3], [2, 2], [3, 3]], [1] * 8, cycle),
    ('empty joint class', [list(range(8)), list(reversed(range(8)))], [[2, 2], [2, 2]], [1] * 8, cycle),
    ('empty tied marginal class', [[1] * 8, list(range(8))], [[4, 1], [2, 2]], [1] * 8, cycle),
    ('different missing rows and common mask', [[0, 1, 2, 3, 4, 5, None, 7], [None, 2, 2, 2, 4, 5, 6, 7]], [[2, 2], [3, 2]], [1, 1, 1, 1, 1, 0, 1, 1], [[1], [0, 2], [1, 3], [2], [], [4], [7], [6]]),
    ('no finite values in one variable', [list(range(8)), [None] * 8], [[2, 2], [2, 2]], [1] * 8, cycle),
    ('all ties and all joins', [[5] * 8, [7] * 8], [[4, 4], [2, 2]], [1] * 8, cycle),
]


def encode_boundary(value):
    if np.isnan(value):
        return 'NaN'
    if np.isneginf(value):
        return '-Infinity'
    if np.isposinf(value):
        return 'Infinity'
    return float(value)


fixtures = []
for name, columns, settings, mask, neighbors in cases:
    values = np.array(columns, dtype=np.float32)
    included = np.isfinite(values) & np.array(mask, dtype=bool)[None, :]
    indicators = np.zeros(values.shape, dtype=bool)
    boundaries = []
    counts = []
    for variable, (quantile_count, selected_quantile) in enumerate(settings):
        sample = values[variable, included[variable]]
        counts.append(int(sample.size))
        bounds = np.quantile(sample, [(selected_quantile - 1) / quantile_count, selected_quantile / quantile_count], method='hazen') if sample.size else np.array([np.nan, np.nan])
        if selected_quantile == 1:
            bounds[0] = -np.inf
        if selected_quantile == quantile_count:
            bounds[1] = np.inf
        boundaries.extend(encode_boundary(bound) for bound in bounds)
        indicators[variable] = included[variable] & (values[variable] >= bounds[0]) & (values[variable] < bounds[1])
    membership = np.all(indicators, axis=0).astype(np.uint32)
    valid = np.all(included, axis=0)
    active = [[neighbor for neighbor in row_neighbors if valid[neighbor]] if valid[row] else [] for row, row_neighbors in enumerate(neighbors)]
    analysis = valid & np.array([len(row_neighbors) > 0 for row_neighbors in active])
    observed = []
    upper_probabilities = []
    classes = []
    for row in range(8):
        draws = [neighbor for neighbor in active[row] if analysis[neighbor]]
        statistic = int(membership[row] * np.sum(membership[draws]))
        observed.append(statistic if analysis[row] else None)
        classes.append(2 if not valid[row] else (3 if not active[row] else 0))
        if not statistic:
            upper_probabilities.append(None)
            continue
        pool = np.flatnonzero(analysis & (np.arange(8) != row))
        permutations = list(itertools.combinations(pool, len(draws)))
        upper_probabilities.append(sum(np.sum(membership[list(draw)]) >= statistic for draw in permutations) / len(permutations))
    fixtures.append(dict(name=name, columns=columns, settings=settings, mask=mask, neighbors=neighbors,
                         boundaries=boundaries, validCounts=counts, membership=membership.tolist(),
                         validMask=valid.astype(np.uint32).tolist(), observed=observed,
                         upperTailProbabilities=upper_probabilities, baseClasses=classes,
                         islandCount=int(np.count_nonzero(valid & ~analysis))))

path = pathlib.Path(__file__).with_name('fixtures') / 'multivariate-quantile-lisa-numpy.json'
path.write_text(json.dumps(dict(oracle='NumPy hazen quantiles and exhaustive conditional combinations', cases=fixtures), indent=2, allow_nan=False) + '\n')
