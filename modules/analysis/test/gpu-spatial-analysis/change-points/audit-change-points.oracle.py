# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/change-points/audit-change-points.oracle.py

Enumerate all 2^(n-1) partitions, never the production dynamic recurrence. Segment costs
are direct NumPy residuals; slope fits use numpy.linalg.lstsq on original slice positions.
The relative variance floor is the explicitly documented convention (1e-11 * overall variance).
Pettitt enumerates cross-split sign comparisons; ties contribute zero and first maxima win.
"""
import itertools
import json
import numpy as np

series = [[1, 2, 1, 3, 9, 14, 10, 12], [1, np.nan, 1, 3, 9, 14, np.nan, 12],
          [3] * 8, [np.nan] * 8, [7, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]


def optimum(values, cost, penalty, minimum):
    positions = np.flatnonzero(np.isfinite(values))
    sample = np.array(values)[positions]
    if len(sample) < minimum:
        return dict(count=0, changes=[0xffffffff], required=0, optimum=float('nan'), overflow=0)
    overall = sample.mean()
    floor = 1e-11 * sample.var() if sample.var() > 0 else 1e-11
    candidates = []
    for cuts in itertools.product([False, True], repeat=len(sample)-1):
        edges = [0, *[index + 1 for index, cut in enumerate(cuts) if cut], len(sample)]
        if min(np.diff(edges)) < minimum:
            continue
        objective = penalty * (len(edges) - 2)
        for start, end in zip(edges[:-1], edges[1:]):
            values = sample[start:end]
            count = len(values)
            if cost == 'mean':
                objective += np.square(values - values.mean()).sum()
            elif cost == 'slope':
                design = np.column_stack([np.ones(count), positions[start:end]])
                residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
                objective += np.square(residual).sum()
            elif cost == 'poisson':
                total = values.sum()
                objective += -2 * total * np.log(total / count) if total > 0 else 0
            else:
                center = overall if cost == 'variance' else values.mean()
                variance = np.square(values - center).mean()
                objective += count * (np.log(2 * np.pi) + np.log(max(variance, floor)) + 1)
        # Reverse boundary tuple realizes the documented smallest-preceding-start tie rule.
        candidates.append((float(objective), tuple(reversed(edges[1:-1])), edges[1:-1]))
    objective, _, edges = min(candidates)
    changes = positions[edges].tolist()
    return dict(count=min(1, len(changes)), changes=changes[:1] if changes else [0xffffffff],
                required=len(changes), optimum=objective, overflow=int(len(changes) > 1))


for cost in ['mean', 'variance', 'mean-variance', 'poisson', 'slope']:
    minimum = 3 if cost == 'slope' else 2 if cost in ['variance', 'mean-variance'] else 1
    print(cost, json.dumps([optimum(values, cost, 3, minimum) for values in series]))
    transformed = [[2 * value + (0 if cost == 'poisson' else 3) for value in values] for values in series]
    scaled_penalty = 6 if cost == 'poisson' else 12 if cost in ['mean', 'slope'] else 3
    print(cost, 'affine', json.dumps([optimum(values, cost, scaled_penalty, minimum) for values in transformed]))

pettitt_series = [[1, 1, 1, 1, 8, 8, 8, 8], [np.nan, 1, 1, np.nan, 8, 8, 8, 8],
                  [3] * 8, [np.nan] * 8, [7, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                  [8, 3, 6, 1, 7, 2, 5, 4]]
for values in pettitt_series:
    positions = np.flatnonzero(np.isfinite(values))
    sample = np.array(values)[positions]
    statistics = [abs(sum(np.sign(first - second) for first in sample[:split] for second in sample[split:])) for split in range(1, len(sample))]
    statistic = int(max(statistics)) if statistics else 0
    change = int(positions[statistics.index(statistic) + 1]) if statistics else 0xffffffff
    count = len(sample)
    probability = min(1, 2 * np.exp(-6 * statistic**2 / (count**3 + count**2))) if count >= 2 else np.nan
    print('pettitt', json.dumps(dict(statistic=statistic, changeSlice=change, pValues=probability)))
