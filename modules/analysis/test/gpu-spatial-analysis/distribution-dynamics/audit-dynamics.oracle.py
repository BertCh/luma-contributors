# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/distribution-dynamics/audit-dynamics.oracle.py

Independent definitions: numpy median; within-group absolute and squared deviations;
multinomial transition MLE; Pearson/LR homogeneity with nonempty-row degrees of freedom;
stationarity by a linear system after a directed support reachability check (not GTH).
No production code or captured production results are used.
"""

import json
import numpy as np
from scipy.stats import chi2
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


def fit(values, classes, mask):
    selected = np.isfinite(values) & (classes < 4) & mask.astype(bool)
    groups = [values[selected & (classes == group)] for group in range(4)]
    groups.append(values[selected])
    medians = [float(np.median(group)) if group.size else 0 for group in groups]
    absolute = [float(np.abs(group - center).sum()) for group, center in zip(groups, medians)]
    squared = [float(np.square(group - group.mean()).sum()) if group.size else 0 for group in groups]
    return dict(counts=[group.size for group in groups], medians=medians,
                absolute=absolute, squared=squared,
                summary=[1 - sum(absolute[:4]) / absolute[4], sum(absolute[:4]),
                         absolute[4], 1 - sum(squared[:4]) / squared[4], sum(squared[:4])])


def stationary(counts):
    active = (counts.sum(axis=0) + counts.sum(axis=1)) > 0
    matrix = counts[np.ix_(active, active)]
    if not matrix.size or np.any(matrix.sum(axis=1) == 0):
        return [float('nan')] * len(counts)
    if connected_components(csr_matrix(matrix), directed=True, connection='strong')[0] != 1:
        return [float('nan')] * len(counts)
    probabilities = matrix / matrix.sum(axis=1, keepdims=True)
    equations = probabilities.T - np.eye(len(matrix))
    equations[-1] = 1
    target = np.zeros(len(matrix))
    target[-1] = 1
    answer = np.zeros(len(counts))
    answer[active] = np.linalg.solve(equations, target)
    return answer.tolist()


def diagnostics(counts):
    pooled = counts.sum(axis=0)
    statistic = likelihood = degrees = 0
    for origin in range(counts.shape[1]):
        observed = counts[:, origin, :]
        totals = observed.sum(axis=1)
        if pooled[origin].sum() == 0:
            continue
        nonempty = totals > 0
        destinations = pooled[origin] > 0
        observed = observed[np.ix_(nonempty, destinations)]
        expected = np.outer(totals[nonempty], pooled[origin, destinations] / pooled[origin].sum())
        statistic += np.sum((observed - expected)**2 / expected)
        positive = observed > 0
        likelihood += 2 * np.sum(observed[positive] * np.log(observed[positive] / expected[positive]))
        degrees += (nonempty.sum() - 1) * (destinations.sum() - 1)
    states = [stationary(chain) for chain in [pooled, *counts]]
    totals = counts.sum(axis=2)
    return dict(homogeneity=[float(statistic), float(chi2.sf(statistic, degrees)),
                             float(likelihood), float(chi2.sf(likelihood, degrees)), int(degrees)],
                steadyStates=np.array(states).ravel().tolist(),
                stateSparsity=[int(((totals > 0) & (totals < 5)).sum()), int((totals == 0).sum())],
                converged=[int(all(not np.isnan(state).any() for state, chain in zip(states, [pooled, *counts]) if chain.sum()))])


values = np.array([-4, -1, -1, 2, 2, 8, 10, 20, 50, np.nan, np.inf, 7.])
classes = np.array([0, 0, 0, 1, 1, 1, 3, 3, 3, 0, 1, 0xffffffff])
mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1])
print('fit', json.dumps(fit(values, classes, mask)))
print('fit affine', json.dumps(fit(2 * values + 3, classes, mask)))

# Eight locations, four periods. Row 7 masked; no-class and out-of-range conditions
# invalidate pairs, and class 2 is unused. Counts condition on the origin period.
panel = np.array([[0, 0, 1, 1, 0, 1, 0, 1],
                  [0, 1, 1, 0, 0, 1, 0xffffffff, 0],
                  [1, 1, 0, 0, 1, 0, 0, 1],
                  [1, 0, 0, 1, 0, 1, 1, 0]], dtype=np.int64)
conditions = np.array([[0, 1, 0, 1, 0, 1, 0, 0],
                       [1, 0, 1, 0, 1, 0, 0, 0],
                       [0, 1, 0, 1, 0, 1, 2, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]])
for lag in [1, 2]:
    counts = np.zeros((2, 3, 3), dtype=np.int64)
    ignored = 0
    for period in range(4 - lag):
        for row in range(8):
            origin, destination, condition = panel[period, row], panel[period + lag, row], conditions[period, row]
            if row == 7 or origin >= 3 or destination >= 3 or condition >= 2:
                ignored += 1
            else:
                counts[condition, origin, destination] += 1
    totals = counts.sum(axis=2)
    probabilities = np.divide(counts, totals[:, :, None], out=np.zeros_like(counts, dtype=float), where=totals[:, :, None] > 0)
    print('transitions', lag, json.dumps(dict(counts=counts.ravel().tolist(),
          probabilities=probabilities.ravel().tolist(), rowTotals=totals.ravel().tolist(), ignored=[ignored],
          **diagnostics(counts))))

for counts in [[[1, 0], [1, 0]], [[0, 1], [0, 1]], [[0, 1], [1, 0]], [[4, 0], [0, 0]]]:
    print('stationary support', counts, stationary(np.array(counts)))

# More than 256 conditions crosses both the public pooled-condition and global-sum boundaries.
# Include different transition laws, reducible chains, and a final empty condition/unused state.
parallel_counts = np.array([[[1, 3, 0], [2, 2, 0], [0, 0, 0]]] * 128 +
                           [[[4, 0, 0], [0, 4, 0], [0, 0, 0]]] * 128 +
                           [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
print('parallel diagnostics', json.dumps(diagnostics(parallel_counts)))

for first, second in [(4143, 4141), (574, 528)]:
    print('nearby regimes', first, second, json.dumps(diagnostics(np.array([[[first, second], [0, 0]], [[second, first], [0, 0]]]))))
