# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/distribution-dynamics/audit-spatial-markov.oracle.py

Spatial lag: paired neighbor's value (unit weight), island lag zero, masked row excluded.
LISA: islands excluded from moments; signs of centered value and centered neighbor lag;
analytic conditional-randomization z from enumerating all other centered values for a
single neighbor (population variance, not sample variance), two-sided normal tails.
Neither the production TypeScript nor its test helpers are imported.
"""
import json
import numpy as np
from scipy.stats import norm, chi2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

values = np.array([[1, 2, 2, 7, 8, 10, 999, -999],
                   [8, 9, 10, 1, 2, np.nan, 999, -999],
                   [2, 8, 1, 10, 7, 2, 999, -999.]])
partners = np.array([1, 0, 3, 2, 5, 4])
no_class = 0xffffffff


def stationary(counts):
    active = (counts.sum(axis=0) + counts.sum(axis=1)) > 0
    selected = counts[np.ix_(active, active)]
    if not selected.size or np.any(selected.sum(axis=1) == 0) or connected_components(csr_matrix(selected), directed=True, connection='strong')[0] != 1:
        return [float('nan')] * len(counts)
    probabilities = selected / selected.sum(axis=1, keepdims=True)
    equations = probabilities.T - np.eye(len(selected))
    equations[-1] = 1
    target = np.zeros(len(selected))
    target[-1] = 1
    result = np.zeros(len(counts))
    result[active] = np.linalg.solve(equations, target)
    return result.tolist()


def transitions(states, conditions, classes, condition_count):
    counts = np.zeros((condition_count, classes, classes), dtype=np.int64)
    ignored = 0
    for period in range(2):
        for row in range(8):
            origin, destination, condition = states[period, row], states[period + 1, row], conditions[period, row]
            if row == 7 or origin >= classes or destination >= classes or condition >= condition_count:
                ignored += 1
            else:
                counts[condition, origin, destination] += 1
    totals = counts.sum(axis=2)
    probabilities = np.divide(counts, totals[:, :, None], out=np.zeros_like(counts, dtype=float), where=totals[:, :, None] > 0)
    pooled = counts.sum(axis=0)
    chains = [pooled, *counts] if condition_count > 1 else [pooled]
    steady = [stationary(chain) for chain in chains]
    result = dict(counts=counts.ravel().tolist(), probabilities=probabilities.ravel().tolist(),
                  rowTotals=totals.ravel().tolist(), ignored=[ignored],
                  steadyStates=np.array(steady).ravel().tolist(),
                  stateSparsity=[int(((totals > 0) & (totals < 5)).sum()), int((totals == 0).sum())],
                  converged=[int(all(not np.isnan(state).any() for state, chain in zip(steady, chains) if chain.sum()))])
    if condition_count > 1:
        statistic = likelihood = degrees = 0
        for origin in range(classes):
            observed = counts[:, origin, :]
            row_totals = observed.sum(axis=1)
            if pooled[origin].sum() == 0:
                continue
            nonempty = row_totals > 0
            destinations = pooled[origin] > 0
            observed = observed[np.ix_(nonempty, destinations)]
            expected = np.outer(row_totals[nonempty], pooled[origin, destinations] / pooled[origin].sum())
            statistic += np.sum((observed - expected)**2 / expected)
            positive = observed > 0
            likelihood += 2 * np.sum(observed[positive] * np.log(observed[positive] / expected[positive]))
            degrees += (nonempty.sum() - 1) * (destinations.sum() - 1)
        result['homogeneity'] = [float(statistic), float(chi2.sf(statistic, degrees)), float(likelihood), float(chi2.sf(likelihood, degrees)), int(degrees)]
    return result


lags = np.zeros_like(values)
lags[:, :6] = values[:, partners]
breaks = np.quantile(lags[:, :7][np.isfinite(lags[:, :7])], [0, 0.5, 1])
classes = np.where(np.isfinite(values), (values > 5).astype(int), no_class)
for closure in ['left', 'right']:
    lag_classes = np.where(lags < breaks[1] if closure == 'left' else lags <= breaks[1], 0, 1)
    lag_classes[~np.isfinite(lags)] = no_class
    lag_classes[:, 7] = no_class
    print('spatial', closure, json.dumps(dict(lagBreaks=breaks.tolist(), lagClassCount=[2], lagValues=lags.ravel().tolist(), lagClasses=lag_classes.ravel().tolist(), **transitions(classes, lag_classes, 2, 2))))

for gating in ['none', 'significance']:
    states = np.full(values.shape, no_class, dtype=np.int64)
    for period in range(3):
        included = np.isfinite(values[period, :6]) & np.isfinite(values[period, partners])
        active = np.flatnonzero(included)
        centered = values[period] - values[period, active].mean()
        for row in active:
            lag = centered[partners[row]]
            quadrant = 1 if centered[row] > 0 and lag > 0 else 2 if centered[row] < 0 and lag > 0 else 3 if centered[row] < 0 and lag < 0 else 4 if centered[row] > 0 and lag < 0 else 0
            others = centered[active[active != row]]
            statistic = (lag - others.mean()) / others.std(ddof=0) * np.sign(centered[row])
            states[period, row] = quadrant if gating == 'none' or 2 * norm.sf(abs(statistic)) <= 0.2 else 0
    print('lisa', gating, json.dumps(dict(quadrants=states.ravel().tolist(), **transitions(states, np.zeros_like(states), 5, 1))))
# Scaling x by 2 preserves every class and transition (lagValues/breaks double).
# LISA also permits an additive shift: x -> 2x+3 leaves centered signs and z unchanged.
