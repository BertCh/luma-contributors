# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/distribution-dynamics/audit-rank-markov.oracle.py

SciPy ordinal ranks, giddy descending rank = population - ascending ordinal rank.
Geographic rank follows the row occupying each rank; full rank follows each row's rank.
Stationarity uses a linear solve and a strong-components check, not GTH elimination.
"""
import json
import numpy as np
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

values = np.array([[8, 6, 6, 4, 2, 0, 50, 9],
                   [0, 8, 6, 6, 4, 2, 40, 8],
                   [2, 0, 8, 6, 6, 4, 30, np.nan],
                   [9, 1, 7, 3, 7, 5, 20, 7]])
mask = np.array([1, 1, 1, 1, 1, 1, 0, 1], dtype=bool)
included = mask & np.isfinite(values).all(axis=0)
population = included.sum()
for tie_order in ['giddy', 'row-id']:
    ranks = np.full(values.shape, 0xffffffff, dtype=np.int64)
    for period in range(4):
        ranks[period, included] = population - rankdata(values[period, included], method='ordinal') if tie_order == 'giddy' else rankdata(-values[period, included], method='ordinal') - 1
    occupants = np.full_like(ranks, 0xffffffff)
    for period in range(4):
        occupants[period, ranks[period, included]] = np.flatnonzero(included)
    mobility = np.abs(np.diff(ranks[:, included], axis=0)).mean(axis=1)
    for mode, classes in [('full-rank', ranks), ('geographic-rank', occupants)]:
        counts = np.zeros((8, 8), dtype=np.int64)
        for period in range(3):
            for origin, destination in zip(classes[period], classes[period+1]):
                if origin < 8 and destination < 8:
                    counts[origin, destination] += 1
        active = counts.sum(axis=1) > 0
        matrix = counts[np.ix_(active, active)]
        probabilities = matrix / matrix.sum(axis=1, keepdims=True)
        assert connected_components(csr_matrix(matrix), directed=True, connection='strong')[0] == 1
        equations = probabilities.T - np.eye(population)
        equations[-1] = 1
        target = np.zeros(population)
        target[-1] = 1
        steady = np.zeros(8)
        steady[active] = np.linalg.solve(equations, target)
        entries = [[int(origin * 8 + destination), int(counts[origin, destination]), float(counts[origin, destination] / counts[origin].sum())] for origin, destination in zip(*np.nonzero(counts))]
        print(json.dumps(dict(mode=mode, tieOrder=tie_order, ranks=ranks.ravel().tolist(), entries=entries,
                              rowTotals=counts.sum(axis=1).tolist(), ignored=[6], mobility=mobility.tolist(),
                              steadyStates=steady.tolist(), stateSparsity=[6, 2], converged=[1])))
# Every value and tie ordering is unchanged by x -> 2x + 3. Exclusions remain exclusions.
