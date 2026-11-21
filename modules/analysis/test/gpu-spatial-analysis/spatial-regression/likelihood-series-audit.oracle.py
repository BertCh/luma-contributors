# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/likelihood-series-audit.oracle.py
Prints likelihood-series-audit-fixture.json. Dense NumPy powers, inverse, slogdet and eigvalsh.
The eight-row fixture has two bipartite components, self loops, repeated eigenvalues and an island.
"""

import json
import numpy as np


def probe_value(probe, row):
    # Random123 Philox4x32-10 specification, independent of production random helpers.
    words, key = [probe, row // 4, 0, 0x50524F42], [17, 0x54524143]
    for iteration in range(10):
        if iteration:
            key = [(key[0] + 0x9E3779B9) & 0xffffffff, (key[1] + 0xBB67AE85) & 0xffffffff]
        first, second = words[0] * 0xD2511F53, words[2] * 0xCD9E8D57
        words = [(second >> 32) ^ words[1] ^ key[0], second & 0xffffffff,
                 (first >> 32) ^ words[3] ^ key[1], first & 0xffffffff]
    return -1 if words[row % 4] & 1 else 1


matrix = np.zeros((8, 8))
matrix[0, 1] = matrix[1, 0] = 1
matrix[2, 3] = matrix[4, 3] = 1
matrix[3, 2] = matrix[3, 4] = .5
matrix[5, 5], matrix[6, 6] = .25, -.5
powers = [np.linalg.matrix_power(matrix, power) for power in range(1, 129)]
offsets, neighbors, weights = [0], [], []
for row in matrix:
    for column, weight in enumerate(row):
        if weight:
            neighbors.append(column)
            weights.append(weight)
    offsets.append(len(neighbors))
fixture = dict(weights=dict(offsets=offsets, neighbors=neighbors, weights=weights),
               eigenvalues=[-1, -1, -.5, 0, 0, .25, 1, 1],
               powerTraces=[np.trace(power) for power in powers], cases=[], traces={})
impact_matrix = np.eye(8) + sum(.3 ** power * powers[power - 1] for power in range(1, 9))
fixture['impacts'] = [value for coefficient in [2., -3.] for value in
                      [coefficient * np.trace(impact_matrix) / 8,
                       coefficient * (impact_matrix.sum() - np.trace(impact_matrix)) / 8,
                       coefficient * impact_matrix.sum() / 8]]
for parameter in [-.99, -.95, -.3, 0., .3, .95, .99]:
    inverse = np.linalg.inv(np.eye(8) - parameter * matrix)
    transformed = matrix @ inverse
    exact = np.linalg.slogdet(np.eye(8) - parameter * matrix)[1]
    for maximum_power in [3, 8, 32, 128]:
        traces = np.array(fixture['powerTraces'][:maximum_power])
        determinant_terms = np.array([-parameter ** power / power for power in range(1, maximum_power + 1)])
        trace_one = sum(parameter ** (power - 1) * traces[power - 1] for power in range(1, maximum_power + 1))
        trace_two = sum((power - 1) * parameter ** (power - 2) * traces[power - 1] for power in range(2, maximum_power + 1))
        fixture['cases'].append(dict(parameter=parameter, maximumPower=maximum_power, exact=exact,
                                     truncated=float(traces @ determinant_terms), traceOne=trace_one, traceTwo=trace_two,
                                     derivativeTail=np.trace(transformed) - trace_one,
                                     secondTail=np.trace(transformed @ transformed) - trace_two,
                                     direct=np.trace(inverse) / 8, total=inverse.sum() / 8))
probes = np.array([[probe_value(probe, row) for row in range(8)] for probe in range(9)])
for mode in ['exact', 'stochastic']:
    draws = np.array([np.einsum('bi,ij,bj->b', probes, power, probes) for power in powers[:8]])
    gram = matrix.T @ matrix
    gram_pair = np.einsum('bi,ij,bj->b', probes, gram @ matrix, probes).mean()
    gram_square = np.einsum('bi,ij,bj->b', probes, gram @ gram, probes).mean()
    fixture['traces'][mode] = dict(powerTraces=(np.trace(np.array(powers[:8]), axis1=1, axis2=2) if mode == 'exact' else draws.mean(axis=1)).tolist(),
                                   powerSums=[power.sum() for power in powers[:8]],
                                   powerTraceStandardErrors=([0.] * 8 if mode == 'exact' else (draws.std(axis=1, ddof=1) / 3).tolist()),
                                   gramTraces=[np.trace(gram), np.trace(gram @ matrix) if mode == 'exact' else gram_pair,
                                               np.trace(gram @ gram) if mode == 'exact' else gram_square,
                                               ((matrix ** 2).sum(axis=0) ** 2).sum()],
                                   infinityNorm=[1.], approximation=[int(mode == 'stochastic')])
print(json.dumps(fixture, indent=2, allow_nan=False))
