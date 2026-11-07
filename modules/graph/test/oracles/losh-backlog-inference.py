# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent exact rational residual-label oracle; Python 3 standard library only.

Population mean of squared deviations from positive-weight local means, no self links.
Enumerate ordered distinct donors excluding the focus, retaining slot order. Inclusive
upper/lower tails; two-sided is twice the smaller inclusive tail, capped at one.
These are exhaustive null probabilities, not finite Monte Carlo plus-one p-values.
"""
import itertools
import json
from fractions import Fraction
from pathlib import Path

values = [0, 1, 3, 8, 15]
neighbors = [[1, 2], [0, 2], [1, 3], [2, 4], [0, 3]]
residuals = [(Fraction(value) - sum(Fraction(values[other]) for other in adjacent) / len(adjacent)) ** 2
             for value, adjacent in zip(values, neighbors)]
mean = sum(residuals) / len(values)
normalized = [residual / mean for residual in residuals]
statistics = []
tails = []
for focus, adjacent in enumerate(neighbors):
    observed = sum(normalized[other] for other in adjacent) / len(adjacent)
    simulated = [sum(normalized[other] for other in draw) / len(adjacent)
                 for draw in itertools.permutations([row for row in range(len(values)) if row != focus], len(adjacent))]
    greater = Fraction(sum(value >= observed for value in simulated), len(simulated))
    lesser = Fraction(sum(value <= observed for value in simulated), len(simulated))
    statistics.append(float(observed))
    tails.append({'greater': float(greater), 'lesser': float(lesser), 'two-sided': float(min(1, 2 * min(greater, lesser)))})
result = {'generator': 'losh-backlog-inference.py; Python standard-library fractions; exhaustive ordered donors',
          'values': values, 'neighbors': neighbors, 'normalizedResiduals': list(map(float, normalized)),
          'statistic': statistics, 'probabilities': tails}
Path(__file__).with_name('fixtures').joinpath('losh-backlog-inference.json').write_text(json.dumps(result, indent=2) + '\n')
