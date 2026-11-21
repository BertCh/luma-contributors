# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/local-variability-audit.oracle.py

Independent exact Fraction definitions. LOSH: r_i=|x_i-neighborMean_i|**power,
H_i=neighborMean_i(r)/mean(r) over finite non-island residuals. Self/zero weights
are removed, masked/missing rows removed before local means, undefined residuals
removed before final neighbor means (a documented extension of complete-data LOSH).
Multivariate Geary: average across variables of weighted neighbor squared
differences divided by each variable's population variance (joint complete
cases, islands retained for moments). This library averages, rather than sums,
the variable contributions; the convention is explicit in its public class.
"""
import importlib.util
from pathlib import Path
from fractions import Fraction
import json

specification = importlib.util.spec_from_file_location('fixture', Path(__file__).with_name('local-autocorrelation-audit.oracle.py'))
fixture = importlib.util.module_from_spec(specification)
specification.loader.exec_module(fixture)
values = [Fraction(value) if row != 9 else None for row, value in enumerate(fixture.VALUES)]
active = [row for row in range(10) if fixture.MASK[row] and values[row] is not None]


def edges(row, population):
    return [(neighbor, Fraction(weight)) for neighbor, weight in fixture.EDGES[row]
            if row != neighbor and weight > 0 and neighbor in population]


def neighbor_mean(row, field):
    neighbors = edges(row, field)
    if not neighbors:
        return None
    return sum(weight*field[neighbor] for neighbor, weight in neighbors) / sum(weight for _, weight in neighbors)


for power in [1, 2]:
    residuals = {}
    for row in active:
        mean = neighbor_mean(row, dict(zip(active, [values[index] for index in active])))
        if mean is not None:
            residuals[row] = abs(values[row]-mean)**power
    average = sum(residuals.values())/len(residuals)
    normalized = {row: value/average for row, value in residuals.items()}
    statistics = [neighbor_mean(row, normalized) if row in normalized else None for row in range(10)]
    print('LOSH power', power, json.dumps({'statistic': [float(value) if value is not None else None for value in statistics],
          'normalizedResiduals': [float(normalized[row]) if row in normalized else None for row in range(10)],
          'status': [0 if value is not None else 1 for value in statistics], 'residualCount': len(residuals)}))

for constant in [Fraction(4), Fraction(1, 10), Fraction(0), Fraction(2**20)]:
    constant_field = dict.fromkeys(active, constant)
    for power in [1, 2]:
        constant_residuals = {row: abs(constant-neighbor_mean(row, constant_field))**power
                              for row in active if edges(row, constant_field)}
        assert len(constant_residuals) == 7
        assert set(constant_residuals.values()) == {Fraction(0)}
        # mean(r)=0: normalization and H are undefined, but zero residuals are finite and counted.
print('constant LOSH', json.dumps({'statistic': [None]*10, 'normalizedResiduals': [None]*10,
      'status': [2]*7+[1]*3, 'residualCount': 7}))

second = [2, 5, None, 7, 4, 3, 11, 999, 9, None]
joint = [row for row in active if second[row] is not None]
columns = [values, [Fraction(value) if value is not None else None for value in second]]
variances = []
for column in columns:
    mean = sum(column[row] for row in joint)/len(joint)
    variances.append(sum((column[row]-mean)**2 for row in joint)/len(joint))
statistics = [None]*10
statuses = [1]*10
for row in joint:
    neighbors = edges(row, joint)
    statuses[row] = 2
    if neighbors:
        total = sum(weight for _, weight in neighbors)
        statistics[row] = sum(weight * sum((column[row]-column[neighbor])**2/variance for column, variance in zip(columns, variances))
                              for neighbor, weight in neighbors)/total/len(columns)
        statuses[row] = 0
print('multivariate Geary', json.dumps({'statistic': [float(value) if value is not None else None for value in statistics],
      'status': statuses, 'includedCount': len(joint), 'variances': list(map(float, variances))}))
