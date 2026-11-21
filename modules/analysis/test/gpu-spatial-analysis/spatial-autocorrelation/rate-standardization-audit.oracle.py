# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-autocorrelation/rate-standardization-audit.oracle.py

Assuncao/Reis (1999), esda assuncao_rate. Use exact rational arithmetic
for weighted population moments, then libm sqrt for the final standardized rate.
Unlike shrinkage EB smoothing, preserve negative a and replace v_i only when
a+b/n_i < 0. This occurs at LARGE populations (small b/n_i), not small ones.
Replicating event/population counts changes sampling noise; it is deliberately
not invariant, even though rates are unchanged. Row permutation is equivariant.
"""
from fractions import Fraction
import json
import math

POPULATIONS = [10, 20, 40, 80, 160, 320, 1, 0, 10, 20]
MASK = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
EVENTS = {
    'under': [1, 2, 5, 9, 15, 34, 1000, 2, -1, None],
    'over': [1, 20, 4, 40, 8, 150, 1000, 2, -1, None],
    'zero': [0, 0, 0, 0, 0, 0, 1000, 2, -1, None],
}

for name, events in EVENTS.items():
    for scale in [1, 2] if name == 'under' else [1]:
        rows = [row for row, event in enumerate(events) if MASK[row]
                and event is not None and event >= 0 and POPULATIONS[row] > 0]
        populations = [Fraction(scale * POPULATIONS[row]) for row in rows]
        counts = [Fraction(scale * events[row]) for row in rows]
        total_population = sum(populations)
        mean = sum(counts) / total_population
        rates = [count / population for count, population in zip(counts, populations)]
        variance = sum(population * (rate-mean)**2 for population, rate in zip(populations, rates)) / total_population
        prior = variance - mean * len(rows)/total_population
        standardized = [None] * 10
        negative_rows = []
        for row, population, rate in zip(rows, populations, rates):
            poisson = mean / population
            spread = prior + poisson
            if spread < 0:
                negative_rows.append(row)
                spread = poisson
            standardized[row] = float(rate-mean)/math.sqrt(float(spread)) if spread > 0 else None
        print(name, scale, json.dumps({'rates': standardized,
              'statistics': [len(rows), float(mean), float(variance), float(prior), float(total_population)],
              'invalidCount': 3, 'negativeRows': negative_rows}))
