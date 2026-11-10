# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""External references for the STX-06 change point, Pettitt and corrected Mann-Kendall kernels.

ruptures 1.1.10 (BSD-2-Clause, Pelt with jump=1, costs l2, linear and normal), pyhomogeneity 1.1
(Pettitt U, location and analytic p) and pymannkendall 1.4.3 (Hamed-Rao and Yue-Wang modified
Mann-Kendall). ruptures' Cython kernel-CPD extension is not needed by Pelt and has no wheel for
this Python, so the extension modules are stubbed; only the pure-Python Pelt and cost classes run.
Every series is rounded to float32 first, the data the GPU kernels read. The R `changepoint`
package (GPL, not installed) is not used; its definitions are checked by the binary64 CPU
reference in the spec, and an external R fixture is pending.

Run: STX0607_ORACLE_SITE=<dir with pymannkendall, pyhomogeneity and ruptures/src>
<venv>/bin/python -I modules/analysis/test/oracles/stx06-change-points.py
"""
import json
import os
import platform
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'fixtures' / 'stx06-change-points.json'

for entry in os.environ['STX0607_ORACLE_SITE'].split(os.pathsep):
    sys.path.append(entry)


class UnavailableExtension(types.ModuleType):
    """Stub of a ruptures Cython extension; only Pelt and the pure-Python costs run here."""

    def __getattr__(self, name):
        def unavailable(*_arguments, **_keywords):
            raise RuntimeError(f'{name} needs a Cython extension that this oracle environment lacks')

        return unavailable


for module_name in [
    'ruptures.detection._detection',
    'ruptures.detection._detection.ekcpd',
    'ruptures.utils._utils',
    'ruptures.utils._utils.convert_path_matrix'
]:
    sys.modules[module_name] = UnavailableExtension(module_name)

import numpy as np  # noqa: E402
import pyhomogeneity  # noqa: E402
import pymannkendall  # noqa: E402
import ruptures  # noqa: E402
import scipy  # noqa: E402
from ruptures.costs import CostL2, CostLinear, CostNormal  # noqa: E402

RUPTURES_VERSION = '1.1.10'  # sdist version; __version__ is generated at build time.


def to_float32(values):
    return np.asarray(np.asarray(values, dtype=np.float64).astype(np.float32), dtype=np.float64)


def make_series(random, length):
    # Steps in the mean with unit-ish noise, a variance change, a slope change and counts.
    segments = np.concatenate([
        random.normal(0.0, 1.0, length // 4),
        random.normal(6.0, 1.0, length // 4),
        random.normal(-3.0, 1.0, length // 4),
        random.normal(2.0, 1.0, length - 3 * (length // 4))
    ])
    variance = np.concatenate([random.normal(0.0, 0.3, length // 2), random.normal(0.0, 3.0, length - length // 2)])
    positions = np.arange(length, dtype=np.float64)
    bend = length // 2
    slope = np.where(positions < bend, 0.8 * positions, 0.8 * bend - 1.5 * (positions - bend)) + random.normal(0.0, 0.5, length)
    counts = np.concatenate([random.poisson(4.0, length // 3), random.poisson(18.0, length // 3), random.poisson(7.0, length - 2 * (length // 3))]).astype(np.float64)
    return {
        'mean': to_float32(segments),
        'variance': to_float32(variance),
        'mean-variance': to_float32(variance + 2.0 * (positions > length // 3)),
        'slope': to_float32(slope),
        'poisson': to_float32(counts)
    }


def segment_cost(cost, series, start, end):
    """Minus twice the log-likelihood used by the spec's definitions, unpruned, independent of Pelt."""
    segment = series[start:end]
    n = end - start
    if cost == 'mean':
        return float(np.sum((segment - segment.mean()) ** 2))
    if cost == 'slope':
        design = np.stack([np.arange(start, end, dtype=np.float64), np.ones(n)], axis=1)
        fit = np.linalg.lstsq(design, segment, rcond=None)[0]
        residual = segment - design @ fit
        return float(np.sum(residual ** 2))
    if cost == 'mean-variance':
        variance = max(float(np.mean((segment - segment.mean()) ** 2)), 1e-300)
        return float(n * (np.log(2 * np.pi) + np.log(variance) + 1.0))
    raise ValueError(cost)


def optimal_partitioning(cost, series, penalty, minimum_size):
    length = len(series)
    best = np.full(length + 1, np.inf)
    previous = np.full(length + 1, -1)
    best[0] = -penalty
    for end in range(minimum_size, length + 1):
        for start in range(0, end - minimum_size + 1):
            if start and start < minimum_size or not np.isfinite(best[start]):
                continue
            candidate = best[start] + segment_cost(cost, series, start, end) + penalty
            if candidate < best[end]:
                best[end] = candidate
                previous[end] = start
    breakpoints = []
    cursor = length
    while previous[cursor] > 0:
        breakpoints.append(int(previous[cursor]))
        cursor = int(previous[cursor])
    return sorted(breakpoints), float(best[length])


def pelt_breakpoints(cost, series, penalty, minimum_size):
    if cost == 'mean':
        model = CostL2()
        signal = series
    elif cost == 'slope':
        model = CostLinear()
        signal = np.column_stack([series, np.arange(len(series), dtype=np.float64), np.ones(len(series))])
    elif cost == 'mean-variance':
        model = CostNormal(add_small_diag=False)
        signal = series
    detector = ruptures.Pelt(custom_cost=model, min_size=minimum_size, jump=1).fit(signal)
    return [int(value) for value in detector.predict(pen=penalty)[:-1]]


COST_PENALTIES = {'mean': (12.0, 2), 'slope': (8.0, 3), 'mean-variance': (10.0, 2)}

random = np.random.default_rng(20261109)
length = 48
locations = []
for location in range(4):
    series_by_cost = make_series(random, length)
    change_points = {}
    for cost, (penalty, minimum_size) in COST_PENALTIES.items():
        breakpoints = pelt_breakpoints(cost, series_by_cost[cost], penalty, minimum_size)
        # Self-check: Pelt equals the unpruned dynamic program this generator runs.
        expected, optimum = optimal_partitioning(cost, series_by_cost[cost], penalty, minimum_size)
        assert breakpoints == expected, (cost, location, breakpoints, expected)
        change_points[cost] = {'penalty': penalty, 'minimumSegmentLength': minimum_size, 'breakpoints': breakpoints, 'optimalCost': optimum}
    locations.append({'series': {cost: values.tolist() for cost, values in series_by_cost.items()}, 'changePoints': change_points})

# Mann-Kendall series: one trend free AR(1), one weak trend with strong AR(1), one plain trend.
trend_series = []
for coefficient, slope in [(0.8, 0.0), (0.75, 0.03), (0.0, 0.2)]:
    noise = random.normal(0.0, 1.0, 60)
    values = np.zeros(60)
    for index in range(1, 60):
        values[index] = coefficient * values[index - 1] + noise[index]
    trend_series.append(to_float32(values + slope * np.arange(60, dtype=np.float64)))
modified_tests = []
for values in trend_series:
    hamed_rao = pymannkendall.hamed_rao_modification_test(values, alpha=0.05)
    hamed_rao_lag = pymannkendall.hamed_rao_modification_test(values, alpha=0.05, lag=3)
    yue_wang = pymannkendall.yue_wang_modification_test(values, alpha=0.05)
    yue_wang_lag = pymannkendall.yue_wang_modification_test(values, alpha=0.05, lag=1)
    plain = pymannkendall.original_test(values, alpha=0.05)
    entry = {'series': values.tolist()}
    for name, result in [('original', plain), ('hamedRao', hamed_rao), ('hamedRaoLag3', hamed_rao_lag), ('yueWang', yue_wang), ('yueWangLag1', yue_wang_lag)]:
        entry[name] = {'s': float(result.s), 'varianceS': float(result.var_s), 'z': float(result.z), 'p': float(result.p)}
    modified_tests.append(entry)

pettitt = []
for location in locations:
    values = np.asarray(location['series']['mean'])
    result = pyhomogeneity.pettitt_test(values, alpha=0.05, sim=None)
    pettitt.append({'u': float(result.U), 'changeSlice': int(result.cp), 'p': float(result.p)})

OUTPUT.write_text(json.dumps({
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'ruptures': RUPTURES_VERSION,
        'pyhomogeneity': '1.1',
        'pymannkendall': '1.4.3'
    },
    'length': length,
    'locations': locations,
    'modifiedMannKendall': modified_tests,
    'pettitt': pettitt
}) + '\n')
print('wrote', OUTPUT)
