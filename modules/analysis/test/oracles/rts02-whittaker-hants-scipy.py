# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""RTS-02 reference fixtures: binary64 SciPy/NumPy implementations of the weighted Whittaker smoother
(SciPy solve_banded on the banded normal equations) and HANTS (reject-and-refit least squares).

phenofit and GRASS r.hants are the published definitions. R and GRASS are not available here, so
these binary64 implementations are written directly from the documented algorithms; no R or GPL
code was read or copied. Output: fixtures/rts02-whittaker-hants-scipy.json.
"""
import json
import pathlib
import platform

import numpy as np
import scipy
from scipy.linalg import solve_banded

OUTPUT = pathlib.Path(__file__).resolve().parent / 'fixtures' / 'rts02-whittaker-hants-scipy.json'
SECONDS_PER_YEAR = 31556952.0


def difference_operator(order, years):
    """Scaled divided-difference matrix: reduces to ordinary differences on a regular axis."""
    n = years.size
    spacing = (years[-1] - years[0]) / (n - 1)
    x = years / spacing
    matrix = np.eye(n)
    current = n
    for level in range(1, order + 1):
        next_matrix = np.zeros((current - 1, n))
        for row in range(current - 1):
            span = (x[row + level] - x[row]) / level
            next_matrix[row] = (matrix[row + 1] - matrix[row]) / span
        matrix = next_matrix
        current -= 1
    return matrix


def regular_difference_operator(order, n):
    matrix = np.eye(n)
    for _ in range(order):
        matrix = np.diff(matrix, axis=0)
    return matrix


def solve_whittaker(y, base_weight, smoothing, operator, order, passes, function, minimum_weight):
    n = y.size
    robust = np.ones(n)
    z = np.zeros(n)
    penalty = operator.T @ operator
    for iteration in range(passes + 1):
        weight = base_weight * robust
        system = np.diag(weight) + smoothing * penalty
        banded = np.zeros((2 * order + 1, n))
        for row in range(n):
            for column in range(max(0, row - order), min(n, row + order + 1)):
                banded[order + row - column, column] = system[row, column]
        rhs = weight * np.nan_to_num(y)
        z = solve_banded((order, order), banded, rhs)
        if iteration == passes:
            break
        mask = base_weight > 0
        residual = y - z
        active_residuals = np.abs(residual[mask])
        scale = 6.0 * np.median(active_residuals)
        next_robust = np.ones(n)
        if function == 'bisquare':
            if scale > 0:
                ratio = np.abs(residual) / scale
                next_robust = np.where(ratio < 1, (1 - ratio * ratio) ** 2, minimum_weight)
        else:
            largest_negative = np.max(np.maximum(-residual[mask], 0.0)) if np.any(mask) else 0.0
            if largest_negative > 0:
                next_robust = np.where(residual < 0, 1 - np.abs(residual) / largest_negative, 1.0)
        robust = np.maximum(next_robust, minimum_weight)
        robust[~mask] = 1.0
    return z


def hants(y, active, years_or_slots, period, harmonic_count, tolerance, overdetermination, passes, side):
    columns = 2 * harmonic_count + 1
    minimum = columns + overdetermination
    active = active.copy()

    def design(times):
        rows = [np.ones(times.size)]
        for harmonic in range(1, harmonic_count + 1):
            angle = 2 * np.pi * harmonic * times / period
            rows += [np.cos(angle), np.sin(angle)]
        return np.column_stack(rows)

    matrix = design(years_or_slots)
    rejected = 0
    passes_used = 0
    finished = False
    coefficients = np.zeros(columns)
    for iteration in range(passes + 1):
        selection = np.where(active)[0]
        coefficients, *_ = np.linalg.lstsq(matrix[selection], y[selection], rcond=None)
        passes_used = iteration + 1
        residual = y - matrix @ coefficients
        magnitude = {'both': np.abs(residual), 'low': -residual, 'high': residual}[side]
        magnitude = np.where(active, magnitude, -np.inf)
        worst = int(np.argmax(magnitude))
        if magnitude[worst] <= tolerance or active.sum() <= minimum:
            finished = True
            break
        if iteration == passes:
            break
        active[worst] = False
        rejected += 1
    return matrix @ coefficients, coefficients, rejected, passes_used, finished


def main():
    random = np.random.default_rng(20261109)
    slice_count = 64
    regular_y = 0.4 + 0.3 * np.sin(2 * np.pi * np.arange(slice_count) / 22.0) + random.normal(0, 0.02, slice_count)
    regular_y_gapped = regular_y.copy()
    regular_y_gapped[[3, 4, 5, 20, 40, 41]] = np.nan
    regular_weight = np.where(np.isnan(regular_y_gapped), 0.0, 1.0)
    days = np.cumsum(random.integers(5, 21, slice_count)).astype(float)
    days -= days[0]
    irregular_times = 365.25 * 86400 * 0 + days * 86400
    seconds = irregular_times
    midpoint = seconds[0] + (seconds[-1] - seconds[0]) // 2
    years = (seconds - midpoint) / SECONDS_PER_YEAR
    irregular_y = 0.5 + 0.25 * np.sin(2 * np.pi * years) + random.normal(0, 0.015, slice_count)
    irregular_y[[10, 11, 30]] = np.nan
    irregular_weight = np.where(np.isnan(irregular_y), 0.0, 1.0)
    clean = np.nan_to_num(irregular_y)
    contaminated = irregular_y.copy()
    contaminated[[7, 25, 50]] -= 0.35

    cases = []
    def add_case(name, y, weight, smoothing, order, times_seconds, passes, function):
        if times_seconds is None:
            operator = regular_difference_operator(order, y.size)
            axis = None
        else:
            operator = difference_operator(order, (times_seconds - times_seconds[0]) / SECONDS_PER_YEAR)
            axis = [int(value) for value in times_seconds]
        smoothed = solve_whittaker(y, weight, smoothing, operator, order, passes, function, 0.01)
        cases.append({
            'name': name, 'values': [None if not np.isfinite(v) else float(v) for v in y],
            'smoothing': smoothing, 'order': order, 'passes': passes, 'function': function,
            'timesSeconds': axis,
            'expected': [float(v) for v in smoothed]
        })

    add_case('regular-gaps-order2-lambda50', regular_y_gapped, regular_weight, 50.0, 2, None, 0, 'bisquare')
    add_case('regular-gaps-order1-lambda200', regular_y_gapped, regular_weight, 200.0, 1, None, 0, 'bisquare')
    add_case('regular-order3-lambda20', regular_y, np.ones(slice_count), 20.0, 3, None, 0, 'bisquare')
    add_case('irregular-order2-lambda50', irregular_y, irregular_weight, 50.0, 2, seconds, 0, 'bisquare')
    add_case('irregular-order2-large-lambda', irregular_y, irregular_weight, 5000.0, 2, seconds, 0, 'bisquare')
    add_case('irregular-robust-bisquare', contaminated, irregular_weight, 50.0, 2, seconds, 3, 'bisquare')
    add_case('irregular-robust-upper-envelope', contaminated, irregular_weight, 50.0, 2, seconds, 4, 'upper-envelope')

    harmonic_times = seconds
    hants_y = 0.5 + 0.3 * np.sin(2 * np.pi * years) + 0.08 * np.cos(4 * np.pi * years) + random.normal(0, 0.01, slice_count)
    hants_y[[6, 22, 41, 42]] -= 0.45
    active = np.ones(slice_count, dtype=bool)
    active[[15]] = False
    hants_y_input = hants_y.copy()
    hants_y_input[15] = np.nan
    smoothed, coefficients, rejected, passes_used, finished = hants(hants_y, active, years, 1.0, 2, 0.05, 2, 8, 'low')
    hants_case = {
        'name': 'irregular-hants-low-rejection',
        'values': [None if not np.isfinite(v) else float(v) for v in hants_y_input],
        'timesSeconds': [int(v) for v in seconds],
        'period': 1.0, 'harmonicCount': 2, 'tolerance': 0.05, 'overdetermination': 2,
        'passes': 8, 'side': 'low', 'expectedSmoothed': [float(v) for v in smoothed],
        'expectedCoefficients': [float(v) for v in coefficients],
        'expectedRejected': rejected, 'expectedPasses': passes_used, 'expectedConverged': finished
    }
    limited_smoothed, limited_coefficients, limited_rejected, limited_passes, limited_finished = hants(hants_y, active, years, 1.0, 2, 0.05, 2, 1, 'low')
    hants_limited = {
        'passes': 1, 'expectedSmoothed': [float(v) for v in limited_smoothed],
        'expectedCoefficients': [float(v) for v in limited_coefficients],
        'expectedRejected': limited_rejected, 'expectedPasses': limited_passes, 'expectedConverged': limited_finished
    }

    fixture = {
        'source': 'binary64 reference written from the phenofit smooth_wWHIT and GRASS r.hants definitions; R and GRASS were not run',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'sliceCount': slice_count,
        'whittaker': cases,
        'hants': hants_case,
        'hantsLimited': hants_limited,
        'axisMidpointSeconds': int(midpoint),
    }
    OUTPUT.write_text(json.dumps(fixture))
    print('wrote', OUTPUT, OUTPUT.stat().st_size)


if __name__ == '__main__':
    main()
