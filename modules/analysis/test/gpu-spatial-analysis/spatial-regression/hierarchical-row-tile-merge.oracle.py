# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Regenerate literals: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/hierarchical-row-tile-merge.oracle.py
"""Independent NumPy float32 cascade from the documented binary-counter order, and
float64 NumPy/SciPy regression estimates. No implementation code is imported.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def cascade(values):
    blocks = []
    for index, value in enumerate(values):
        value = np.float32(value)
        taken = index
        while taken & 1:
            value = np.float32(blocks.pop() + value)
            taken >>= 1
        blocks.append(value)
    total = np.float32(0)
    for block in reversed(blocks):
        total = np.float32(block + total)
    return total


for count in (4096, 4097, 8192, 8193):
    values = np.array([2**20 if index == 0 else -2**20 if index == 4096
                       else 1 if index == 8192 else .125 if index % 3 == 0 else .25
                       for index in range(count)], dtype=np.float32)
    groups = [cascade(values[start:start + 4096]) for start in range(0, count, 4096)]
    result = cascade(groups)
    print('merge', count, float(result), 'bits', int(result.view(np.uint32)),
          'float64', float(values.astype(np.float64).sum()))
    if count == 8193:
        print('scaled channels', [float(cascade([cascade((values * factor)[start:start + 4096])
              for start in range(0, count, 4096)])) for factor in range(1, 7)])

rows = 4097
indices = np.arange(rows)
predictor = ((indices * 17) % 101 - 50) / 16
lag_predictor = np.roll(predictor, 1)
response = (1.25 + .75 * predictor + .2 * lag_predictor +
            ((indices * 29) % 67 - 33) / 64).astype(np.float32).astype(np.float64)
design = np.column_stack((np.ones(rows), predictor))
lag_design = np.roll(design, 1, axis=0)
lag_response = np.roll(response, 1)
instruments = np.column_stack((design, lag_predictor))
structural = np.column_stack((design, lag_response))
projected = instruments @ np.linalg.lstsq(instruments, structural, rcond=None)[0]
two_stage = np.linalg.solve(projected.T @ structural, projected.T @ response)
lag_ml = np.linalg.lstsq(structural, response, rcond=None)[0]
# A directed cycle has trace(W^q)=0 for 1 <= q < n; hence the documented
# finite trace-series log determinant is exactly zero at every searched parameter.
def filtered_fit(parameter):
    filtered_design = design - parameter * lag_design
    filtered_response = response - parameter * lag_response
    beta = np.linalg.lstsq(filtered_design, filtered_response, rcond=None)[0]
    residual = filtered_response - filtered_design @ beta
    return beta, residual @ residual


error_ml = minimize_scalar(lambda parameter: filtered_fit(parameter)[1],
                           bounds=(-.99, .99), method='bounded',
                           options={'xatol': 1e-13}).x
ordinary = np.linalg.lstsq(design, response, rcond=None)[0]
residual = response - design @ ordinary
lag_residual = np.roll(residual, 1)
twice_lag_residual = np.roll(residual, 2)
squares = residual @ residual
cross = residual @ lag_residual
lag_squares = lag_residual @ lag_residual
twice_cross = residual @ twice_lag_residual
lag_twice_cross = lag_residual @ twice_lag_residual
twice_squares = twice_lag_residual @ twice_lag_residual
# Kelejian-Prucha moment equations, eliminating the nonnegative nuisance variance
# by least squares at each parameter. W is a permutation, so tr(W'W)=n.
matrix = np.array([[2 * cross, -lag_squares, rows],
                   [2 * lag_twice_cross, -twice_squares, rows],
                   [twice_cross + lag_squares, -lag_twice_cross, 0]]) / rows
target = np.array([squares, lag_squares, cross]) / rows
def objective(parameter):
    free = target - matrix[:, 0] * parameter - matrix[:, 1] * parameter**2
    variance = max(0, free @ matrix[:, 2] / (matrix[:, 2] @ matrix[:, 2]))
    difference = free - variance * matrix[:, 2]
    return difference @ difference


grid = np.linspace(-.99, .99, 20001)
best = int(np.argmin([objective(parameter) for parameter in grid]))
gm = minimize_scalar(objective, bounds=(grid[max(0, best - 1)], grid[min(20000, best + 1)]),
                     method='bounded', options={'xatol': 1e-13}).x
print('2sls', two_stage.tolist())
print('lag-ml', lag_ml.tolist())
print('error-ml', filtered_fit(error_ml)[0].tolist() + [error_ml])
print('gm-kp', filtered_fit(gm)[0].tolist() + [gm])
print('sigma2', '2sls', float(np.mean((response - structural @ two_stage)**2)),
      'lag-ml', float(np.mean((response - structural @ lag_ml)**2)),
      'error-ml', filtered_fit(error_ml)[1] / rows,
      'gm-kp', filtered_fit(gm)[1] / rows)

# Nonstationarity observed estimates: supplied local coefficients form a finite grid,
# checked independently with NumPy's population standard deviation. The companion spec
# asserts the observed spread directly; it does not use a replayed permutation as an oracle.
coefficients = np.column_stack((indices % 17 / 8, indices % 13 / 4))
print('observed-spread', np.std(coefficients, axis=0).tolist())
print('error-ml parameter standard error', float(np.sqrt((1-error_ml**2) / rows)))
four_groups = np.zeros(12289, dtype=np.float32)
four_groups[0], four_groups[8192], four_groups[12288] = 2**25, -2**25, 1
totals = [cascade(four_groups[start:start+4096]) for start in range(0,12289,4096)]
serial = np.float32(0)
for value in totals:
    serial = np.float32(serial + value)
print('four groups cascade', float(cascade(totals)), 'serial', float(serial),
      'float64', float(four_groups.astype(np.float64).sum()))
