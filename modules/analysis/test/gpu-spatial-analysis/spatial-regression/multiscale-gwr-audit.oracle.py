# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/multiscale-gwr-audit.oracle.py

Solve simultaneous normal equations for all component surfaces and their response operators.
This is a block linear solve, independent of cyclic backfitting and its derivative propagation.
The fixture has tied positions, a balanced +/-1 predictor at each site and two masked rows.
Balance makes the component cross-operators zero, so both surfaces reach their fixed point in
one sweep; a second sweep certifies the documented contribution-change tolerance from either
OLS or zero initialization. Both variance conventions use the coupled response smoother.
"""
import numpy as np

# R6: both design columns are orthogonal to y, so beta=0, tr(S)=tr(S'S)=2.
overflow_response = np.array([1e20, 1e20, -1e20, -1e20] * 2, dtype=np.float32).astype(float)
print('R6 variance:', np.dot(overflow_response, overflow_response) / 6, 'published float32: Infinity; inference status: 2')

position = np.array([0] * 6 + [2] * 6, dtype=float)
predictor = np.array([-1] * 3 + [1] * 3 + [-1] * 3 + [1] * 3, dtype=float)
response = np.array([0, 1, 2, 1, 3, 5, 2, 3, 4, 0, 2, 4], dtype=float)
distance = abs(position[:, None] - position[None, :])
identity = np.eye(12)
diagonal = np.diag(predictor)
for kernel in ['gaussian', 'bisquare']:
    weights = [np.exp(-.5 * (distance / bandwidth) ** 2) if kernel == 'gaussian'
               else np.maximum(0, 1 - (distance / bandwidth) ** 2) ** 2
               for bandwidth in [2.5, 3]]
    scalar = [weights[0] / weights[0].sum(axis=1)[:, None],
              weights[1] @ diagonal / (weights[1] @ predictor ** 2)[:, None]]
    system = np.block([[identity, scalar[0] @ diagonal], [scalar[1], identity]])
    response_operators = np.linalg.solve(system, np.vstack(scalar))
    coefficients = (response_operators @ response).reshape(2, 12).T
    smoother = response_operators[:12] + diagonal @ response_operators[12:]
    fitted = smoother @ response
    trace = np.trace(smoother)
    trace_squared = np.sum(smoother ** 2)
    residual_squares = sum((response - fitted) ** 2)
    conditional_errors = np.column_stack([np.sqrt(2 * np.sum(operator ** 2, axis=1)) for operator in scalar])
    effective = [np.trace(response_operators[:12]), np.trace(diagonal @ response_operators[12:])]
    print(kernel)
    print('coefficients:', coefficients[[0, 6]].tolist())
    print('fitted:', fitted[[0, 3, 6, 9]].tolist())
    print('conditionalStandardErrors:', conditional_errors[[0, 6]].tolist())
    print('effectiveParameters:', effective)
    for convention in ['residual-df', 'smoother-df']:
        degrees = 12 - trace if convention == 'residual-df' else 12 - 2 * trace + trace_squared
        variance = residual_squares / degrees
        standard_errors = np.sqrt(variance * np.sum(response_operators ** 2, axis=1)).reshape(2, 12).T
        print(convention, 'summary:', [trace, trace_squared, variance, degrees])
        print(convention, 'standardErrors:', standard_errors[[0, 6]].tolist())
