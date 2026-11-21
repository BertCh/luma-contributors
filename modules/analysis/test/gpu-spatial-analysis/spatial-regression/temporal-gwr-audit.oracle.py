# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/temporal-gwr-audit.oracle.py

Independent small GTWR weighted QR and MGTWR block solve. The declared distance is
sqrt(ds^2 + tau*dt^2), Gaussian exp(-d^2/(2h^2)), bisquare max(1-d^2/h^2,0)^2.
Adaptive radii include self/ties and inflate the kth distance by 1.0000001 (mgtwr).
Balanced +/-1 predictors at both sites make the component cross operators vanish, so one
MGTWR sweep reaches the block solution. The first SOC compares that solution with the initial
GTWR fit, and the second is exactly zero. No backfit loop is used to obtain expectations.
"""
import numpy as np
from fractions import Fraction
from pathlib import Path
import re

positions = np.array([0] * 6 + [2] * 6, dtype=float)
times = np.array([0] * 6 + [3] * 6, dtype=float)
predictors = np.array([-1] * 3 + [1] * 3 + [-1] * 3 + [1] * 3, dtype=float)
design = np.column_stack([np.ones(12), predictors])
observed = np.array([0, 1, 2, 1, 3, 5, 2, 3, 4, 0, 2, 4], dtype=float)
print('R2: y=1000000+x belongs exactly to the full-rank intercept/predictor design: beta=[1000000,1], residual=0 at every included row.')


def weights(kernel, adaptive, bandwidth, scale):
    distance = np.sqrt((positions[:, None] - positions[None, :]) ** 2 + scale * (times[:, None] - times[None, :]) ** 2)
    radius = np.sort(distance, axis=1)[:, int(bandwidth) - 1] * 1.0000001 if adaptive else np.full(12, bandwidth)
    ratio = distance / radius[:, None]
    return np.exp(-.5 * ratio ** 2) if kernel == 'gaussian' else np.maximum(0, 1 - ratio ** 2) ** 2


def gtwr(kernel, adaptive, bandwidth, scale):
    spatial_weights = weights(kernel, adaptive, bandwidth, scale)
    coefficients = np.array([np.linalg.lstsq(np.sqrt(weight)[:, None] * design, np.sqrt(weight) * observed, rcond=None)[0] for weight in spatial_weights])
    fitted = (coefficients * design).sum(axis=1)
    hat = np.array([row @ np.linalg.inv(design.T @ np.diag(weight) @ design) @ row for weight, row in zip(spatial_weights, design)])
    residuals = observed - fitted
    squares, trace = residuals @ residuals, hat.sum()
    likelihood = -.5 * 12 * (np.log(squares / 12) + np.log(2 * np.pi) + 1)
    summary = [squares, trace, -2 * likelihood + 24 * (trace + 1) / (10 - trace), 1 - squares / sum((observed - observed.mean()) ** 2), 12, None, 0, squares / (12 - trace), None, None, likelihood, 12]
    return coefficients, fitted, hat, residuals, summary


for kernel in ['gaussian', 'bisquare']:
    for adaptive in [False, True]:
        coefficients, fitted, hat, residuals, summary = gtwr(kernel, adaptive, 8 if adaptive else 5, 1.25)
        print(kernel, 'adaptive' if adaptive else 'fixed')
        print('GTWR coefficients:', coefficients[[0, 6]].tolist())
        print('GTWR fitted:', fitted[[0, 3, 6, 9]].tolist())
        print('GTWR hat:', hat[[0, 6]].tolist())
        print('GTWR summary:', summary)
        initial = gtwr(kernel, adaptive, 10 if adaptive else 6, .5)[0]
        component_weights = [weights(kernel, adaptive, 8 if adaptive else 5, 1.25), weights(kernel, adaptive, 10 if adaptive else 6, 2)]
        scalar = [component_weights[0] / component_weights[0].sum(axis=1)[:, None], component_weights[1] * predictors[None, :] / (component_weights[1] @ predictors ** 2)[:, None]]
        system = np.block([[np.eye(12), scalar[0] @ np.diag(predictors)], [scalar[1], np.eye(12)]])
        coefficients = np.linalg.solve(system, np.concatenate([operator @ observed for operator in scalar])).reshape(2, 12).T
        fitted = (coefficients * design).sum(axis=1)
        score = np.sqrt(np.sum((design * (coefficients - initial)) ** 2) / 12 / np.sum(fitted ** 2))
        print('MGTWR coefficients:', coefficients[[0, 6]].tolist())
        print('MGTWR fitted:', fitted[[0, 3, 6, 9]].tolist())
        print('MGTWR scores:', [float(score), 0])

# Check the numerical documentation illustration separately from the model oracle values.
# Exact rational arithmetic gives 2134/8883, not the former documented 0.05.
source = (Path(__file__).parents[3] / 'src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression.ts').read_text()
documented_gap = float(re.search(r'; ([0-9.]+) at `n = 200`, `k = 10`', source).group(1))
exact_gap = 2 * 11 * (Fraction(200, 188) - Fraction(199, 189))
assert abs(documented_gap - float(exact_gap)) < .005
print('AICc documentation gap at n=200, k=10:', float(exact_gap))
