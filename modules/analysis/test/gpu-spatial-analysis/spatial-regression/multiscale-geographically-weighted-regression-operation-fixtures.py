# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Solve the simultaneous fixed-point equations directly with NumPy, without backfit sweeps.

Run with /usr/bin/python3. Each scalar smoother is explicit; a single block linear solve gives
all coefficient surfaces. Also records known-variance conditional errors independently.
"""
import json
from pathlib import Path

import numpy as np


def create_fixture(kernel):
    count = 12
    rows = np.arange(count)
    positions = np.column_stack([rows * 0.5, (rows % 3) * 0.125]).astype(np.float32).astype(float)
    predictors = np.column_stack([np.where(rows % 2, -1, 1) * (1 + rows / 30), np.cos(rows * 1.7)]).astype(np.float32).astype(float)
    response = (1 + 0.05 * rows + predictors[:, 0] * (0.7 - 0.03 * rows) - 0.4 * predictors[:, 1] + 0.1 * np.sin(rows)).astype(np.float32).astype(float)
    mask = np.ones(count, dtype=int)
    mask[4] = 0
    active = mask != 0
    design = np.column_stack([np.ones(count), predictors])[active]
    distance = np.linalg.norm(positions[active, None, :] - positions[None, active, :], axis=2)
    bandwidths = [3.5, 2, 2.75]
    smoothers = []
    conditional_errors = []
    for column, bandwidth in enumerate(bandwidths):
        ratio = distance / bandwidth
        weights = np.maximum(1 - ratio ** 2, 0) ** 2 if kernel == 'bisquare' else np.exp(-0.5 * ratio ** 2)
        denominator = weights @ (design[:, column] ** 2)
        smoother = weights * design[None, :, column] / denominator[:, None]
        smoothers.append(smoother)
        conditional_errors.append(np.sqrt(0.25 * np.sum(smoother ** 2, axis=1)))
    active_count = active.sum()
    blocks = [[np.eye(active_count) if first == second else smoothers[first] * design[None, :, second]
               for second in range(3)] for first in range(3)]
    right = np.concatenate([smoother @ response[active] for smoother in smoothers])
    surfaces = np.linalg.solve(np.block(blocks), right).reshape(3, active_count).T
    coefficients = np.full((count, 3), np.nan)
    coefficients[active] = surfaces
    fitted = np.full(count, np.nan)
    fitted[active] = np.sum(surfaces * design, axis=1)
    errors = np.full((count, 3), np.nan)
    errors[active] = np.array(conditional_errors).T
    derivatives = np.linalg.solve(np.block(blocks), np.concatenate(smoothers, axis=0)).reshape(3, active_count, active_count)
    response_smoother = sum(design[:, column, None] * derivatives[column] for column in range(3))
    trace = np.trace(response_smoother)
    trace_squared = np.sum(response_smoother ** 2)
    full_inference = {}
    for convention in ['residual-df', 'smoother-df']:
        degrees = active_count - (trace if convention == 'residual-df' else 2 * trace - trace_squared)
        variance = np.sum((response[active] - fitted[active]) ** 2) / degrees
        standard_errors = np.full((count, 3), np.nan)
        standard_errors[active] = np.sqrt(variance * np.sum(derivatives ** 2, axis=2)).T
        full_inference[convention] = {
            'standardErrors': standard_errors,
            'effectiveParameters': np.array([np.sum(design[:, column] * np.diag(derivatives[column])) for column in range(3)]),
            'summary': np.array([trace, trace_squared, variance, degrees])
        }
    def encode(values):
        return [float(value) if np.isfinite(value) else None for value in values.ravel()]
    return {'kernel': kernel, 'predictorCount': 2, 'positions': positions.ravel().tolist(),
            'predictors': predictors.ravel().tolist(), 'response': response.tolist(), 'mask': mask.tolist(),
            'parameters': [1e-10] + bandwidths, 'noiseVariance': [0.25],
            'expected': {'coefficients': encode(coefficients), 'fitted': encode(fitted), 'conditionalStandardErrors': encode(errors),
                         'inference': {convention: {port: encode(values) for port, values in result.items()} for convention, result in full_inference.items()}}}


Path(__file__).with_suffix('.json').write_text(json.dumps([create_fixture(kernel) for kernel in ['bisquare', 'gaussian']], indent=2) + '\n')
