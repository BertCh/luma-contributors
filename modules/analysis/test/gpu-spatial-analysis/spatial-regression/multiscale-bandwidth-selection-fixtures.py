# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent NumPy MGWR partial-response bandwidth backfit; /usr/bin/python3.

The scalar fit/hat matrices are explicit, unlike the GPU/CPU row loops. Golden transitions
follow PySAL mgwr/search.py (delta .38197, lower point wins ties). The score tolerance is
max(1e-6, 4e-6 * max(abs(scores))), the existing float32 GWR contract. OLS initialization and
maximum relative contribution convergence follow this contributor, rather than PySAL's
single-bandwidth GWR initialization/SOC stopping rule. Bandwidths are not rounded to .01.
The upper endpoint is also evaluated to retain a feasible fit when both interiors are invalid.
AICc uses mgwr/diagnostics.py's Gaussian formula. Component search uses tr(H_j) on the partial
response. Full inference propagates R_j = H_j (I - sum_{k != j} R_k), and tr(S) = sum tr(R_j).
These fixtures are a NumPy reference, not results from an installed mgwr package.
"""
import json
from pathlib import Path
import numpy as np


def golden(lower, upper, evaluate):
    delta = .38197
    first = lower + delta * (upper - lower)
    second = upper - delta * (upper - lower)
    first_score, second_score = evaluate(first), evaluate(second)
    endpoint, endpoint_score = upper, evaluate(upper)
    for iteration in range(12):
        first_wins = np.isfinite(first_score) and (not np.isfinite(second_score) or first_score <= second_score)
        best = first if first_wins else second if np.isfinite(second_score) else np.nan
        interior_score = first_score if first_wins else second_score
        if np.isfinite(endpoint_score) and (not np.isfinite(interior_score) or endpoint_score < interior_score):
            best = endpoint
        if ((not np.isfinite(first_score) and not np.isfinite(second_score) and not np.isfinite(endpoint_score)) or
            (np.isfinite(first_score) and np.isfinite(second_score) and
             abs(first_score - second_score) <= max(1e-6, 4e-6 * max(abs(first_score), abs(second_score)))) or iteration == 11):
            return best
        if first_wins:
            upper, second, second_score = second, first, first_score
            first = lower + delta * (upper - lower)
            first_score = evaluate(first)
        else:
            lower, first, first_score = first, second, second_score
            second = upper - delta * (upper - lower)
            second_score = evaluate(second)


def fixture(kernel, criterion, method, inference=False):
    count = 25
    row = np.arange(count)
    positions = np.column_stack([row * .35, np.sin(row * 1.4) * .5]).astype(np.float32).astype(float)
    predictors = np.column_stack([np.cos(row * 1.8) * (1 + row / 50), np.sin(row * 2.3)]).astype(np.float32).astype(float)
    response = (1 + .09 * row + (1 - .03 * row) * predictors[:, 0] - .4 * predictors[:, 1] + .24 * np.cos(row * .7)).astype(np.float32).astype(float)
    mask = np.ones(count, dtype=int)
    mask[[4, 24]] = 0
    # Excluded outliers must not change bandwidth scores or default index extent.
    positions[-1] = [1e8, -1e8]
    active = mask != 0
    design = np.column_stack([np.ones(count), predictors])[active]
    target = response[active]
    distance = np.linalg.norm(positions[active, None] - positions[None, active], axis=2)
    used_count, width = design.shape
    coefficients = np.zeros((used_count, width)) if inference else np.tile(np.linalg.lstsq(design, target, rcond=None)[0], (used_count, 1))
    operators = np.zeros((width, used_count, used_count))
    derivatives = np.zeros_like(operators)
    candidates = [[.8, 9], [.9, 8], [1, 10]] if method == 'golden-section' else [[.01, 1.8, 3.5, 7], [.01, 1.5, 3, 6], [.01, 2, 4, 8]]
    bandwidths = np.zeros(width)
    def smoother(column, bandwidth, leave_one_out=False):
        ratio = distance / bandwidth
        weights = np.maximum(1 - ratio ** 2, 0) ** 2 if kernel == 'bisquare' else np.exp(-.5 * ratio ** 2)
        if leave_one_out:
            np.fill_diagonal(weights, 0)
        denominator = weights @ (design[:, column] ** 2)
        if np.any(denominator <= 0):
            return None
        return weights * design[None, :, column] / denominator[:, None]
    for sweep in range(1000):
        previous = coefficients.copy()
        for column in range(width):
            partial = target - np.sum(design * coefficients, axis=1) + design[:, column] * coefficients[:, column]
            def evaluate(bandwidth):
                matrix = smoother(column, bandwidth, criterion == 'leave-one-out')
                if matrix is None:
                    return np.nan
                residual = partial - design[:, column] * (matrix @ partial)
                residual_squares = residual @ residual
                if criterion == 'leave-one-out':
                    return residual_squares
                trace = np.sum(design[:, column] * np.diag(matrix))
                degrees = used_count - trace - 2
                if degrees <= 0:
                    return np.nan
                return used_count * np.log(max(residual_squares / used_count, 1e-30)) + used_count * np.log(2 * np.pi) + used_count * (used_count + trace) / degrees
            if method == 'golden-section':
                bandwidths[column] = golden(*candidates[column], evaluate)
            else:
                scores = [evaluate(bandwidth) for bandwidth in candidates[column]]
                bandwidths[column] = candidates[column][int(np.argmin(np.where(np.isfinite(scores), scores, np.inf)))]
            matrix = smoother(column, bandwidths[column])
            coefficients[:, column] = matrix @ partial
            if inference:
                others = sum(operators[other] for other in range(width) if other != column)
                derivatives[column] = matrix @ (np.eye(used_count) - others)
                operators[column] = design[:, column, None] * derivatives[column]
        change = np.max(np.abs(design * (coefficients - previous)) / np.maximum(1, np.abs(design * previous)))
        if change <= 1e-3:
            break
    else:
        raise RuntimeError('fixture did not converge')
    def encode(values):
        return [float(value) if np.isfinite(value) else None for value in np.asarray(values).ravel()]
    full_coefficients = np.full((count, width), np.nan)
    full_coefficients[active] = coefficients
    fitted = np.full(count, np.nan)
    fitted[active] = np.sum(design * coefficients, axis=1)
    expected = {'coefficients': encode(full_coefficients), 'fitted': encode(fitted), 'selectedBandwidths': encode(bandwidths), 'iterationCount': [sweep + 1]}
    if inference:
        trace = sum(np.trace(operator) for operator in operators)
        total = sum(operators)
        trace_squared = np.sum(total ** 2)
        degrees = used_count - trace
        variance = np.sum((target - fitted[active]) ** 2) / degrees
        errors = np.full((count, width), np.nan)
        errors[active] = np.sqrt(np.sum(derivatives ** 2, axis=2).T * variance)
        expected['inferenceSummary'] = encode([trace, trace_squared, variance, degrees])
        expected['inferenceEffectiveParameters'] = encode([np.trace(operator) for operator in operators])
        expected['inferenceStandardErrors'] = encode(errors)
    return {'kernel': kernel, 'criterion': criterion, 'searchMethod': method, 'inference': inference, 'predictorCount': 2,
            'positions': encode(positions), 'predictors': encode(predictors), 'response': encode(response), 'mask': encode(mask),
            'parameters': [1e-3, 4, 4, 4], 'bandwidthCandidateCount': len(candidates[0]), 'bandwidthCandidates': encode(candidates), 'expected': expected}


fixtures = [fixture(kernel, criterion, method) for kernel in ['gaussian', 'bisquare'] for criterion in ['leave-one-out', 'AICc'] for method in ['finite-grid', 'golden-section']]
fixtures += [fixture(kernel, 'AICc', 'golden-section', True) for kernel in ['gaussian', 'bisquare']]
Path(__file__).with_suffix('.json').write_text(json.dumps(fixtures, indent=2) + '\n')
