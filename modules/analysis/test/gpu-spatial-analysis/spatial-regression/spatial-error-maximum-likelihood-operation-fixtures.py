# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Regenerate with /usr/bin/python3; dense NumPy least squares and SciPy objective minimization.

The oracle filters the raw design/response afresh for each parameter, rather than forming a
rational profile derivative. Matrix powers supply traces and the finite inverse series.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm


def get_probe(seed, probe, row):
    words = [probe, row >> 2, 0, 0x50524F42]
    key = [seed, 0x54524143]
    mask = 0xFFFFFFFF
    for round_index in range(10):
        if round_index:
            key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
        first = 0xD2511F53 * words[0]
        second = 0xCD9E8D57 * words[2]
        words = [(second >> 32) ^ words[1] ^ key[0], second & mask,
                 (first >> 32) ^ words[3] ^ key[1], first & mask]
    return -1 if words[row & 3] & 1 else 1


def get_tail(traces):
    last, before, first = traces[-1], traces[-2], traces[-3]
    ratio = min((last + before) / (before + first), 1) if before + first > 1e-30 and last + before > 0 else 0
    alternating = max((last - before) / (before - first), -1) if abs(before - first) > 1e-30 and (last - before) * (before - first) < 0 else 0
    return [(0.5 * (last + before), ratio), (0.5 * (last - before) if alternating else 0, alternating)]


def get_fixture(name, matrix, predictors, response, maximum_power=32, stochastic=False, tail=False, exact=False):
    row_count, predictor_count = predictors.shape
    design = np.column_stack([np.ones(row_count), predictors])
    infinity_norm = np.max(np.sum(abs(matrix), axis=1))
    scaled_matrix = matrix / infinity_norm
    probes = np.array([[get_probe(42, probe, row) for row in range(row_count)] for probe in range(12)]) if stochastic else np.eye(row_count)
    powers = []
    current = np.eye(row_count)
    for _ in range(maximum_power):
        current = current @ scaled_matrix
        powers.append(current)
    draws = np.array([np.einsum('bi,ij,bj->b', probes, power, probes) for power in powers])
    traces = draws.mean(axis=1) if stochastic else draws.sum(axis=1)
    errors = draws.std(axis=1, ddof=1) / np.sqrt(len(probes)) if stochastic else np.zeros(maximum_power)
    eigenvalues = np.linalg.eigvals(matrix).real.astype(np.float32).astype(float) if exact else None
    tail_model = get_tail(traces) if tail and not exact else []
    limit = 0.99 if tail or exact else brentq(lambda value: value ** (maximum_power + 1) / ((maximum_power + 1) * (1 - value)) - 1e-5, 0, 0.99)
    bound = limit / infinity_norm

    def get_log_determinant(parameter):
        if exact:
            return np.log1p(-parameter * eigenvalues).sum()
        scaled = parameter * infinity_norm
        value = -sum(scaled ** (index + 1) * trace / (index + 1) for index, trace in enumerate(traces))
        for level, ratio in tail_model:
            term = scaled ** maximum_power
            for extra in range(1, 10000):
                term *= scaled * ratio
                increment = -level * term / (maximum_power + extra)
                updated = value + increment
                if updated == value:
                    break
                value = updated
        return value

    def fit(parameter):
        filtered = np.eye(row_count) - parameter * matrix
        filtered_design = filtered @ design
        filtered_response = filtered @ response
        coefficients = np.linalg.lstsq(filtered_design, filtered_response, rcond=None)[0]
        residual = filtered_response - filtered_design @ coefficients
        sigma_squared = residual @ residual / row_count
        objective = -0.5 * row_count * np.log(sigma_squared) + get_log_determinant(parameter)
        return objective, coefficients, sigma_squared, filtered_design

    grid = np.linspace(-bound, bound, 1025)
    objectives = np.array([fit(value)[0] for value in grid])
    candidates = []
    if objectives[0] >= objectives[1]:
        candidates.append((-bound, True))
    for index in range(1, len(grid) - 1):
        if objectives[index] >= objectives[index - 1] and objectives[index] >= objectives[index + 1]:
            optimized = minimize_scalar(lambda value: -fit(value)[0], bounds=(grid[index - 1], grid[index + 1]), method='bounded', options={'xatol': 1e-14})
            candidates.append((optimized.x, False))
    if objectives[-1] > objectives[-2]:
        candidates.append((bound, True))
    parameter, boundary = max(candidates, key=lambda value: fit(value[0])[0])
    objective, coefficients, sigma_squared, filtered_design = fit(parameter)
    residuals = response - design @ coefficients
    inverse = np.linalg.inv(filtered_design.T @ filtered_design)
    standard_errors = np.sqrt(sigma_squared * np.diag(inverse))
    scaled = parameter * infinity_norm
    if exact:
        trace_one = sum(eigenvalues / (1 - parameter * eigenvalues))
        trace_two = sum((eigenvalues / (1 - parameter * eigenvalues)) ** 2)
    else:
        trace_one = infinity_norm * sum(scaled ** index * trace for index, trace in enumerate(traces))
        trace_two = infinity_norm ** 2 * sum(index * scaled ** (index - 1) * traces[index] for index in range(1, maximum_power))
        for level, ratio in tail_model:
            geometric = scaled * ratio / (1 - scaled * ratio)
            trace_one += infinity_norm * level * scaled ** (maximum_power - 1) * geometric
            trace_two += infinity_norm ** 2 * level * scaled ** (maximum_power - 2) * ((maximum_power - 1) * geometric + geometric / (1 - scaled * ratio))
    inverse_series = np.eye(row_count)
    for index in range(1, maximum_power):
        inverse_series += scaled ** index * powers[index - 1]
    transformed = matrix @ inverse_series @ probes.T
    trace_three = np.sum(transformed ** 2) / (len(probes) if stochastic else 1)
    information = trace_two + trace_three - 2 * trace_one ** 2 / row_count
    standard_errors = np.append(standard_errors, np.sqrt(1 / information) if information > 0 else np.nan)
    coefficients = np.append(coefficients, parameter)
    statistics = coefficients / standard_errors
    table = np.column_stack([coefficients, standard_errors, statistics, 2 * norm.sf(abs(statistics))]).ravel()
    means = predictors.mean(axis=0)
    centered_design = design.copy()
    centered_design[:, 1:] -= means
    filtered_centered = (np.eye(row_count) - parameter * matrix) @ centered_design
    normalized = filtered_centered / np.linalg.norm(filtered_centered, axis=0)
    singular = np.linalg.svd(normalized, compute_uv=False)
    raw_singular = np.linalg.svd(filtered_design / np.linalg.norm(filtered_design, axis=0), compute_uv=False)
    conditioning = [raw_singular[0] / raw_singular[-1], (singular[0] / singular[-1]) ** 2,
                    max(np.diag(np.linalg.inv(normalized.T @ normalized))[1:])]
    log_likelihood = objective - 0.5 * row_count * (np.log(2 * np.pi) + 1)
    summary = [row_count, parameter, sigma_squared, log_likelihood,
               2 * (predictor_count + 1) - 2 * log_likelihood,
               (predictor_count + 1) * np.log(row_count) - 2 * log_likelihood,
               np.corrcoef(response, design @ coefficients[:-1])[0, 1] ** 2,
               np.nan, residuals @ residuals, *conditioning, float(conditioning[1] > 1e4),
               bound, float(stochastic), float(not boundary), float(boundary), float(limit < 0.99),
               0 if exact else row_count * abs(scaled) ** (maximum_power + 1) / ((maximum_power + 1) * (1 - abs(scaled))) + abs(get_log_determinant(parameter) + sum(scaled ** (index + 1) * trace / (index + 1) for index, trace in enumerate(traces))),
               float(information > 0), len(candidates), get_log_determinant(parameter)]
    offsets, neighbors, weights = [0], [], []
    for row in matrix:
        for column, weight in enumerate(row):
            if weight:
                neighbors.append(column)
                weights.append(weight)
        offsets.append(len(neighbors))
    return {'name': name, 'predictorCount': predictor_count,
            'options': {'maximumPower': maximum_power, 'traceMode': 'stochastic' if stochastic else 'exact',
                        'traceProbeCount': 12, 'tailExtrapolation': tail, 'logDeterminant': 'exact' if exact else 'series'},
            'weights': {'offsets': offsets, 'neighbors': neighbors, 'weights': weights},
            'predictors': predictors.ravel().tolist(), 'response': response.tolist(),
            'eigenvalues': eigenvalues.tolist() if exact else None,
            'expected': {'table': table.tolist(), 'summary': summary, 'residuals': residuals.tolist(),
                         'powerTraceStandardErrors': errors.tolist()}}


generator = np.random.default_rng(28)
row_count = 18
symmetric = np.zeros((row_count, row_count))
for row in range(row_count - 1):
    for distance in [1, 3]:
        neighbor = (row + distance) % (row_count - 1)
        symmetric[row, neighbor] = symmetric[neighbor, row] = generator.uniform(0.2, 1)
matrix = (symmetric / np.maximum(symmetric.sum(axis=1)[:, None], 1e-30)).astype(np.float32).astype(float)
predictors = generator.normal(size=(row_count, 2)).astype(np.float32).astype(float)
noise = generator.normal(scale=0.4, size=row_count)
response = (0.15 + predictors @ [0.4, -0.25] + np.linalg.solve(np.eye(row_count) - 0.55 * matrix, noise)).astype(np.float32).astype(float)
fixtures = [get_fixture('ordinary-series', matrix, predictors, response),
            get_fixture('series-boundary', matrix, predictors, response, maximum_power=3),
            get_fixture('tail', matrix, predictors, response, maximum_power=8, tail=True),
            get_fixture('stochastic-seed-42', matrix, predictors, response, stochastic=True),
            get_fixture('eigenvalues', matrix, predictors, response, exact=True)]
directed = matrix.copy()
directed[::2] *= -0.75
directed[1::3] *= 1.25
fixtures.append(get_fixture('signed-directed-series', directed, predictors, response))


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


Path(__file__).with_suffix('.json').write_text(json.dumps(clean(fixtures), indent=2) + '\n')
for fixture in fixtures:
    print(fixture['name'], 'lambda', fixture['expected']['summary'][1],
          'p-values', fixture['expected']['table'][3::4])
