# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 <this file>.

Marshall global/spatial EB and Assuncao-Reis standardization, directly as NumPy
weighted moments, not a TypeScript transcription. Zero population is invalid;
row 7 is masked, row 6 is an island, and event counts include ties at zero.
Population-unit equivariance: doubling populations divides rates by two and
prior variances by four, preserving standardized rates and excess risks.
"""
import json
import numpy as np

EVENTS = np.array([0, 20, 0, 40, 5, 0, 0, 10], dtype=float)
POPULATIONS = np.array([10, 10, 0, 20, 10, 10, 10, 10], dtype=float)
MASK = np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
NEIGHBORS = [[1, 2], [0, 3], [1, 4], [2, 5], [0, 5], [0, 4], [], [0]]
INCLUDED = MASK & (POPULATIONS > 0)


def moments(indices, populations):
    events, population = EVENTS[indices], populations[indices]
    mean = events.sum() / population.sum()
    variance = np.average((events / population - mean) ** 2, weights=population)
    prior = variance - mean / population.mean()
    return mean, variance, prior


def global_result(populations):
    indices = np.flatnonzero(INCLUDED)
    mean, variance, prior = moments(indices, populations)
    raw = np.full(8, np.nan)
    standardized, smoothed, excess = raw.copy(), raw.copy(), raw.copy()
    raw[indices] = EVENTS[indices] / populations[indices]
    noise = mean / populations[indices]
    standardized[indices] = (raw[indices] - mean) / np.sqrt(np.where(prior + noise < 0, noise, prior + noise))
    clamped = max(0, prior)
    weight = clamped / (clamped + noise)
    smoothed[indices] = mean + weight * (raw[indices] - mean)
    excess[indices] = raw[indices] / mean
    return {'raw': raw, 'standardized': standardized, 'smoothed': smoothed, 'excess': excess,
            'summary': [6, 65, populations[indices].sum(), mean, variance, prior, clamped, 0]}


def spatial_result(populations):
    result = {name: np.full(8, np.nan) for name in ['spatialRate', 'smoothed', 'priorMean', 'priorVariance']}
    for row in np.flatnonzero(INCLUDED):
        indices = [row] + [neighbor for neighbor in NEIGHBORS[row] if INCLUDED[neighbor]]
        mean, _, prior = moments(indices, populations)
        prior = max(0, prior)
        denominator = prior + mean / populations[row]
        weight = prior / denominator if denominator > 0 else 0
        result['spatialRate'][row] = result['priorMean'][row] = mean
        result['priorVariance'][row] = prior
        result['smoothed'][row] = mean + weight * (EVENTS[row] / populations[row] - mean)
    return result


def encode(value):
    if isinstance(value, dict):
        return {name: encode(entry) for name, entry in value.items()}
    if isinstance(value, (np.ndarray, list)):
        return [encode(entry) for entry in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


global_expected, spatial_expected = global_result(POPULATIONS), spatial_result(POPULATIONS)
global_scaled, spatial_scaled = global_result(2 * POPULATIONS), spatial_result(2 * POPULATIONS)
for name in ['raw', 'smoothed']:
    np.testing.assert_allclose(global_scaled[name] * 2, global_expected[name], equal_nan=True)
for name in ['standardized', 'excess']:
    np.testing.assert_allclose(global_scaled[name], global_expected[name], equal_nan=True)
for name in spatial_expected:
    factor = 4 if name == 'priorVariance' else 2
    np.testing.assert_allclose(factor * spatial_scaled[name], spatial_expected[name], equal_nan=True)
print(json.dumps(encode({'global': global_expected, 'spatial': spatial_expected}), indent=2))
