# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent NumPy normal-equation GTWR oracle and exhaustive bandwidth/time-scale grids.
Run with /usr/bin/python3 from the repository root. No optimizer or production code is used.
"""
import json
from pathlib import Path
import numpy as np

random = np.random.default_rng(604)
rows = 36
positions = random.uniform(0, 5, (rows, 2)).astype(np.float32)
times = random.uniform(0, 4, rows).astype(np.float32)
positions[1] = positions[0]
times[1] = times[0]  # zero-distance tie, counted separately
predictors = random.normal(size=rows).astype(np.float32)
response = (1 + .2 * positions[:, 0] + .5 * times +
            (1 + np.sin(positions[:, 1]) + .4 * times) * predictors +
            random.normal(0, .3, rows)).astype(np.float32)
mask = np.ones(rows, dtype=np.uint32)
mask[-2:] = 0  # padded rows must not enter kth-neighbour selection
positions[-2:] = 0
times[-2:] = 0
predictors[-2:] = 0
response[-2:] = 0
included = np.flatnonzero(mask)
design = np.column_stack([np.ones(len(included)), predictors[included]]).astype(float)
observed = response[included].astype(float)


def fit(bandwidth, time_scale, adaptive, bisquare):
    coordinates = positions[included].astype(float)
    ticks = times[included].astype(float)
    distance = np.sqrt(((coordinates[:, None] - coordinates[None, :])**2).sum(axis=2) +
                       time_scale * (ticks[:, None] - ticks[None, :])**2)
    local_bandwidth = (np.sort(distance, axis=1)[:, int(bandwidth)-1] * 1.0000001
                       if adaptive else np.full(len(included), bandwidth))
    ratio = distance / local_bandwidth[:, None]
    weights = np.where(ratio < 1, (1-ratio**2)**2, 0) if bisquare else np.exp(-ratio**2/2)
    matrices = np.einsum('ij,jk,jl->ikl', weights, design, design)
    right = np.einsum('ij,jk,j->ik', weights, design, observed)
    coefficients = np.linalg.solve(matrices, right[..., None])[..., 0]
    fitted = (design * coefficients).sum(axis=1)
    inverse = np.linalg.inv(matrices)
    hat = np.einsum('ij,ijk,ik->i', design, inverse, design)
    residual_squares = ((observed - fitted)**2).sum()
    trace = hat.sum()
    count = len(included)
    log_likelihood = -.5 * count * (np.log(2*np.pi*residual_squares/count)+1)
    score = -2*log_likelihood + 2*count*(trace+1)/(count-trace-2) if count-trace-2 > 0 else float('inf')
    return {'coefficients': coefficients.ravel().tolist(), 'fitted': fitted.tolist(),
            'hat': hat.tolist(), 'aicc': float(score)}


cases = []
for adaptive, bisquare, bounds, fixed in [
    (False, False, [0.7, 4, 0, 4], [1.8, 1.3]),
    (True, True, [12, 34, 0, 4], [22, 1.3]),
    (False, True, [2.5, 6, 0, 0], [4, 0]),
]:
    bandwidths = np.arange(bounds[0], bounds[1]+1) if adaptive else np.linspace(bounds[0], bounds[1], 81)
    scales = np.linspace(bounds[2], bounds[3], 61) if bounds[2] != bounds[3] else [0]
    grid = [(fit(bandwidth, scale, adaptive, bisquare)['aicc'], float(bandwidth), float(scale))
            for bandwidth in bandwidths for scale in scales]
    best = min(grid)
    cases.append({'adaptive': adaptive, 'kernel': 'bisquare' if bisquare else 'gaussian',
                  'bounds': bounds, 'fixed': fixed, 'expected': fit(*fixed, adaptive, bisquare),
                  'grid': {'aicc': best[0], 'bandwidth': best[1], 'timeScale': best[2]}})

fixture = {'rows': rows, 'positions': positions.ravel().tolist(), 'times': times.tolist(),
           'predictors': predictors.tolist(), 'response': response.tolist(), 'mask': mask.tolist(),
           'cases': cases}
# Independent fixed-parameter Gauss-Seidel backfit, including the adaptive distance bandwidth.
initial = fit(22, 1.3, True, True)
coefficients = np.array(initial['coefficients']).reshape(-1, 2)
components = coefficients * design
error = observed - components.sum(axis=1)
for component in range(2):
    partial_response = error + components[:, component]
    coordinates = positions[included].astype(float)
    ticks = times[included].astype(float)
    distance = np.sqrt(((coordinates[:, None] - coordinates[None, :])**2).sum(axis=2) +
                       1.3 * (ticks[:, None] - ticks[None, :])**2)
    local_bandwidth = np.sort(distance, axis=1)[:, 21] * 1.0000001
    ratio = distance / local_bandwidth[:, None]
    weights = np.where(ratio < 1, (1-ratio**2)**2, 0)
    column = design[:, component]
    coefficients[:, component] = (weights @ (column * partial_response)) / (weights @ (column**2))
    components[:, component] = column * coefficients[:, component]
    error = partial_response - components[:, component]
fixture['multiscale'] = {'coefficients': coefficients.ravel().tolist(), 'fitted': components.sum(axis=1).tolist(),
                         'residuals': error.tolist()}

# Exhaustive adaptive bandwidth selection on a single time-scale slice, before each Gauss-Seidel
# update. The component optimum differs from the initial multivariate optimum on this scene.
bandwidths = range(12, 35)
initial_bandwidth = min(bandwidths, key=lambda bandwidth: fit(bandwidth, 1.3, True, True)['aicc'])
coefficients = np.array(fit(initial_bandwidth, 1.3, True, True)['coefficients']).reshape(-1, 2)
components = coefficients * design
error = observed - components.sum(axis=1)
selected_bandwidths = [initial_bandwidth]
coordinates = positions[included].astype(float)
ticks = times[included].astype(float)
distance = np.sqrt(((coordinates[:, None] - coordinates[None, :])**2).sum(axis=2) +
                   1.3 * (ticks[:, None] - ticks[None, :])**2)
for component in range(2):
    partial_response = error + components[:, component]
    column = design[:, component]
    candidates = []
    for bandwidth in bandwidths:
        local_bandwidth = np.sort(distance, axis=1)[:, bandwidth-1] * 1.0000001
        ratio = distance / local_bandwidth[:, None]
        weights = np.where(ratio < 1, (1-ratio**2)**2, 0)
        denominator = weights @ (column**2)
        beta = (weights @ (column * partial_response)) / denominator
        smoothed = column * beta
        trace = np.sum(column**2 / denominator)
        count = len(included)
        residual_squares = np.sum((partial_response - smoothed)**2)
        score = count * (np.log(2*np.pi*residual_squares/count)+1) + 2*count*(trace+1)/(count-trace-2)
        candidates.append((float(score), bandwidth, beta, smoothed))
    _, bandwidth, beta, smoothed = min(candidates, key=lambda candidate: candidate[:2])
    selected_bandwidths.append(bandwidth)
    coefficients[:, component] = beta
    components[:, component] = smoothed
    error = partial_response - smoothed
fixture['multiscaleSearch'] = {'bandwidths': selected_bandwidths, 'coefficients': coefficients.ravel().tolist(),
                              'fitted': components.sum(axis=1).tolist(), 'residuals': error.tolist()}
# Regression: rounding the interior pair on [15, 19] must not lose the unimodal optimum 18.
collision_scores = [fit(bandwidth, 1, True, True)['aicc'] for bandwidth in range(15, 20)]
fixture['roundedCandidates'] = {'bandwidths': list(range(15, 20)), 'aicc': collision_scores,
                                'optimum': 15 + int(np.argmin(collision_scores))}

# Large but finite RSS must stay representable until the logarithm is evaluated.
large_positions = np.column_stack([np.arange(8), np.zeros(8)]).astype(np.float32)
large_response = (np.array([-1, 1] * 4) * 5e18).astype(np.float32).astype(float)
large_predictors = (np.arange(8) % 3 - 1).astype(np.float32).astype(float)
large_distance = np.arange(8)[:, None] - np.arange(8)[None, :]
large_weights = np.exp(-.5 * (large_distance / 100)**2)
large_expected = {}
for model in ['component', 'gtwr']:
    large_design = (np.ones((8, 1)) if model == 'component' else
                    np.column_stack([np.ones(8), large_predictors]))
    large_matrices = np.einsum('ij,jk,jl->ikl', large_weights, large_design, large_design)
    large_right = np.einsum('ij,jk,j->ik', large_weights, large_design, large_response)
    large_coefficients = np.linalg.solve(large_matrices, large_right[..., None])[..., 0]
    large_fitted = np.sum(large_design * large_coefficients, axis=1)
    large_hat = np.einsum('ij,ijk,ik->i', large_design, np.linalg.inv(large_matrices), large_design)
    large_residual_squares = np.sum((large_response - large_fitted)**2)
    large_trace = np.sum(large_hat)
    large_score = 8 * (np.log(large_residual_squares) - np.log(8) + np.log(2*np.pi) + 1) + 16 * (large_trace + 1) / (6 - large_trace)
    large_expected[model] = {'fitted': large_fitted.tolist(), 'hat': large_hat.tolist(), 'rss': float(large_residual_squares),
                             'aicc': float(large_score)}
fixture['largeResponse'] = {'positions': large_positions.ravel().tolist(),
                          'points': np.column_stack([large_positions, np.zeros(8)]).ravel().tolist(),
                          'response': large_response.tolist(), 'predictors': large_predictors.tolist(),
                          'expected': large_expected}
Path('modules/analysis/test/oracles/fixtures/gtwr-bandwidth-search.json').write_text(json.dumps(fixture, indent=2) + '\n')
