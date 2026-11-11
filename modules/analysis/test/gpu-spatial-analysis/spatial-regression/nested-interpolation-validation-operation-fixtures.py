# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run /usr/bin/python3 on this file. SciPy RBFInterpolator and LAPACK oracle.

The independent Philox integer implementation pins fold membership, including
outer folds that remove whole blocks. All inputs are rounded to upload bytes.
No output from the CPU twin is used to generate this fixture.
"""
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.interpolate import RBFInterpolator
from scipy.optimize import brentq


def block_key(block, seed):
    counter = [int(block), 0, 0, 0]
    key = [seed, 0x464F4C44]
    mask = 0xFFFFFFFF
    for _ in range(10):
        first = counter[0] * 0xD2511F53
        second = counter[2] * 0xCD9E8D57
        counter = [((second >> 32) ^ counter[1] ^ key[0]) & mask,
                   second & mask,
                   ((first >> 32) ^ counter[3] ^ key[1]) & mask,
                   first & mask]
        key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    return counter[0]


random = np.random.default_rng(20261111)
positions = random.uniform(-2, 2, (30, 2)).astype(np.float32).astype(float)
response = (np.sin(positions[:, 0]) + .4 * positions[:, 1]**2 +
            .15 * np.cos(3 * positions[:, 1])).astype(np.float32).astype(float)
blocks = np.repeat(np.arange(15), 2)
seed = 19
inner_count = 3
neighbor_count = 8
buffer_distance = .4
candidates = [dict(mode='rbf', kernel=kernel, epsilon=.85, smoothing=.04)
              for kernel in ['thin-plate-spline', 'cubic', 'gaussian', 'linear',
                             'quintic', 'multiquadric', 'inverse-multiquadric', 'inverse-quadratic']]
candidates += [dict(mode='ordinary-kriging', covariance=dict(model=model, range=2.5, sill=1.3, nugget=.03))
               for model in ['spherical', 'exponential', 'gaussian', 'matern-3/2', 'matern-5/2']]


def predict(candidate, training, query_row):
    available = np.flatnonzero(training)
    query = positions[query_row]
    distances = np.linalg.norm(positions[available] - query, axis=1)
    retained = available[np.lexsort((available, distances))[:neighbor_count]]
    sites, values = positions[retained], response[retained]
    if candidate['mode'] == 'rbf':
        # SciPy defaults to degree 0 for positive-definite kernels; the contributor defaults -1.
        degree = (2 if candidate['kernel'] == 'quintic' else
                  1 if candidate['kernel'] in ['thin-plate-spline', 'cubic'] else
                  0 if candidate['kernel'] in ['linear', 'multiquadric'] else -1)
        model = RBFInterpolator(sites, values, kernel=candidate['kernel'].replace('-', '_'),
                                epsilon=candidate['epsilon'], smoothing=candidate['smoothing'], degree=degree)
        return float(model(query[None])[0])
    covariance = candidate['covariance']
    model = covariance['model']

    def correlate(distance):
        scaled = distance / covariance['range']
        if model == 'spherical':
            return np.where(scaled >= 1, 0, 1 - 1.5 * scaled + .5 * scaled**3)
        if model == 'exponential':
            return np.exp(-3 * scaled)
        if model == 'gaussian':
            return np.exp(-3 * scaled**2)
        if model == 'matern-3/2':
            factor = brentq(lambda value: (1 + value) * np.exp(-value) - np.exp(-3), 1, 20)
            argument = factor * scaled
            return (1 + argument) * np.exp(-argument)
        factor = brentq(lambda value: (1 + value + value**2/3) * np.exp(-value) - np.exp(-3), 1, 20)
        argument = factor * scaled
        return (1 + argument + argument**2/3) * np.exp(-argument)

    pair_distance = np.linalg.norm(sites[:, None] - sites[None, :], axis=2)
    cross_distance = np.linalg.norm(sites - query, axis=1)
    matrix = covariance['sill'] * correlate(pair_distance) + (pair_distance == 0) * covariance['nugget']
    cross = covariance['sill'] * correlate(cross_distance) + (cross_distance == 0) * covariance['nugget']
    system = np.block([[matrix, np.ones((len(sites), 1))],
                       [np.ones((1, len(sites))), np.zeros((1, 1))]])
    weights = np.linalg.solve(system, np.r_[cross, 1])[:-1]
    return float(weights @ values)


folds = []
for outer in range(3):
    # Whole blocks are held out; inner block ranks must be recomputed for each outer fold.
    testing = blocks % 3 == outer
    training = ~testing
    groups = sorted(set(blocks[training]), key=lambda block: (block_key(block, seed), block))
    assignments = {block: rank % inner_count for rank, block in enumerate(groups)}
    inner_folds = np.array([assignments.get(block, -1) for block in blocks])
    scores = []
    buffered_counts = []
    for candidate in candidates:
        errors = []
        for split in range(inner_count):
            inner_testing = training & (inner_folds == split)
            inner_training = training & (inner_folds != split)
            distances = np.linalg.norm(positions[:, None] - positions[inner_testing][None, :], axis=2)
            inner_training &= np.all(distances > buffer_distance, axis=1)
            if len(scores) == 0:
                buffered_counts.append(int(np.sum(training & (inner_folds != split)) - np.sum(inner_training)))
            for row in np.flatnonzero(inner_testing):
                errors.append(predict(candidate, inner_training, row) - response[row])
        scores.append([float(np.mean(np.square(errors))), len(errors), 0])
    selected = int(np.argmin(np.array(scores)[:, 0]))
    predictions = [predict(candidates[selected], training, row) if testing[row] else None for row in range(30)]
    actual = np.array([predictions[row] for row in np.flatnonzero(testing)])
    errors = actual - response[testing]
    metrics = [len(errors), 0, float(np.mean(errors**2)), float(np.mean(np.abs(errors))),
               float(np.mean(errors)), float(1 - np.sum(errors**2) / np.sum((response[testing] - response[testing].mean())**2))]
    # Each candidate also gets an outer prediction oracle, so every kernel is checked as a winner.
    candidate_predictions = [[predict(candidate, training, row) if testing[row] else None
                              for row in range(30)] for candidate in candidates]
    folds.append(dict(training=training.astype(int).tolist(), testing=testing.astype(int).tolist(),
                      innerFolds=inner_folds.tolist(), bufferedCounts=buffered_counts,
                      scores=scores, selected=selected, predictions=predictions, metrics=metrics,
                      candidatePredictions=candidate_predictions))

result = dict(provenance=dict(numpy=np.__version__, scipy=scipy.__version__,
                             oracle='SciPy RBFInterpolator, NumPy LAPACK bordered covariance solve, independent Philox'),
              positions=positions.ravel().tolist(), response=response.tolist(), blocks=blocks.tolist(),
              seed=seed, innerFoldCount=inner_count, neighborCount=neighbor_count,
              innerBufferDistance=buffer_distance, candidates=candidates, folds=folds)
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
