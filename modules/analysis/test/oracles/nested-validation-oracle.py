# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent nested held-out oracle; no runtime WGSL/code translation is evaluated.

Run /tmp/luma-coverage-oracles/bin/python modules/analysis/test/oracles/nested-validation-oracle.py.
RBF fits call scipy.interpolate.RBFInterpolator; ordinary kriging solves the
bordered covariance equations with NumPy LAPACK. Neighborhoods use explicit
(distance, original row) ordering to make equal-distance ties reproducible.
The inner partition is fixed here (blocks 0, 1 and 4 in fold 1, the others in
fold 0); the spec picks the seed whose block order gives it and checks that on
the GPU. Scores, predictions, and metrics are computed independently.
"""
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.interpolate import RBFInterpolator

assert np.__version__ == '2.2.6'
assert scipy.__version__ == '1.15.3'
POSITIONS = np.array([[row % 4, row // 4] for row in range(12)], dtype=float)
# Match the observation bytes uploaded by the browser test, then calculate in f64.
RESPONSE = (2 + .3 * POSITIONS[:, 0] + .8 * POSITIONS[:, 1]).astype(np.float32).astype(float)
BLOCKS = np.arange(12) // 2

# Inner fold of each of the six blocks. (These are the values the former `hash(block ^ 2) % 2`
# rule gave, kept so the fixture did not change when the contributor moved to a ranked order.)
BLOCK_FOLDS = [1, 1, 0, 0, 1, 0]
INNER_FOLDS = np.array([BLOCK_FOLDS[group] for group in BLOCKS])

def predict(candidate, training, query):
    available = np.flatnonzero(training)
    distances = np.linalg.norm(POSITIONS[available] - query, axis=1)
    retained = available[np.lexsort((available, distances))[:4]]
    sites, values = POSITIONS[retained], RESPONSE[retained]
    if candidate:
        model = RBFInterpolator(sites, values, kernel='gaussian', epsilon=[.5, 1][candidate-1], smoothing=.1, degree=-1)
        return float(model(query[None])[0])
    # Effective range3: covariance exp(-3d/3), sill1, coincident nugget.01.
    pair_distance = np.linalg.norm(sites[:, None] - sites[None, :], axis=2)
    covariance = np.exp(-pair_distance) + np.eye(len(sites)) * .01
    cross_distance = np.linalg.norm(sites - query, axis=1)
    cross = np.exp(-cross_distance) + (cross_distance == 0) * .01
    system = np.block([[covariance, np.ones((len(sites), 1))], [np.ones((1, len(sites))), np.zeros((1, 1))]])
    weights = np.linalg.solve(system, np.r_[cross, 1])[:-1]
    return float(weights @ values)

folds = []
for outer in range(2):
    training = np.arange(12) % 2 != outer
    testing = ~training
    scores = []
    for candidate in range(3):
        errors = []
        for inner in range(2):
            inner_training = training & (INNER_FOLDS != inner)
            inner_testing = training & (INNER_FOLDS == inner)
            # All sites in this fixture are distinct: default zero-radius exclusion
            # removes no additional rows after the inner test set is removed.
            for row in np.flatnonzero(inner_testing):
                errors.append(predict(candidate, inner_training, POSITIONS[row]) - RESPONSE[row])
        scores.append([float(np.mean(np.square(errors))), len(errors), 0])
    selected = int(np.argmin(np.array(scores)[:, 0]))
    predictions = [predict(selected, training, POSITIONS[row]) if testing[row] else None for row in range(12)]
    actual = np.array([predictions[row] for row in np.flatnonzero(testing)])
    observed = RESPONSE[testing]
    errors = actual - observed
    metrics = [len(errors), 0, float(np.mean(errors**2)), float(np.mean(np.abs(errors))), float(np.mean(errors)), float(1 - np.sum(errors**2)/np.sum((observed-observed.mean())**2))]
    folds.append(dict(training=training.astype(int).tolist(), testing=testing.astype(int).tolist(), scores=scores, selected=selected, predictions=predictions, metrics=metrics))

result = dict(provenance=dict(generator='nested-validation-oracle.py', numpy=np.__version__, scipy=scipy.__version__, ordinaryKriging='Independent bordered covariance equations solved by numpy.linalg.solve', rbf='Direct scipy.interpolate.RBFInterpolator, deterministic caller-selected neighborhoods', precision='Input response rounded to float32; independent float64 calculation', scope='Twelve finite unique sites; two outer and two seeded inner folds; three candidate models; four-neighbor ties by source row'), positions=POSITIONS.ravel().tolist(), response=RESPONSE.tolist(), blocks=BLOCKS.tolist(), innerFolds=INNER_FOLDS.tolist(), folds=folds)
output = Path(__file__).parent / 'fixtures/nested-validation-oracle.json'
output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({'fixture': str(output), 'selected': [fold['selected'] for fold in folds], 'mse': [fold['scores'] for fold in folds]}))
