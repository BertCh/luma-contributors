# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent NumPy masks and SciPy Wasserstein and kmeans2 fixtures.

Run with /usr/bin/python3 from the repository root, without -I.
"""
import json
from pathlib import Path
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


def create_outputs(positions, folds, fold_count, buffer_distance):
    separation = cdist(positions, positions)
    testing = np.equal.outer(np.arange(fold_count), folds)
    training = np.logical_and(folds[None, :] < fold_count, ~testing)
    for fold in range(fold_count):
        training[fold] &= ~np.any(separation[:, testing[fold]] <= buffer_distance, axis=1)
    counts = [[int(training[fold].sum()), int(testing[fold].sum()),
               int((folds < fold_count).sum() - training[fold].sum() - testing[fold].sum())] for fold in range(fold_count)]
    status = [int(row[0] == 0) + 2*int(row[1] == 0) for row in counts]
    return dict(expectedFolds=folds.tolist(), expectedTraining=training.astype(int).ravel().tolist(),
                expectedTesting=testing.astype(int).ravel().tolist(), expectedCounts=np.array(counts).ravel().tolist(), expectedStatus=status)


positions = np.array([[0, 0], [0.4, 0.2], [1.2, 0], [2, 1], [4, 0], [5, 2], [7, 0], [10, 3]], dtype=np.float32).astype(float)
predictions = np.array([[0, 0], [0.8, 0.6], [2.5, 2], [6, 4], [12, 3]], dtype=np.float32).astype(float)
candidates = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0, 1, 1]])
prediction_distances = cdist(predictions, positions).min(axis=1)
objectives = []
for assignments in candidates:
    if max(np.bincount(assignments)) > 0.8*len(positions):
        objectives.append(float('inf'))
        continue
    distances = cdist(positions, positions)
    distances[assignments[:, None] == assignments[None, :]] = float('inf')
    objectives.append(wasserstein_distance(distances.min(axis=1), prediction_distances))
selected = int(np.argmin(objectives))
knndm = dict(positions=positions.ravel().tolist(), predictionPositions=predictions.ravel().tolist(),
             candidates=candidates.ravel().tolist(), candidateCount=len(candidates), selected=selected,
             objectives=[value if np.isfinite(value) else None for value in objectives],
             **create_outputs(positions, candidates[selected], 2, 0.6))

features = np.array([[-3, -2, 7], [-2, -1, 7], [-4, -3, 7], [5, 7, 7], [6, 6, 7], [4, 8, 7]], dtype=float)
standardized = features - features.mean(axis=0)
scales = features.std(axis=0, ddof=1)
scales[scales == 0] = 1
standardized /= scales
centers, labels = kmeans2(standardized, standardized[[1, 5]], iter=100, minit='matrix')
environmental = dict(positions=positions[:6].ravel().tolist(), features=features.ravel().tolist(),
                     objective=float(((standardized-centers[labels])**2).sum()),
                     **create_outputs(positions[:6], labels, 2, 0.6))
# These restart outcomes are also hand-checked in the node spec. SciPy supplies the Lloyd
# iteration and Wasserstein implementation; it does not reuse the twin's clustering or CDF walk.
restart_positions = np.array([[7, 10], [7, 9], [8, 2], [0, 6], [9, 5], [1, 11]], dtype=float)
restart_predictions = np.array([[8, 3], [4, 5]], dtype=float)
restart_prediction_distances = cdist(restart_predictions, restart_positions).min(axis=1)
restart_partitions, restart_objectives = [], []
for candidate in range(3):
    seed_row = ((1 + candidate*2654435761) % 2**32) % len(restart_positions)
    first_center = restart_positions[seed_row]
    farthest_row = int(np.argmax(((restart_positions-first_center)**2).sum(axis=1)))
    initial_centers = restart_positions[[seed_row, farthest_row]]
    _, assignments = kmeans2(restart_positions, initial_centers, iter=100, minit='matrix')
    distances = cdist(restart_positions, restart_positions)
    distances[assignments[:, None] == assignments[None, :]] = float('inf')
    restart_partitions.append(assignments.tolist())
    restart_objectives.append(wasserstein_distance(distances.min(axis=1), restart_prediction_distances))
restart_selected = int(np.argmin(restart_objectives))
generated = dict(positions=restart_positions.ravel().tolist(), predictionPositions=restart_predictions.ravel().tolist(),
                 partitions=restart_partitions, objectives=restart_objectives, selected=restart_selected,
                 **create_outputs(restart_positions, np.array(restart_partitions[restart_selected]), 2, 0.6))
Path(__file__).with_suffix('.json').write_text(json.dumps(dict(knndm=knndm, environmental=environmental, generated=generated), indent=2) + '\n')
