# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-validation/area-of-applicability-audit.oracle.py

CAST/Meyer-Pebesma definition: sample standardization, mean off-diagonal
distance, fold-restricted nearest distance, NumPy linear/type-7 quartiles.
"""
import numpy as np
from scipy.spatial.distance import cdist

training = np.array([0, 1, 1, 2, 4, 6, 10, 12], dtype=float)[:, None]
query = np.array([1.5, 20], dtype=float)[:, None]
scale = training.std(axis=0, ddof=1)
distances = cdist(training / scale, training / scale)
mean_distance = distances.sum() / (8 * 7)
folds = np.array([0, 0, 1, 1, 0, 1, 2, 2])
training_masks = np.array([[0, 0, 1, 1, 0, 1, 1, 1],
                           [1, 1, 0, 0, 1, 0, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)
nearest = [distances[row, training_masks[fold]].min() if training_masks[fold].any()
           else float('nan') for row, fold in enumerate(folds)]
indices = np.array(nearest) / mean_distance
first, third = np.quantile(indices[np.isfinite(indices)], [0.25, 0.75])
threshold = min(third + 1.5 * (third - first), np.nanmax(indices))
print('training', indices.tolist() + [float('nan')])
print('query', (cdist(query / scale, training / scale).min(axis=1) / mean_distance).tolist() + [float('nan')])
print('summary', [mean_distance, threshold, 8, 6, first, third])
print('applicable', [1, 0, 0], 'status', 0)
