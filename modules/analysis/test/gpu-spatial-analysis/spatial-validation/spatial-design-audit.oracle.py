# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-validation/spatial-design-audit.oracle.py

SciPy assigns points to specified centers; centroids and squared-error are evaluated
independently. A ninth masked row is omitted before fitting. Ties go to center zero.
"""
import numpy as np
from scipy.cluster.vq import vq
from scipy.spatial.distance import cdist

features = np.array([0, 1, 1, 2, 10, 11, 11, 12], dtype=float)[:, None]
for centers in [[[0], [12]], [[1], [11]]]:
    labels, distances = vq(features, np.array(centers, dtype=float))
    print({'labels': labels.tolist(), 'objective': float(distances @ distances)})


def philox_word(block, seed):
    """Random123 Philox4x32-10 round definition, with exact Python products."""
    counter, key = [block, 0, 0, 0], [seed, 0x464F4C44]
    for _ in range(10):
        first, second = 0xD2511F53 * counter[0], 0xCD9E8D57 * counter[2]
        counter = [(second >> 32) ^ counter[1] ^ key[0], second & 0xFFFFFFFF,
                   (first >> 32) ^ counter[3] ^ key[1], first & 0xFFFFFFFF]
        key = [(key[0] + 0x9E3779B9) & 0xFFFFFFFF,
               (key[1] + 0xBB67AE85) & 0xFFFFFFFF]
    return counter[0]


positions = np.array([0, 1, 1, 2, 4, 6, 10, 12, 99], dtype=float)[:, None]
blocks = np.array([30, 10, 30, 50, 70, 10, 90, 110, 999])
usable = np.array([True] * 8 + [False])
ordered = sorted(set(blocks[usable]), key=lambda block: (philox_word(int(block), 1234), block))
folds = np.array([ordered.index(block) % 4 if valid else 0xFFFFFFFF
                  for block, valid in zip(blocks, usable)])
testing = np.equal.outer(np.arange(4), folds)
training = (folds[None, :] < 4) & ~testing
for fold in range(4):
    training[fold] &= ~np.any(cdist(positions, positions[testing[fold]]) <= 1, axis=1)
counts = [[int(training[fold].sum()), int(testing[fold].sum()), int(usable.sum() - training[fold].sum() - testing[fold].sum())] for fold in range(4)]
print('block folds', folds.tolist())
print('block training', training.astype(int).ravel().tolist())
print('block testing', testing.astype(int).ravel().tolist())
print('block counts', np.array(counts).ravel().tolist())
print('block status', [int(row[0] == 0) + 2 * int(row[1] == 0) for row in counts])

# R1: independently assigned by SciPy vq from farthest-first sites; no TS helpers.
features = (np.arange(2048) / 2047).astype(np.float32).astype(float)[:, None]
centers = features[[0]]
for _ in range(12):
    nearest = cdist(features, centers, metric='sqeuclidean').min(axis=1)
    centers = np.vstack([centers, features[np.argmax(nearest)]])
previous = np.full(2048, -1)
iterations = 0
while True:
    labels, distances = vq(features, centers)
    iterations += 1
    centers = np.array([features[labels == group].mean(axis=0) for group in range(13)])
    if np.array_equal(labels, previous):
        break
    previous = labels
print('R1', iterations, np.bincount(labels, minlength=13).tolist(), float(distances @ distances))

# R2: NNDM's radius is one; explicit radius three dominates. Test row is its fold ID.
positions = np.arange(8)[:, None]
training = cdist(positions, positions).T > max(1, 3)
print('R2 training', training.astype(int).ravel().tolist())
print('R2 counts', [[int(row.sum()), 1, 7 - int(row.sum())] for row in training])

# R8: shift-independent SciPy assignment to the two stable rational means.
for shift in [0, 500000, -500000]:
    features = (shift + np.array([0, 1, 2, 5, 6, 7, 9, 11]) / 32).astype(np.float32).astype(float)[:, None]
    labels, distances = vq(features, shift + np.array([[2 / 32], [8.25 / 32]]))
    print('R8', shift, labels.tolist(), float(distances @ distances))

for low, high in [(1, 1 + 2 ** -22), (1 + 2 ** -21, 10 + 2 ** -20)]:
    features = np.array([10, low, high, low, high, 10, low, high], dtype=np.float32).astype(float)[:, None]
    labels, distances = vq(features, np.array([[10], [low], [high]]))
    print('review 2 R1', labels.tolist(), float(distances @ distances))

# v3-1: three identical packed coordinates must have exactly their common value
# as centroid. Summing raw coordinates followed by reciprocal-based division can
# move it an extra ulp; zero deviations divided by three cannot move the origin.
coordinate = np.float32(10 + 2 ** -20)
raw_sum = np.float32(np.float32(coordinate + coordinate) + coordinate)
reciprocal_mean = np.float32(raw_sum * np.float32(1 / 3))
anchored_mean = np.float32(coordinate + np.float32(0 * np.float32(1 / 3)))
assert anchored_mean == coordinate and reciprocal_mean != coordinate
print('v3-1 constant centroid', {'expected': float(coordinate),
                               'raw reciprocal mean': float(reciprocal_mean),
                               'anchored mean': float(anchored_mean)})
