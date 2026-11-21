# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regionalization/regionalization-audit-evaluation.oracle.py
"""Direct partition moments and sklearn Calinski-Harabasz on complete finite labelled rows."""
import numpy as np
from sklearn.metrics import calinski_harabasz_score

values = np.array([[0, 1], [2, 3], [4, 5], [8, 1], [10, 3], [12, 5], [20, 9], [22, 11],
                   [100, 100], [np.nan, 0], [0, np.inf], [-100, -100]], dtype=float)
labels = np.array([0, 0, 0, 2, 2, 2, 3, 3, 4294967295, 0, 2, 9])
used = (labels < 5) & np.isfinite(values).all(axis=1)
for scale, shift in [(1, 0), (3, 64)]:
    table = values * scale + shift
    means = [table[used & (labels == label)].mean(axis=0).tolist()
             if np.any(used & (labels == label)) else [float('nan')] * 2 for label in range(5)]
    sizes = [np.sum(used & (labels == label)) for label in range(5)]
    within = [np.sum((table[used & (labels == label)] - means[label]) ** 2) for label in range(5)]
    global_mean = table[used].mean(axis=0)
    between = sum(sizes[label] * np.sum((means[label] - global_mean) ** 2)
                  for label in range(5) if sizes[label])
    total = np.sum((table[used] - global_mean) ** 2)
    print('scale, shift', scale, shift, 'sizes', sizes, 'means', means, 'within', within)
    print('summary', [3, sum(within), between, total, 2, 3, 2 / 7, 4,
                      calinski_harabasz_score(table[used], labels[used])], 'invalidRowCount', 2)

# Two row tiles and two label tiles, with all special counts in the final partial tile.
values = np.array([row % 2 for row in range(256)] + [10, 12, np.nan, 99], dtype=float)
labels = np.array([0] * 256 + [259, 259, 0, 4294967295])
used = (labels < 260) & np.isfinite(values)
within = sum(np.sum((values[used & (labels == region)] - values[used & (labels == region)].mean()) ** 2)
             for region in [0, 259])
total = np.sum((values[used] - values[used].mean()) ** 2)
print('two-tile summary', [2, within, total - within, total, 2, 256, 2 / 3, 2,
                           calinski_harabasz_score(values[used, None], labels[used])], 'invalidRowCount', 1)
