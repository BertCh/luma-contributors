# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regionalization/regionalization-review.oracle.py
import numpy as np

values = np.array([0., 1., 4.])
masses = np.array([1., 1., 2.])
labels = np.array([0, 0, 1])
within = sum(np.sum((values[labels == region] - values[labels == region].mean()) ** 2) for region in [0, 1])
print('R1 final masses', [masses[labels == region].sum() for region in [0, 1]], 'SSD', within)

values = np.array([992.5, 1000., 1000., 1000., 999.])
labels = np.array([0, 0, 1, 1, 2])
def get_within(table, partition):
    return sum(np.sum((table[partition == region] - table[partition == region].mean()) ** 2)
               for region in np.unique(partition))
for destination in [1, 2]:
    candidate = labels.copy()
    candidate[1] = destination
    gain = get_within(values, labels) - get_within(values, candidate)
    # Only the four regional squared-moment terms define the documented cutoff.
    scale = sum(values[partition == region].sum() ** 2 / np.sum(partition == region)
                for partition in [labels, candidate] for region in [0, destination])
    print('R2 destination', destination, 'direct SSD gain', gain, 'cutoff', 4e-6 * scale)

print('R3 connected path-group SSD', get_within(np.array([1.,3.,0.,6.,0.,3.]), np.array([0,0,0,0,1,1])))
# Growth partitions are the explicit pre-enclave regions. Every region contributes to the
# score even if subsequent enclave assignment removes every member of one of them.
values = np.array([3.,0.,6.,0.,3.,1.])
for groups in [[[0,1], [3,4]], [[0,1], [2,3], [4,5]]]:
    print('R4 grown SSD', sum(np.sum((values[group] - values[group].mean()) ** 2) for group in groups))
print('R4 selected final SSD', get_within(values, np.array([0,0,1,1,1,1])))

values = np.array([0.9, 0.9, 0.9, 2., 2., 2.], dtype=np.float32)
labels = np.array([0, 0, 0, 1, 1, 1])
within = get_within(values.astype(float), labels)
raw_sum = np.float32(np.float32(values[0] + values[1]) + values[2])
raw_mean = np.float32(raw_sum / np.float32(3))
raw_within = np.float32(3) * np.float32(np.float32(values[0] - raw_mean) ** 2)
total = np.sum((values.astype(float) - values.astype(float).mean()) ** 2)
print('R5 stored values', values.astype(float).tolist(), 'oracle within', within,
      'total/between', total, 'pseudo-F', 1, 'uncentered f32 within', raw_within)

amplitude = 3.0 * 2.0 ** 61
values = np.array([-amplitude, -amplitude / 2, amplitude, 1.5 * amplitude], dtype=np.float32).astype(float)
labels = np.array([0,0,1,1])
within = get_within(values, labels)
total = np.sum((values - values.mean()) ** 2)
between = total - within
with np.errstate(over='ignore'):
    old_pseudo = np.float32(np.float32(between) * np.float32(2)) / np.float32(within)
print('R9 within/between/total', within, between, total, 'oracle pseudo-F', (between / 1) / (within / 2),
      'old f32 pseudo-F', old_pseudo)
print('R9 regional means', [values[labels == region].mean() for region in [0, 1]],
      'regional within', [np.sum((values[labels == region] - values[labels == region].mean()) ** 2) for region in [0, 1]])

for scale in [1., 2. ** -24]:
    values = np.array([16777216., -16777215., -.5, -.5], dtype=np.float32) * np.float32(scale)
    labels = np.array([0, 0, 1, 1])
    table = values.astype(float)
    means = [table[labels == region].mean() for region in [0, 1]]
    old_first = np.float32(values[0] + np.float32(np.float32(values[1] - values[0]) / np.float32(2)))
    between = sum(2 * mean ** 2 for mean in means)
    print('R12 scale', scale, 'oracle means', means, 'between', between, 'old first mean', old_first,
          'within', get_within(table, labels), 'total', np.sum(table ** 2))

values = np.full(3, 1000000.125, dtype=np.float32)
old_global = np.float32(np.sum(values, dtype=np.float32) / np.float32(3))
print('R13 oracle within/between/total', [0., 0., 0.], 'old global mean', float(old_global),
      'old between', np.float32(3) * np.float32(np.float32(values[0] - old_global) ** 2))

for exponent in [-74, -63]:
    scale = 2. ** exponent
    within = np.float32(2 * scale ** 2)
    between = np.float32(32 * scale ** 2)
    with np.errstate(divide='ignore'):
        old_ratio = between / np.float32(within / np.float32(6))
    print('R15 scale exponent', exponent, 'within/between', within, between,
          'old float32 pseudo-F', old_ratio, 'oracle pseudo-F', 96.)
