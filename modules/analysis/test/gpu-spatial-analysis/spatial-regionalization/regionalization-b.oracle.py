# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regionalization/regionalization-b.oracle.py
"""NumPy centered SSD and an independent sorted-edge union-find forest.

Forest costs use population standardization; Ward prefixes use raw values. Ward ties use the documented lowbias32
pair key (seed zero), then the two smallest member rows. Disconnected components
are never joined. REDCAP-named single linkage retains the cheapest forest edges;
this oracle does not claim to implement Guo's SSD partitioning step.
"""
import numpy as np

VALUES = np.array([0, 2, 10, 14, 1, 3, 20, 26, 99], dtype=np.float64)
EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
         (4, 5), (4, 6), (5, 6), (5, 7), (6, 7)]
NEIGHBORS = [sorted(b if a == row else a for a, b in EDGES if row in (a, b))
             for row in range(9)]
OFFSETS = np.cumsum([0] + [len(row) for row in NEIGHBORS]).tolist()
SLOTS = {(a, b): OFFSETS[a] + NEIGHBORS[a].index(b) for a, b in EDGES}
TABLE = (VALUES - VALUES.mean()) / VALUES.std(ddof=0)


def ssd(members):
    data = VALUES[members]
    return float(np.sum((data - data.mean()) ** 2))


def key(first, second):
    value = ((first * 0x9e3779b9) ^ second) & 0xffffffff
    value ^= value >> 16
    value = (value * 0x7feb352d) & 0xffffffff
    value ^= value >> 15
    value = (value * 0x846ca68b) & 0xffffffff
    return value ^ (value >> 16)


def labels(groups):
    return [min(next(group for group in groups if row in group)) for row in range(9)]


parents = list(range(9))


def root(row):
    while parents[row] != row:
        row = parents[row]
    return row


forest = []
for a, b in sorted(EDGES, key=lambda edge: ((TABLE[edge[0]] - TABLE[edge[1]]) ** 2,
                                          SLOTS[edge])):
    if root(a) != root(b):
        parents[max(root(a), root(b))] = min(root(a), root(b))
        forest.append((a, b))
forest.sort(key=lambda edge: SLOTS[edge])
print('offsets', OFFSETS, 'neighbors', sum(NEIGHBORS, []))
print('standardized', TABLE.tolist())
print('forest slots', [SLOTS[edge] for edge in forest])
print('forest costs', [float((TABLE[a] - TABLE[b]) ** 2) for a, b in forest])

for schedule in ['sequential', 'mutual-nearest']:
    for budget in range(8):
        groups = [[row] for row in range(9)]
        merges = rounds = 0
        exhausted = False
        while len(groups) > 1 and rounds < budget:
            rounds += 1
            candidates = []
            for first_index, first in enumerate(groups):
                for second in groups[first_index + 1:]:
                    if not any((a in first and b in second) or (b in first and a in second)
                               for a, b in EDGES):
                        continue
                    a, b = sorted([min(first), min(second)])
                    candidates.append((ssd(first + second) - ssd(first) - ssd(second),
                                       key(a, b), a, b, first, second))
            candidates.sort(key=lambda pair: pair[:4])
            if not candidates:
                exhausted = True
                break
            selected = candidates[:1]
            if schedule == 'mutual-nearest':
                nearest = {}
                for pair in candidates:
                    for row in pair[2:4]:
                        nearest.setdefault(row, pair)
                selected = [pair for pair in candidates
                            if nearest[pair[2]] is pair and nearest[pair[3]] is pair]
            for *_, first, second in selected:
                groups.remove(first)
                groups.remove(second)
                groups.append(sorted(first + second))
                merges += 1
        print(schedule, budget, labels(groups), sum(ssd(group) for group in groups),
              [merges, 0, int(not exhausted and rounds == budget), 0,
               rounds if schedule == 'mutual-nearest' else 0])

# Reproduce the pre-fix float32 standardizer arithmetic, independently of TS/WGSL.
large = (VALUES * 2**64).astype(np.float32)
with np.errstate(over='ignore', invalid='ignore'):
    deviations = large - large[0]
    deviations -= np.sum(deviations, dtype=np.float32) / np.float32(9)
    variance = np.sum(deviations * deviations, dtype=np.float32) / np.float32(9)
    broken = deviations / np.sqrt(variance)
print('pre-fix variance', variance, 'standardized', broken.tolist())

# A power-of-two normalization changes units exactly, without changing significands.
# This float32 arithmetic probe is additional evidence; NumPy's float64 TABLE above is the oracle.
for scale, shift in [(1, 0), (2**-80, 0), (2**64, 64), (2**120, 64)]:
    data = ((VALUES + shift) * scale).astype(np.float32)
    maximum = np.max(np.abs(data))
    normalizer = np.float32(2.0 ** np.clip(np.frexp(float(maximum))[1] - 1, -126, 126))
    centered = data / normalizer - data[0] / normalizer
    centered -= np.sum(centered, dtype=np.float32) / np.float32(9)
    variance = np.sum(centered * centered, dtype=np.float32) / np.float32(9)
    corrected = centered / np.sqrt(variance)
    print('power-of-two probe', scale, 'largest z-score error', float(np.max(np.abs(corrected - TABLE))))
print('Ward CPU objective rounding error', abs(float(np.float32(30 / VALUES.var())) - 30 / VALUES.var()))
