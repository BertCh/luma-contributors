# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-weights/weights-helpers-audit.oracle.py

Independent NumPy/SciPy definitions for the internal weights helper audit.
No TypeScript or implementation-generated outputs are read.
"""
import numpy as np
from scipy.spatial.distance import cdist
from fractions import Fraction
import struct

positions = np.array([[0, 0], [2, 0], [-2, 0], [0, 3], [2, 0], [10, 10], [np.nan, 0], [0, 0]])
mask = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=bool)
distances = cdist(positions, positions)
distances[~np.isfinite(distances)] = np.inf
np.fill_diagonal(distances, np.inf)
nearest = np.argmin(distances, axis=1).astype(np.uint32)
nearest[~np.isfinite(distances.min(axis=1))] = 0xffffffff
print('nearest', nearest.tolist())  # argmin resolves equal distances by lowest row.
print('selected nearest', np.where(mask, nearest, 0xffffffff).tolist(), 'count', mask.sum())

high = np.array([2, 1, 2, 1, 0xffffffff, 1, 0, 0xffffffff], dtype=np.uint32)
low = np.array([1, 9, 1, 2, 0xffffffff, 2, 7, 0xffffffff], dtype=np.uint32)
order = np.lexsort((np.arange(8), low, high))
ordered = np.column_stack([high[order], low[order]])
starts = np.r_[True, np.any(ordered[1:] != ordered[:-1], axis=1)]
starts |= np.all(ordered == 0xffffffff, axis=1)
print('pair order', order.tolist())
print('pair run flags', starts.astype(int).tolist())
print('pair group index', np.cumsum(starts).tolist())
print('pair boundaries', np.r_[np.flatnonzero(starts), 8].tolist())

keys = np.array([3, 0, 5, 3, 1, 5, 0, 3])  # 5 denotes unused; columns 2 and 4 are empty.
order = np.argsort(keys, kind='stable')
print('slot order', order.tolist(), 'keys', keys[order].tolist())
print('column offsets', np.searchsorted(keys[order], np.arange(6)).tolist())

def binary32_fraction(bits):
    return Fraction.from_float(struct.unpack('!f', struct.pack('!I', bits))[0])

def round_binary32_fraction(value):
    """Binary search the representable numbers, then exact nearest/even choice."""
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)
    maximum = binary32_fraction(0x7f7fffff)
    if value > maximum:
        return sign | (0x7f800000 if value >= maximum + 2**103 else 0x7f7fffff)
    lower, upper = 0, 0x7f7fffff
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if binary32_fraction(middle) <= value:
            lower = middle
        else:
            upper = middle - 1
    if binary32_fraction(lower) == value:
        return sign | lower
    distance_lower = value - binary32_fraction(lower)
    distance_upper = binary32_fraction(lower + 1) - value
    return sign | (lower + int(distance_upper < distance_lower or
                              (distance_upper == distance_lower and lower % 2)))

counts = [16777217, 30000001, 0xffffefff, 3, 0xffffffff, 16777217, 0xffffffff, 30000001]
values = [16777216, 1, 1234567, 8, -0.125, 2**-120, 2**-149, 2**100]
bits = []
for count, value in zip(counts, values):
    value = Fraction.from_float(float(value))  # each fixture value is exactly binary32
    bits.extend(round_binary32_fraction(result) for result in
                [value / count, value * count, Fraction(count - 1, count), count / value])
print('count arithmetic bits', [hex(value) for value in bits])
small_counts = [1, 2, 3, 4, 8, 16, 256, 16777216]
small_values = [1, 1, 8, -0.125, 2**-120, 2**100, 16777216, 1]
print('ordinary count bits', [hex(round_binary32_fraction(result))
    for count, value in zip(small_counts, small_values)
    for result in [Fraction(value) / count, Fraction(value) * count,
                   Fraction(count - 1, count), count / Fraction(value)]])

left_rows = [{1,3},{0},set(),{1,2,7},{5},set(),{0,7},{6}]
right_rows = [{2,3},{2},{0},{2,4},set(),{4},{0},{6}]
for name, operation in [('union', set.union), ('intersection', set.intersection),
                        ('difference', set.difference), ('symmetric-difference', set.symmetric_difference)]:
    rows = [sorted(operation(left, right)) for left, right in zip(left_rows, right_rows)]
    print(name, 'offsets', np.r_[0, np.cumsum(list(map(len, rows)))].tolist(),
          'neighbors', sum(rows, []))

left_weights = [[11,13],[20],[],[41,42,47],[0],[],[70,77],[86]]
right_weights = [[102,5],[112],[120],[132,134],[],[154],[160],[176]]
left_distances = [sorted(row) for row in left_rows]
right_distances = [[value + 10 for value in sorted(row)] for row in right_rows]
rules = {'left': lambda a,b:a, 'right':lambda a,b:b, 'sum':lambda a,b:a+b,
         'min':min, 'max':max, 'product':lambda a,b:a*b, 'binary':lambda a,b:1}
for rule, combine in rules.items():
    values, distances = [], []
    for row in range(8):
        left = dict(zip(sorted(left_rows[row]),left_weights[row]))
        right = dict(zip(sorted(right_rows[row]),right_weights[row]))
        for neighbor in sorted(left.keys() | right.keys()):
            values.append(1 if rule == 'binary' else combine(left[neighbor],right[neighbor])
                          if neighbor in left and neighbor in right else left.get(neighbor,right.get(neighbor)))
            distances.append(neighbor if neighbor in left else neighbor+10)
    print('union',rule,'weights',values)
print('union distances',distances)
self_rows = [set(row) for row in left_rows]
self_rows[1] = {1}
print('self offsets',np.r_[0,np.cumsum([len(row | {index}) for index,row in enumerate(self_rows)])].tolist())
print('self neighbors',sum([sorted(row | {index}) for index,row in enumerate(self_rows)],[]))
print('self weights',sum([[0.5*(index+1) if neighbor==index else
    dict(zip(sorted(left_rows[index]),left_weights[index]))[neighbor]
    for neighbor in sorted(row|{index})] for index,row in enumerate(self_rows)],[]))
print('self distances',sum([[0 if neighbor==index else neighbor
    for neighbor in sorted(row|{index})] for index,row in enumerate(self_rows)],[]))

spatial = np.array([[5,2,0,0],[3,0,0,0],[0,0,0,0],[0,4,0,6]], dtype=float)
space_time = np.zeros((12,12))
for lag, weight, include_self in [(-1,2,True),(0,0.5,False),(1,3,True)]:
    temporal = np.diag(np.ones(3-abs(lag)), k=-lag)
    block = spatial.copy()
    if include_self:
        np.fill_diagonal(block,1)
    space_time += weight * np.kron(temporal,block)
from scipy.sparse import csr_matrix
for layout, order in [('period-major',np.arange(12)),('cell-major',np.arange(12).reshape(3,4).T.flatten())]:
    matrix = csr_matrix(space_time[np.ix_(order,order)])
    print('spaceTime',layout,'offsets',matrix.indptr.tolist(),'neighbors',matrix.indices.tolist(),'weights',matrix.data.tolist())

matrix = np.zeros((8,8))
for row in range(8):
    for column, value in zip(sorted(left_rows[row]), left_weights[row]):
        matrix[row,column] = value
print('construction summary',matrix.sum(),((matrix+matrix.T)**2).sum()/2,
      ((matrix.sum(axis=0)+matrix.sum(axis=1))**2).sum())
