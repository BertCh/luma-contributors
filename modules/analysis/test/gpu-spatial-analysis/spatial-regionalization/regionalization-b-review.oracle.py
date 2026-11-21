# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regionalization/regionalization-b-review.oracle.py
"""Independent NumPy centered SSD / closed-form review counterexamples."""
import numpy as np

def lane_pairwise_total(values):
    """Reproduce only the f32 reduction under review; the oracle remains float64 z-scores."""
    count = len(values)
    lane_terms = (count + 255) // 256
    plane = np.zeros((1 << (lane_terms - 1).bit_length(), 256), dtype=np.float32)
    plane.flat[:count] = values
    while len(plane) > 1:
        plane = np.float32(plane[::2] + plane[1::2])
    lanes = plane[0].copy()
    for stride in [128, 64, 32, 16, 8, 4, 2, 1]:
        lanes[:stride] = np.float32(lanes[:stride] + lanes[stride:2*stride])
    return lanes[0]

for count, balanced in [(5596622, False), (30000, True)]:
    sample = np.full(count, np.float32(0.3 if balanced else 0.1))
    sample[0] = 0
    if balanced:
        sample[1] = np.float32(0.5999)
    exact = sample.astype(np.float64)
    scores = (exact-exact.mean()) / exact.std(ddof=0)
    mean = np.float32(lane_pairwise_total(sample) / np.float32(count))
    residual = np.float32(sample - mean)
    deviation = np.float32(np.sqrt(lane_pairwise_total(np.float32(residual*residual))/np.float32(count)))
    print('review 2 R3', count, 'mean', float(mean), 'old common z', float(np.float32(residual[-1]/deviation)), 'oracle scores', scores[:3])
    correction = np.float32(lane_pairwise_total(residual) / np.float32(count))
    corrected = np.float32(residual-correction)
    corrected_deviation = np.float32(np.sqrt(lane_pairwise_total(np.float32(corrected*corrected))/np.float32(count)))
    print('review 2 R3 corrected common z', float(np.float32(corrected[-1]/corrected_deviation)))

def ssd(values):
    return float(np.sum((values - values.mean()) ** 2))

for values in [[1000000, 0, 2/128, 3/128], [2**60, 0, 2, 3]]:
    values = np.array(values, dtype=np.float32).astype(np.float64)
    print('review 2 R1 singleton edge SSD', [ssd(values[index:index+2]) for index in range(3)])

for shift in [0, 1000000]:
    values = (np.array([0, 1/16, 3/16, 6/16]) + shift).astype(np.float32).astype(float)
    print('R1 shift', shift, 'costs', [ssd(values[:3]) - ssd(values[:2]), ssd(values[2:])],
          'objective', ssd(values[:3]))

for count in [65536, 1048576]:
    values = np.full(count, np.float32(0.1), dtype=np.float32)
    values[0] = 0
    lanes = np.zeros(256, dtype=np.float32)
    for row in range(0, count, 256):
        lanes = np.float32(lanes + values[row:row+256])
    for stride in [128, 64, 32, 16, 8, 4, 2, 1]:
        lanes[:stride] = np.float32(lanes[:stride] + lanes[stride:2*stride])
    mean = np.float32(lanes[0] / np.float32(count))
    squares = np.zeros(256, dtype=np.float32)
    for row in range(0, count, 256):
        differences = np.float32(values[row:row+256] - mean)
        squares = np.float32(squares + np.float32(differences * differences))
    for stride in [128, 64, 32, 16, 8, 4, 2, 1]:
        squares[:stride] = np.float32(squares[:stride] + squares[stride:2*stride])
    deviation = np.float32(np.sqrt(np.float32(squares[0] / np.float32(count))))
    print('R3 count', count, 'closed form', 1 / np.sqrt(count-1),
          'old common z', np.float32((values[1]-mean)/deviation))

extreme = np.array([-1e19, 1e19], dtype=np.float32).astype(float)
print('R4 centered SSD', ssd(extreme), 'float32 gain', float(np.float32(ssd(extreme))))
two_columns=np.array([[2e38,-1e19],[2e38,1e19]],dtype=np.float32).astype(float)
column_ssd=np.sum((two_columns-two_columns.mean(axis=0))**2,axis=0)
print('review 2 R4 column SSD',column_ssd.tolist(),'gain',float(np.float32(column_ssd.sum())))

scores = [-2, -2, 3, 0, -2, 10, -2, 0]
keys = [9, 4, 0, 0, 4, 0, 4, 0]
for values in [[7, 6, 4, 2, 5, 3, 1, 0], [0, 6, 4, 2, 5, 3, 1, 0]]:
    winner = min(zip(scores, keys, values))[2]
    without_key = min(zip(scores, values))[1]
    print('R7 fixture', values, 'correct', winner, 'ignoring key', without_key)
old_six = [9, 8, -1, 7, 6, 5]
for stride in [3, 1]:
    for lane in range(stride):
        old_six[lane] = min(old_six[lane], old_six[lane+stride])
print('R7 old six-lane reduction', old_six[0], 'independent minimum', -1)
