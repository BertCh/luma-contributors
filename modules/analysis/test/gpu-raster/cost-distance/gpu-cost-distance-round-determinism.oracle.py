# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Exact zero-cost oracle for the diagnostic's 64 x 257 serpentine.

Run: /usr/bin/python3 modules/analysis/test/gpu-raster/cost-distance/gpu-cost-distance-round-determinism.oracle.py

All traversable cells have friction zero. Every admissible move therefore costs zero,
including diagonals. Connectivity supplies a zero-cost path and nonnegative edge costs
supply the matching lower bound. Barriers stay at infinity. This independent flood checks
that the fixture has no disconnected traversable islands; it does not emulate relaxation.
"""

from collections import deque
import hashlib
import heapq
import json
import numpy as np

width, height = 64, 257
traversable = set()
for row in range(height):
    if row % 2 == 0:
        traversable.update(range(row * width, (row + 1) * width))
    else:
        connector = width - 1 if (row // 2) % 2 == 0 else 0
        traversable.add(row * width + connector)

reached = {0}
frontier = deque([0])
while frontier:
    cell = frontier.popleft()
    row, column = divmod(cell, width)
    for row_delta in (-1, 0, 1):
        for column_delta in (-1, 0, 1):
            neighbor_row = row + row_delta
            neighbor_column = column + column_delta
            if not (0 <= neighbor_row < height and 0 <= neighbor_column < width):
                continue
            neighbor = neighbor_row * width + neighbor_column
            if neighbor in traversable and neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)

assert reached == traversable
assert width * height - 1 in reached
print(json.dumps({
    "finiteCount": len(reached),
    "barrierCount": width * height - len(traversable),
    "firstCost": 0,
    "lastCost": 0,
}, sort_keys=True))

# Original unit-friction spiral, independently integrated with correctly rounded float32
# edge costs and sums. The digest covers the entire row-major surface, barriers included.
size = 256
passable = np.ones(size * size, dtype=bool)
for row in range(size):
    for column in range(size):
        ring = min(row, column, size - 1 - row, size - 1 - column)
        if ring % 8 < 4:
            continue
        gap_left = (ring // 8) % 2 == 0
        in_gap = abs(row - size // 2) < 2 and (column < size // 2 if gap_left else column >= size // 2)
        if not in_gap:
            passable[row * size + column] = False
costs = np.full(size * size, np.inf, dtype=np.float32)
costs[0] = 0
frontier_heap = [(0.0, 0)]
while frontier_heap:
    cost, cell = heapq.heappop(frontier_heap)
    if cost != float(costs[cell]):
        continue
    row, column = divmod(cell, size)
    for row_delta in (-1, 0, 1):
        for column_delta in (-1, 0, 1):
            if row_delta == column_delta == 0:
                continue
            neighbor_row, neighbor_column = row + row_delta, column + column_delta
            if not (0 <= neighbor_row < size and 0 <= neighbor_column < size):
                continue
            neighbor = neighbor_row * size + neighbor_column
            if not passable[neighbor]:
                continue
            edge = np.float32(np.sqrt(2.0) if row_delta and column_delta else 1.0)
            candidate = np.float32(np.float32(cost) + edge)
            if candidate < costs[neighbor]:
                costs[neighbor] = candidate
                heapq.heappush(frontier_heap, (float(candidate), neighbor))
assert int(np.isfinite(costs).sum()) == int(passable.sum()) == 33984
print(json.dumps({
    "fixture": "original-256-spiral",
    "finiteCount": int(np.isfinite(costs).sum()),
    "float32SurfaceSha256": hashlib.sha256(costs.astype('<f4').tobytes()).hexdigest(),
    "checkpoints": {cell: float(costs[cell]) for cell in
                    (0, 1, 255, 256, 65280, 65535, 32896, 32639, 32888, 33124, 1028)},
}, sort_keys=True))
