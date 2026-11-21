# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run: /usr/bin/python3 -B modules/analysis/test/gpu-spatial-analysis/spatial-weights/audit-weights-a-review.oracle.py"""

import numpy as np
from scipy.sparse.csgraph import shortest_path

# R3: all reachable other rows are already direct neighbors; exact distance two is empty.
adjacency = np.array([[0, 1, 1], [0, 0, 1], [0, 1, 0]])
distances = shortest_path(adjacency, directed=True, unweighted=True)
print("R3 exact-order-two counts", (distances == 2).sum(axis=1).tolist())
# A one-slot initial work prefix retains only 0->1, forgetting that 0->2 was direct.
# The retained computation can rediscover 0->2 through row 1; its count 1 exceeds exact 0.

# R8: scaled row norms prevent float32 squares overflowing. Q=sqrt(2)+1, n=3.
print("R8 variance", [3 / ((np.sqrt(2)+1)*np.sqrt(2))]*2 + [3/(np.sqrt(2)+1)])
print("R8 nonfinite double total", [0,0,0])

# R12: uint64 sums distinguish the true prefix from its wrapped uint32 low word.
prefixes = np.r_[np.uint64(0), np.cumsum(np.array([2**32-2,3,7],dtype=np.uint64))]
print("R12 exact prefixes", prefixes.tolist())
print("R12 clamped prefixes", np.minimum(prefixes,4).tolist())
