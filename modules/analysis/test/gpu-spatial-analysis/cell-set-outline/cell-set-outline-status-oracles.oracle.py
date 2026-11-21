# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/cell-set-outline/cell-set-outline-status-oracles.oracle.py
from shapely.geometry import LineString
from shapely.ops import polygonize

# Resolution-2 tiles (1,1) and (3,1) are separated by tile (2,1).
# Axis-aligned affine coordinates preserve their connectivity and ring counts.
edges = []
for west in [1, 3]:
    corners = [(west, 1), (west + 1, 1), (west + 1, 2), (west, 2)]
    edges.extend(LineString([corners[index], corners[(index + 1) % 4]]) for index in range(4))
assert len(list(polygonize(edges))) == 2
assert len(list(polygonize(edges[:4]))) == 1
print('full outline: 8 edges, 2 rings; retained prefix: 4 edges, 1 closed ring')
