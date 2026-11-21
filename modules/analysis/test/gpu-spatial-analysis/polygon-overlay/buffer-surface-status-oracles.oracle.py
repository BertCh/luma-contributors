# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/polygon-overlay/buffer-surface-status-oracles.oracle.py
# Independent GEOS union retains the stroke/cap junctions at x=0 and x=2.
from shapely.geometry import box
from shapely.ops import unary_union

polygon = unary_union([box(-1, -1, 0, 1), box(0, -1, 2, 1), box(2, -1, 3, 1)])
coordinates = list(polygon.exterior.coords)
assert len(coordinates) == 9
print({"boundaryRequiredCount": len(coordinates) - 1,
       "closedRingPositions": len(coordinates),
       "overflowAtCapacities8And9": [int(len(coordinates) > capacity) for capacity in [8, 9]]})
