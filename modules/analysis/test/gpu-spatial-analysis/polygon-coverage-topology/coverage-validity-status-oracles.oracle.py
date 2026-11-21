# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/polygon-coverage-topology/coverage-validity-status-oracles.oracle.py
# Independent edge predicates for these two scenes; no production flag constants are imported.
from shapely.geometry import LineString

scenes = {
    "crossing": [[[2, 0], [2, 2], [0, 2], [0, 0]],
                 [[3, 1], [3, 3], [1, 3], [1, 1]]],
    "mismatch": [[[1, 0], [1, 1], [0, 1], [0, 0]],
                 [[1, 0], [2, 0], [2, 1], [1, 1], [1, 0.5]]],
}
for name, rings in scenes.items():
    edges = [[LineString([point, ring[(row + 1) % len(ring)]])
              for row, point in enumerate(ring)] for ring in rings]
    flags = []
    for feature, feature_edges in enumerate(edges):
        for edge in feature_edges:
            others = edges[1 - feature]
            if any(edge.crosses(other) for other in others):
                flags.append(2)  # Proper interior crossing: public bit 2.
            elif any(edge.intersection(other).length > 0 and not edge.equals(other)
                     for other in others):
                flags.append(1)  # Collinear partial match: public unmatched bit 1.
            else:
                flags.append(0)
    print(name, flags)
    assert flags == ([2, 2, 0, 0, 0, 0, 2, 2] if name == "crossing"
                     else [1, 0, 0, 0, 0, 0, 0, 1, 1])
    print("swapped-bit mutant rejected:", flags != [{1: 2, 2: 1}.get(flag, flag) for flag in flags])
gap = [0, 0, 4, 0, 4, 0, 0, 0]
print("old gap-only oracle accepts swapped-bit mutant:", gap == [{1: 2, 2: 1}.get(flag, flag) for flag in gap])
