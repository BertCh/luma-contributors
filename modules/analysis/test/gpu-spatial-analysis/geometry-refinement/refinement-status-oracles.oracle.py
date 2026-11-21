# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geometry-refinement/refinement-status-oracles.oracle.py
from scipy.spatial import Delaunay

sites = [(0, 0), (6, 0), (0, 4), (1, 1)]
triangles = sorted(tuple(sorted(triangle)) for triangle in Delaunay(sites).simplices)
assert triangles == [(0, 1, 3), (0, 2, 3), (1, 2, 3)]
print('independent Delaunay triangles', triangles)
exteriorEdges = [(0, 1), (0, 2), (1, 2)]
lengthsSquared = [sum((sites[first][axis] - sites[second][axis]) ** 2 for axis in [0, 1])
                  for first, second in exteriorEdges]
assert lengthsSquared == [36, 16, 52]
# All three candidates share interior site 3. The unique largest boundary edge
# wins at all incident vertices, so only triangle {1,2,3} is removed. Site 3
# then becomes a boundary vertex and neither surviving triangle is removable.
print('exterior squared lengths', lengthsSquared, 'unique removal: (1,2,3)')
print('budget 0: validity by identity [1,1,1], converged 0')
print('budget 1: validity by identity [1,1,0], converged 1')
