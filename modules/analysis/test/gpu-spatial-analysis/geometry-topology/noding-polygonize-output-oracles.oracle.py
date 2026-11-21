# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Reproduce: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geometry-topology/noding-polygonize-output-oracles.oracle.py
from shapely import node, polygonize_full, set_precision
from shapely.geometry import MultiLineString

crossing = MultiLineString([[(2, 2), (10, 2)], [(4, 0), (4, 8)]])
for geometry in [crossing, set_precision(crossing, 0.5)]:
    print("quarter crossing", [list(part.coords) for part in node(geometry).geoms])
# Both sources meet at parameter 1/4: (4-2)/(10-2) = (2-0)/(8-0).
# Source IDs 70,80, line rows 0,1 and start-vertex edge rows 0,2.
# Requested grid .3 rounds up to .5; origin floor(minimum/.5)*.5 = (2,0).

joined = MultiLineString([
    [(0,0),(1,0),(1,1),(0,1),(0,0)],
    [(3,0),(4,0),(4,1),(3,1),(3,0)],
    [(1,0),(3,0)],
])
for geometry in [joined, set_precision(joined, 0.5)]:
    faces, cuts, dangles, invalid = polygonize_full(list(node(geometry).geoms))
    print("faces", [face.area for face in faces.geoms])
    print("cuts", [list(part.coords) for part in cuts.geoms])
    print("dangles", len(dangles.geoms), "invalid", len(invalid.geoms))
# Nine source edges, no interior splits: 8 closedRing (1), one cut edge
# openChain|cut|unused = 2|4|16 = 22; diagnostics count edges, not rings.
# The planar graph is connected: two bounded left faces and one outer walk.
# All vertices have degree 2 except the two degree-3 bridge attachment vertices.

for label, lines in [
    ('inward spur', [[(0,0),(4,0),(4,4),(0,4),(0,0)],[(0,0),(1,1)]]),
    ('short terminal spur', [[(0,0),(4,0),(4,4),(0,4),(0,0)],[(0,0),(1,1),(1+1/128,1+1/128)]]),
    ('open chain', [[(0,0),(1,0),(2,0),(3,0)]])
]:
    # Atomic edges make diagnostic counts comparable with the contributor's source rows.
    segments = [edge for line in lines for edge in zip(line, line[1:])]
    faces, cuts, dangles, invalid = polygonize_full(list(node(MultiLineString(segments)).geoms))
    print(label, 'areas', [face.area for face in faces.geoms],
          'cuts', len(cuts.geoms), 'dangles', len(dangles.geoms), 'invalid', len(invalid.geoms))
# Spur: [16], 0 cuts, 1 dangle -> [1,1,1,1,26]. Chain: [], 0 cuts, 3 dangles -> [26,26,26].
off_grid = MultiLineString([[(0,0),(6,2)],[(2,0),(2,2)]])
print('off-grid snap', [list(part.coords) for part in node(set_precision(off_grid, .5)).geoms])
# The intersection on the .5 grid is (2,.5); source projections are 13/40 and 1/4.

ordinary = MultiLineString([[(0,0),(12,4)],[(9,3),(9,5)]])
print('ordinary exact split', [list(part.coords) for part in node(ordinary).geoms])
# Projection: ((9,3) dot (12,4)) / ((12,4) dot (12,4)) = 120/160 = 3/4.
import numpy as np
direction = np.array([12,4], dtype=np.float32)
point = np.array([9,3], dtype=np.float32)
print('ordinary projection', np.dot(point,direction)/np.dot(direction,direction))
print('ordinary arbitrary scaling', np.dot(point/12,direction/12)/np.dot(direction/12,direction/12))

# With tolerance 1/8 the square endpoints (0,0) and (0,3/32) match directly.
# The exterior endpoint (0,-3/32) also matches (0,0), but not the other end.
# Close the square at its first vertex for the independent polygon oracle;
# the exterior edge is a dangle for both permutations, independent of row IDs.
square = [(0,0),(4,0),(4,4),(0,4),(0,0)]
exterior = [(-1,-1),(0,0)]
for lines in [[square,exterior],[exterior,square]]:
    segments = [edge for line in lines for edge in zip(line,line[1:])]
    faces,cuts,dangles,invalid=polygonize_full(list(node(MultiLineString(segments)).geoms))
    print('tolerance-closed square permutation', 'areas', [face.area for face in faces.geoms],
          'cuts', len(cuts.geoms), 'dangles', len(dangles.geoms), 'invalid',len(invalid.geoms))

# Prior pruning indexed only x: a column contains four directed starts per
# square. Each of 8*N starts scans 4*N candidates, regardless of y separation.
for square_count in [8,32,4096]:
    print('old x-only visits', square_count, 'per endpoint',4*square_count,'total',32*square_count*square_count)
# A 2-D neighbor-cell index visits only the two coincident starts of each corner.
