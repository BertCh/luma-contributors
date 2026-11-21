# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geometry-refinement/precision-output-oracles.oracle.py
from shapely import Polygon, MultiLineString, LineString, set_precision, normalize, node, polygonize_full, union_all
import numpy as np

# Explicit closures avoid any ambiguity about closing-edge source row identities.
for corners in [[(.2, .2), (4.2, .2), (4.2, 2.2), (.2, 2.2)],
                [(10.2, 4.2), (13.2, 4.2), (13.2, 7.2), (10.2, 7.2)]]:
    result = normalize(set_precision(Polygon(corners), 1))
    print(list(result.exterior.coords), result.area)
    assert result.area in (8, 9)

# R2 uses exactly the stored f32 coordinates AND step, rather than decimal real inputs.
coordinates = np.array([[.35,0],[.41,0],[.41,.61],[.35,.61]], dtype=np.float32)
step = float(np.float32(.1))
result = set_precision(Polygon(coordinates.astype(float)), step)
print('R2', result.bounds, result.area, 'quotient',float(coordinates[0,0])/step)
assert len(result.exterior.coords) == 5
assert np.allclose(result.bounds, [3*step,0,4*step,6*step], rtol=0, atol=1e-12)

# Review 2 R4: the unperturbed extent is exactly the supported lattice width.
wide = np.array([[-158750.609375,.1],[260679.796875,.1],
                 [260679.796875,.2],[-158750.609375,.2]], dtype=np.float32)
result = set_precision(Polygon(wide.astype(float)), step)
print('review 2 R4', result.bounds, result.area, 'width steps', (float(wide[1,0])-float(wide[0,0]))/step)
assert (float(wide[1,0])-float(wide[0,0]))/step == 2**22
assert len(result.exterior.coords) == 5
assert abs(result.area - 41943.04125000001) < 1e-7

# Polygonize removes degree-one fringes recursively. Preserve source multiplicity
# when mapping GEOS's unique geometric edges back to contributor diagnostics.
for name, walk, expected_classes, expected_counts in [
    ('peeled chain', [(0,0),(1,0),(2,0),(3,0),(2,0),(1,0),(0,0)],
     [26,26,26,26,26,26], [0,0,6]),
    ('anchored bridge', [(1,0),(1,1),(0,1),(0,0),(1,0),(3,0),(4,0),(4,1),(3,1),(3,0),(1,0)],
     [1,1,1,1,22,1,1,1,1,22], [8,2,0]),
]:
    source_segments = [LineString([first,last]) for first,last in zip(walk,walk[1:])]
    faces,cuts,dangles,invalid = polygonize_full(list(node(MultiLineString(source_segments)).geoms))
    boundaries = union_all([polygon.boundary for polygon in faces.geoms])
    classes = [1 if boundaries.covers(segment) else 22 if cuts.covers(segment) else
               26 if dangles.covers(segment) else 16 for segment in source_segments]
    counts = [classes.count(1), classes.count(22), classes.count(26)]
    print(name, 'faces',len(faces.geoms),'unique cuts',len(cuts.geoms),
          'unique dangles',len(dangles.geoms),'source classes',classes,'source counts',counts)
    assert classes == expected_classes
    assert counts == expected_counts
    assert len(invalid.geoms) == 0
