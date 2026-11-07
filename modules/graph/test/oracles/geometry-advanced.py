# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct pinned Shapely cases for geometry tree normalization and arbitrary precision grids."""
import json
from pathlib import Path
import numpy as np
import shapely
from shapely import from_wkt
assert shapely.__version__ == '2.1.2'
KINDS = {'Point': 0, 'MultiPoint': 1, 'LineString': 2, 'LinearRing': 3, 'MultiLineString': 4, 'Polygon': 5, 'MultiPolygon': 6, 'GeometryCollection': 7}

def tree(geometry):
    positions, nodes, children = [], [], []
    def visit(part):
        kind = KINDS[part.geom_type]
        if kind in [0, 2, 3]:
            values = np.asarray(part.coords, dtype=np.float32).reshape(-1).tolist()
            first, count = len(positions) // 2, len(values) // 2
            positions.extend(values)
        else:
            parts = ([part.exterior, *part.interiors] if not part.is_empty else []) if kind == 5 else part.geoms
            ids = [visit(child) for child in parts]
            first, count = len(children), len(ids)
            children.extend(ids)
        node = len(nodes) // 4
        nodes.extend([kind, first, count, 1000 + node])
        return node
    root = visit(geometry)
    return dict(positions=positions, nodes=nodes, children=children, root=root)

def signature(geometry):
    kind = KINDS[geometry.geom_type]
    if kind in [0, 2, 3]: return [kind, np.asarray(geometry.coords, dtype=np.float32).reshape(-1).tolist()]
    parts = ([geometry.exterior, *geometry.interiors] if not geometry.is_empty else []) if kind == 5 else geometry.geoms
    return [kind, [signature(child) for child in parts]]

wkts = [
 'GEOMETRYCOLLECTION (POINT (1 2), POINT EMPTY, LINESTRING (4 0, 2 1, 0 0), LINESTRING EMPTY, MULTIPOINT ((0 0), (3 1), (2 5)), GEOMETRYCOLLECTION EMPTY)',
 'GEOMETRYCOLLECTION (MULTILINESTRING ((4 0,3 0),(2 2,0 0,1 0,2 2)), POLYGON ((4 4,0 4,0 0,4 0,4 4),(1 1,1 2,2 2,2 1,1 1)), GEOMETRYCOLLECTION (POINT (5 5), GEOMETRYCOLLECTION (LINESTRING (3 0, 2 0))))',
 'GEOMETRYCOLLECTION (POLYGON EMPTY, MULTIPOLYGON EMPTY, MULTILINESTRING EMPTY, MULTIPOINT EMPTY, POINT EMPTY)',
 'LINESTRING (2 2,0 0,1 0,2 2)', 'LINESTRING (3 0, 2 0, 2 0, 1 0)', 'POINT (2 3)'
]
collections=[]
for wkt in wkts:
    geometry = from_wkt(wkt)
    normalized = shapely.normalize(geometry)
    collections.append(dict(wkt=wkt, input=tree(geometry), canonical=signature(normalized), normalized=tree(normalized)))
precision=[]
for wkt in ['POLYGON ((0.11 0.13,4.11 0.13,4.11 2.13,0.11 2.13,0.11 0.13))', 'POLYGON ((0 0,5 0,5 5,0 5,0 0),(1.1 1.1,1.1 3.9,3.9 3.9,3.9 1.1,1.1 1.1))', 'POLYGON ((0.01 0.01,0.02 0.01,0.02 0.02,0.01 0.02,0.01 0.01))']:
    geometry = shapely.from_wkt(wkt)
    geometry = shapely.transform(geometry, lambda coordinates: coordinates.astype(np.float32).astype(np.float64))
    for grid in [0.1, 0.3, 2.5]:
        for mode in ['valid_output', 'keep_collapsed']:
            result = shapely.set_precision(geometry, float(np.float32(grid)), mode=mode)
            rings = [geometry.exterior, *geometry.interiors]
            coords, offsets = [], [0]
            for ring in rings:
                coords.extend(np.asarray(ring.coords, dtype=np.float32).reshape(-1).tolist()); offsets.append(len(coords)//2)
            precision.append(dict(positions=coords, ringOffsets=offsets, gridSize=grid, mode=mode.replace('_','-'), area=result.area, empty=result.is_empty))
offsets=[]
for coordinates in [[(0,0),(5,0),(5,5)],[(0,0),(4,0),(4,1),(0,1)],[(0,0),(4,4),(0,4),(4,0)],[(0,0),(6,0),(6,4),(3,4),(3,1),(0,1)]]:
 for distance in [-0.5,0.5,1.5]:
  for join in ['mitre','round','bevel']:
   line=shapely.LineString(coordinates)
   result=shapely.offset_curve(line,distance,quad_segs=8,join_style=join)
   parts=[] if result.is_empty else ([result] if result.geom_type=='LineString' else list(result.geoms))
   offsets.append(dict(positions=[value for point in coordinates for value in point],distance=distance,joinStyle=join,expected=[list(part.coords) for part in parts],length=result.length))
result=dict(generator='geometry-advanced.py', versions=dict(shapely=shapely.__version__, geos=shapely.geos_version_string, numpy=np.__version__), collections=collections, precision=precision, offsets=offsets)
(Path(__file__).parent/'fixtures'/'geometry-advanced.json').write_text(json.dumps(result, indent=2)+'\n')
# Coverage for ratio endpoints, degeneracy, disconnected clusters, and circle error contracts.
hulls=[]
for coordinates in [[],[(1,2)],[(0,0),(1,0)],[(0,0),(1,0),(2,0),(1,0)],[(0,0),(1,0),(1,1),(0,1)],[(0,0),(1,0),(0,1),(10,0),(11,0),(11,1)],[(0,0),(4,0),(4,4),(0,4),(1,1),(3,1),(3,3),(1,3),(2,2)]]:
 for ratio in [0,0.5,1]:
  for holes in [False,True]:
   hull=shapely.concave_hull(shapely.MultiPoint(coordinates),ratio=ratio,allow_holes=holes)
   hulls.append(dict(positions=[v for point in coordinates for v in point],ratio=ratio,allowHoles=holes,area=hull.area,geometryType=hull.geom_type))
circles=[]
subdivision=[]
for wkt in ['POLYGON ((0 0,8 0,8 2,2 2,2 8,0 8,0 0))','POLYGON ((0 0,10 0,10 10,0 10,0 0),(3 3,3 7,7 7,7 3,3 3))','POLYGON ((0 0,5 0,7 2,7 5,5 7,2 7,0 5,-1 2,0 0))','POLYGON ((0 0,0.01 0,0.01 8,0 8,0 0))']:
 geometry=from_wkt(wkt)
 coordinates=[];ringOffsets=[0]
 for ring in [geometry.exterior,*geometry.interiors]:
  coordinates.extend(np.asarray(ring.coords,dtype=np.float32).reshape(-1).tolist());ringOffsets.append(len(coordinates)//2)
 line=shapely.maximum_inscribed_circle(geometry,tolerance=0.00001)
 circles.append(dict(positions=coordinates,ringOffsets=ringOffsets,radius=line.length,tolerance=0.002))
 lowerX,lowerY,upperX,upperY=geometry.bounds
 queries=[(lowerX+(column+0.413)*(upperX-lowerX)/29,lowerY+(row+0.317)*(upperY-lowerY)/29) for row in range(29) for column in range(29)]
 subdivision.append(dict(positions=coordinates,ringOffsets=ringOffsets,area=geometry.area,queries=queries,contains=[bool(geometry.covers(shapely.Point(point))) for point in queries]))
result.update(hulls=hulls,circles=circles,subdivision=subdivision)
(Path(__file__).parent/'fixtures'/'geometry-advanced.json').write_text(json.dumps(result, indent=2)+'\n')
