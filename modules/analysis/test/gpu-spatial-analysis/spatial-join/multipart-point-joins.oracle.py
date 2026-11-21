# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-join/multipart-point-joins.oracle.py
# Shapely is independent of the production WGSL and host helpers. Half-open probes use a
# small (+epsilon^2,+epsilon) displacement; all non-boundary fixture distances exceed 0.1.
import json
from shapely.geometry import Polygon, Point, LineString, MultiPoint, MultiLineString

polygons = [Polygon([(0,0),(4,0),(4,4),(0,4)], [[(1,1),(2,1),(2,2),(1,2)]]),
            Polygon([(4,0),(8,0),(8,4),(4,4)]), Polygon([(8,0),(12,0),(12,4),(8,4)]),
            Polygon([(0,4),(4,4),(4,8),(0,8)]), Polygon([(-4,0),(0,0),(0,4),(-4,4)]),
            Polygon([(4,4),(8,4),(8,8),(4,8)])]
points = [[(.5,.5),(5,1)], [(4,2),(4,3)], [(20,20)], [], [(1.5,1.5),(.5,3)],
          [(-2,2),(10,2)], [(2,5),(5,5)], [(4,4)], [(12,2)], [(0,0)],
          [(2,2),(3,3),(3,3)], [(1,1)], [(6,1),(6,7),(30,30)]]
lines = [[ [(-3,2.5),(11,2.5)], [(1,3),(3,3)] ], [[(20,20),(21,21)]], [],
         [[(.5,.5),(.5,3.5)], [(4.5,.5),(7.5,.5)]], [[(1.5,.5),(1.5,3.5)]],
         [[(-2,1),(-2,3)]], [[(9,1),(11,3)]], [[(1,5),(3,7)], [(5,5),(7,7)]],
         [[(3,2),(5,2)],[(7,2),(9,2)]], [[(2,3),(2,5)]], [[(6,3),(6,7)]],
         [[(-5,1),(-3,1)]], [[(.5,.5),(3,.5)],[(.5,3.5),(3,3.5)]]]
for rule in ['closed','open','half-open']:
    masks = [[int(any(p.intersects(Point(x,y)) if rule == 'closed' else
                          p.contains(Point(x,y)) if rule == 'open' else
                          p.contains(Point(x+1e-12,y+1e-6)) for x,y in feature))
              for p in polygons] for feature in points]
    print(rule, json.dumps(dict(mask=sum(masks, []), assignments=[next((i for i,v in enumerate(row) if v),4294967295) for row in masks], counts=[sum(row[i] for row in masks) for i in range(6)], sums=[sum((f+1)*row[i] for f,row in enumerate(masks)) for i in range(6)])))
segments = [LineString([(0,0),(12,0)]), LineString([(-4,6),(8,6)])]
distances = [[min(Point(part).distance(segment) for part in feature) if feature else -1 for segment in segments] for feature in points]
print('buffer', json.dumps(dict(distances=[min(row) if 0 <= min(row) <= 1 else -1 for row in distances], rows=[row.index(min(row)) if 0 <= min(row) <= 1 else 4294967295 for row in distances])))
lengths = [[sum(LineString(part).intersection(p).length for part in feature) for p in polygons] for feature in lines]
print('lines',json.dumps(dict(lengths=[sum(row[i] for row in lengths) for i in range(6)], weighted=[sum((f+2)*row[i] for f,row in enumerate(lengths)) for i in range(6)], segments=[sum(LineString(part).intersection(p).length > 0 for feature in lines for part in feature) for p in polygons])))
