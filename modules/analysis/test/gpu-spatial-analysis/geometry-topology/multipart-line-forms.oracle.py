# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Reproduce: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geometry-topology/multipart-line-forms.oracle.py
from shapely import node, set_precision, polygonize_full
from shapely.geometry import MultiLineString, box
from shapely import wkt, is_valid, is_valid_reason, buffer, shared_paths, hausdorff_distance
from shapely.geometry import Point
from math import dist

lines = MultiLineString([[(0,0),(2,2)],[(0,2),(2,0)],[(5,0),(7,0)]])
print([list(part.coords) for part in node(lines).geoms])
print([list(part.coords) for part in node(set_precision(lines, 1)).geoms])
# The integral coordinates and proper crossing (1,1) already lie on this unit lattice.
# [(0,0)-(1,1), (1,1)-(2,2), (0,2)-(1,1), (1,1)-(2,0), (5,0)-(7,0)]
# First two parts belong to feature 1 (ID 17), third to feature 2 (ID 42); feature 0 is empty.
# Input edges start at vertex rows [0,2,4]; crossings halve the first two edges' parameters.
# Independent chronological loop-erasure derivation: the bowtie walk
# (0,0)->(2,2)->(0,2)->(2,0) visits the proper intersection (1,1) twice.
# Erasing strictly between its first and second visits leaves (0,0),(1,1),(2,0).
# Its outgoing source edge rows are 0 and 2; the final incoming row is 2.
# Two independent straight parts remain (5,0),(7,0) and (10,0),(12,0), source rows
# [4,4] and [6,6]. Thus vertices=7, offsets=[0,3,5,7], features=[0,0,2,3].

triangles = MultiLineString([[(0,0),(2,0)],[(2,0),(0,2)],[(0,2),(0,0)],[(5,0),(7,0)],[(7,0),(5,2)],[(5,2),(5,0)]])
faces, cuts, dangles, invalid = polygonize_full(list(triangles.geoms))
print([polygon.area for polygon in faces.geoms], cuts.is_empty, dangles.is_empty, invalid.is_empty)
# Areas [2,2], all other collections empty. Six source segments each bound a triangle, each
# triangle has two directed face cycles (bounded CCW and unbounded CW): 4 required face rings.
# Each polygon drops its closing duplicate and keeps 3 vertices; all diagnostic edge classes=1,
# closedRingCount=6, other diagnostic counters=0. These are the literals in the shared fixture.

for text in ['MULTIPOINT EMPTY','MULTIPOINT ((0 0),(NaN 1))','MULTIPOINT ((1 2),(1 2))',
             'MULTILINESTRING EMPTY','MULTILINESTRING ((0 0,1 0),(5 0,6 0))',
             'MULTILINESTRING ((2 2,2 2))','MULTILINESTRING ((0 0,NaN 1))']:
    geometry = wkt.loads(text)
    print(text, bool(is_valid(geometry)), is_valid_reason(geometry))
# Multipoint: [true,false,true]. Multiline: [true,true,false,false]; final empty feature true.
# Diagnostic mapping: invalid finite collapsed part -> tooFewPoints=512, nonfinite point -> 1.
surface = buffer(MultiLineString([[(0,0),(2,0)],[(5,0),(7,0)]]), .5, cap_style='flat', join_style='mitre')
print(surface.wkt, surface.area)
print('buffer rings:', sum(1 + len(polygon.interiors) for polygon in surface.geoms))
# Two 2-by-1 rectangles: 8 boundary segments, 8 polygon vertices, total area 4. Both parts
# belong to logical feature row 1; caller feature ID 17 must not replace provenance sourceRows.
# There are exactly 2 output rings (2 shells, 0 holes): ringCount=requiredRingCount=2.
# This is different from polygonize's bounded/unbounded directed face-cycle inventory above.
print(shared_paths(MultiLineString([[(0,0),(2,0)],[(5,0),(7,0)]]),
                   MultiLineString([[(1,0),(2,0)],[(7,0),(6,0)]])))
# Forward (1,0)-(2,0), backward (6,0)-(7,0); both left parts belong to ID 17;
# right logical IDs are 42 and 73, with an empty feature between them. Part rows stay 0,1.
clip_lines = MultiLineString([[(0,0),(2,0)],[(5,0),(7,0)]])
print(clip_lines.intersection(box(1,-1,6,1)))
print(clip_lines.difference(box(1,-1,6,1)))
# Inside: (1,0)-(2,0), (5,0)-(6,0); outside: (0,0)-(1,0), (6,0)-(7,0).
# Parts 0,1 both belong to logical feature 1, ID17; original edge rows 0,2.
track_b=MultiLineString([[(0,0),(1,0)],[(9,0),(10,0)],[(5,0),(5,1)]])
for track_a in [MultiLineString([[(0,0),(1,0)],[(9,0),(10,0)]]),MultiLineString([[(0,2),(1,2)]])]:
    first=[tuple(vertex) for part in track_a.geoms for vertex in part.coords]
    second=[tuple(vertex) for part in track_b.geoms for vertex in part.coords]
    sspd=(sum(Point(vertex).distance(track_b) for vertex in first)/len(first)+sum(Point(vertex).distance(track_a) for vertex in second)/len(second))/2
    print('tracks:',hausdorff_distance(track_a,track_b,densify=.5),sspd,max(dist(a,b) for a in first for b in second))
# A false connector between x=1 and x=9 reduces the first Hausdorff distance to 1;
# the correct disconnected union has sqrt(17). SSPD uses every original vertex, including
# duplicated part endpoints; empty feature status=1, invalid pair/inactive row status=4.
for track_a in [MultiLineString([[(0,0),(1,0)],[(9,0),(10,0)]]),MultiLineString([[(0,2),(1,2)]])]:
    first=[tuple(vertex) for part in track_a.geoms for vertex in part.coords]
    second=[tuple(vertex) for part in track_b.geoms for vertex in part.coords]
    def box_distance(vertex,bounds):
        return dist((max(bounds[0]-vertex[0],0,vertex[0]-bounds[2]),max(bounds[1]-vertex[1],0,vertex[1]-bounds[3])),(0,0))
    forward=[box_distance(vertex,track_b.bounds) for vertex in first]
    backward=[box_distance(vertex,track_a.bounds) for vertex in second]
    print('search bounds:',max(forward+backward),(sum(forward)/len(first)+sum(backward)/len(second))/2*(1-(len(first)+len(second))*1.2e-7))
# H bounds [1,sqrt85], threshold5; SSPD bounds [.0833332333,3.0050802225], threshold1.
# One survivor, first pair matches; second pruned, empty/invalid pairs status2.
# topK=1: feature0 [none,incomplete], feature1 [pair0,complete], feature2 [none,complete].
for text in ['MULTIPOINT EMPTY','MULTILINESTRING EMPTY']:
    geometry=wkt.loads(text)
    print('empty:',geometry.length,geometry.centroid.wkt,geometry.bounds,geometry.is_simple,geometry.equals_exact(geometry,0),len(geometry.geoms))
# Every empty feature: length0, vertex/part counts0, NaN centroid and all four bounds,
# simple=true, exact/identical self comparison=true, other unary flags=false.
# Group featureCounts count logical rows, even when empty: IDs [1,0] -> [1,1,0].
