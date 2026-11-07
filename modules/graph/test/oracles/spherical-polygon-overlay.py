# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct S2BooleanOperation oracle. s2geometry==0.14.0, no planar overlay reference."""
import importlib.metadata
import json
import math
from pathlib import Path
import s2geometry as s2
assert importlib.metadata.version('s2geometry') == '0.14.0'
def direction(x,y):
    scale=math.sqrt(1+x*x+y*y)
    return [x/scale,y/scale,1/scale]
def point(value):
    return s2.S2LatLng.FromRadians(math.asin(value[2]),math.atan2(value[1],value[0])).ToPoint()
def polygon(rings):
    loops=[s2.S2Loop([point(value) for value in ring]) for ring in rings]
    for loop in loops: loop.Normalize()
    result=s2.S2Polygon(); result.InitNested(loops); assert result.IsValid(); return result
left=[[direction(x,y) for x,y in ring] for ring in [[[-.8,-.8],[.8,-.8],[.8,-.1],[.1,-.1],[.1,.8],[-.8,.8]],[[-.6,.1],[-.3,.1],[-.3,.4],[-.6,.4]]]]
right=[[direction(x,y) for x,y in [[-.5,-.6],[.6,-.6],[.6,.6],[-.5,.6]]]]
cases=[]
for operation,kind in [('intersection',s2.S2BooleanOperation.OpType_INTERSECTION),('union',s2.S2BooleanOperation.OpType_UNION),('difference',s2.S2BooleanOperation.OpType_DIFFERENCE),('symmetric-difference',s2.S2BooleanOperation.OpType_SYMMETRIC_DIFFERENCE)]:
    first=s2.MutableS2ShapeIndex(); first.Add(polygon(left))
    second=s2.MutableS2ShapeIndex(); second.Add(polygon(right))
    output=s2.S2Polygon(); boolean=s2.S2BooleanOperation(kind,s2.S2PolygonLayer(output)); assert boolean.Build(first,second)
    assert output.IsValid()
    cases.append({'operation':operation,'area':output.GetArea(),'rings':[[[output.loop(ring).vertex(vertex).x(),output.loop(ring).vertex(vertex).y(),output.loop(ring).vertex(vertex).z()] for vertex in range(output.loop(ring).num_vertices())] for ring in range(output.num_loops())]})
path=Path(__file__).resolve().parents[1]/'gpu-spatial-analysis/global-coordinates/fixtures/spherical-polygon-overlay.json'
path.write_text(json.dumps({'provenance':'s2geometry==0.14.0 direct S2BooleanOperation.Build/GetArea, nonconvex shell plus hole','left':left,'right':right,'cases':cases},indent=2)+'\n')
