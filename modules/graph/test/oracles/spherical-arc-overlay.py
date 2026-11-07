# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct non-hemisphere S2BooleanOperation references, s2geometry==0.14.0."""
import importlib.metadata,json,math
from pathlib import Path
import s2geometry as s2
assert importlib.metadata.version('s2geometry')=='0.14.0'
def value(latitude,longitude):
 point=s2.S2LatLng.FromDegrees(latitude,longitude).ToPoint();return [point.x(),point.y(),point.z()]
def point(value):return s2.S2LatLng.FromRadians(math.asin(value[2]),math.atan2(value[1],value[0])).ToPoint()
def polygon(vertices,complement):return s2.S2Polygon(s2.S2Loop([point(vertex) for vertex in (vertices[::-1] if complement else vertices)]))
left=[value(latitude,longitude) for latitude,longitude in zip([-15,25,-15,25],[-135,-45,45,135])]
right=[value(latitude,longitude) for latitude,longitude in zip([25,-15,25,-15],[-135,-45,45,135])]
reference=value(90,0);cases=[]
for firstComplement,secondComplement in [(False,False),(True,False),(True,True)]:
 for operation,kind in [('intersection',1),('union',0),('difference',2),('symmetric-difference',3)]:
  first=polygon(left,firstComplement);second=polygon(right,secondComplement);assert first.IsValid() and second.IsValid()
  firstInside=first.Contains(point(reference));secondInside=second.Contains(point(reference))
  firstIndex=s2.MutableS2ShapeIndex();firstIndex.Add(first)
  secondIndex=s2.MutableS2ShapeIndex();secondIndex.Add(second)
  output=s2.S2Polygon();boolean=s2.S2BooleanOperation(kind,s2.S2PolygonLayer(output));assert boolean.Build(firstIndex,secondIndex);assert output.IsValid()
  cases.append({'operation':operation,'leftInside':firstInside,'rightInside':secondInside,'area':output.GetArea(),'rings':output.num_loops(),'referenceInside':output.Contains(point(reference))})
path=Path(__file__).resolve().parents[1]/'gpu-spatial-analysis/global-coordinates/fixtures/spherical-arc-overlay.json'
path.write_text(json.dumps({'provenance':'s2geometry==0.14.0 direct S2BooleanOperation.Build/GetArea/Contains; equatorial zigzag loops cannot fit in one hemisphere chart; reversed S2Loops represent complements','left':left,'right':right,'reference':reference,'cases':cases},indent=2)+'\n')
