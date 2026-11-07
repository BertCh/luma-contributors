# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct S2Polygon.Contains oracle, s2geometry==0.14.0; analytic synthetic shapes."""
import importlib.metadata
import json
import math
import random
from pathlib import Path
import s2geometry as s2
assert importlib.metadata.version('s2geometry') == '0.14.0'

def direction(x, y, axis):
    scale = math.sqrt(1 + x*x + y*y)
    value = [x/scale, y/scale, 1/scale]
    return value if axis == 'pole' else [-value[2], value[1], value[0]]
def point(value):
    return s2.S2LatLng.FromRadians(math.asin(value[2]), math.atan2(value[1], value[0])).ToPoint()
shapes = [
    [[[-.8,-.8],[.8,-.8],[.8,-.1],[.1,-.1],[.1,.8],[-.8,.8]], [[-.6,.1],[-.3,.1],[-.3,.4],[-.6,.4]]],
    [[[-.9,-.9],[.9,-.9],[.9,.9],[-.9,.9]], [[-.6,-.6],[-.2,-.6],[-.2,-.2],[-.6,-.2]], [[.2,.2],[.6,.2],[.6,.6],[.2,.6]]]
]
randomizer = random.Random(7301)
cases = []
for axis in ['pole', 'antimeridian']:
    for rings in shapes:
        vertices = [[direction(x,y,axis) for x,y in ring] for ring in rings]
        loops = [s2.S2Loop([point(value) for value in ring]) for ring in vertices]
        for loop in loops: loop.Normalize()
        polygon = s2.S2Polygon()
        polygon.InitNested(loops)
        assert polygon.IsValid()
        queries = [direction(randomizer.uniform(-1.2,1.2), randomizer.uniform(-1.2,1.2),axis) for _ in range(128)]
        queries += [[-component for component in direction(0,0,axis)]]
        cases.append({'rings':vertices, 'queries':queries, 'contains':[int(polygon.Contains(point(value))) for value in queries], 'area':polygon.GetArea()})
output=Path(__file__).resolve().parents[1]/'gpu-spatial-analysis/global-coordinates/fixtures/spherical-polygon-contains.json'
output.write_text(json.dumps({'provenance':'s2geometry==0.14.0 direct S2Polygon.InitNested/Contains/GetArea; nonconvex shells and holes at poles and antimeridian; boundary avoided for semi-open S2 comparison','cases':cases},indent=2)+'\n')
