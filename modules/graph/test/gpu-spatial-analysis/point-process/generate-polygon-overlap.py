# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Pinned Shapely 2.1.2 exact planar overlay reference for a 25-hole window."""
import json
from pathlib import Path
import shapely
from shapely.geometry import Polygon
from shapely.affinity import translate
assert shapely.__version__ == '2.1.2'
outer = [(0,0),(20,0),(20,20),(0,20),(0,0)]
holes = [[(2+3*x,2+3*y),(2+3*x,3+3*y),(3+3*x,3+3*y),(3+3*x,2+3*y),(2+3*x,2+3*y)] for y in range(5) for x in range(5)]
window = Polygon(outer, holes)
positions = [(.25,.25),(.5,.375),(1.25,.25),(3.25,.5),(17.25,.25),(19.25,.25)]
radii = [.3,1,3.1,19.1]
values = []
for radius in radii:
    total = 0.0
    for first in positions:
        for second in positions:
            if first == second:
                continue
            horizontal, vertical = second[0]-first[0], second[1]-first[1]
            if horizontal**2+vertical**2 <= radius**2:
                total += 1/window.intersection(translate(window,horizontal,vertical)).area
    values.append(total)
Path(__file__).with_name('polygon-overlap-fixture.json').write_text(json.dumps(dict(reference=dict(shapely=shapely.__version__,method='Polygon.intersection(affinity.translate) exact planar overlay; independent ordered pair sum'),vertices=outer+sum(holes,[]),ringOffsets=list(range(0,131,5)),polygonOffsets=[0,26],positions=positions,radii=radii,k=values),indent=2)+'\n')
