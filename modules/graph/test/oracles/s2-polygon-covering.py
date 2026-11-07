# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Pinned direct S2 polygon/cell containment for adaptive cover guarantees."""
import importlib.metadata
import json
import math
from pathlib import Path
import s2geometry as s2
assert importlib.metadata.version('s2geometry')=='0.14.0'
root=Path(__file__).resolve().parents[1]/'gpu-spatial-analysis/global-coordinates/fixtures'
source=json.loads((root/'spherical-polygon-contains.json').read_text())
def point(value):return s2.S2LatLng.FromRadians(math.asin(value[2]),math.atan2(value[1],value[0])).ToPoint()
def words(value):return [value&0xffffffff,value>>32]
cases=[]
for case in source['cases'][:2]:
 loops=[s2.S2Loop([point(value) for value in ring]) for ring in case['rings']]
 for loop in loops:loop.Normalize()
 polygon=s2.S2Polygon();polygon.InitNested(loops)
 level=5; cells=[]; identifier=s2.S2CellId.Begin(level);end=s2.S2CellId.End(level)
 while identifier.id()!=end.id():
  cell=s2.S2Cell(identifier)
  cells.append({'cell':words(identifier.id()),'contained':polygon.Contains(cell),'intersects':polygon.MayIntersect(cell),'area':cell.ExactArea()})
  identifier=identifier.next()
 cases.append({'rings':case['rings'],'level':level,'cells':cells,'area':polygon.GetArea()})
(root/'s2-polygon-covering.json').write_text(json.dumps({'provenance':'s2geometry==0.14.0 direct S2Polygon.Contains(S2Cell)/MayIntersect/GetArea and S2Cell.ExactArea, all level5 cells','cases':cases},separators=(',',':'))+'\n')
