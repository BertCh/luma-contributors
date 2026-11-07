# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Pinned direct GeographicLib polygon area, exact f32 input coordinates."""
import geographiclib,json,struct
from pathlib import Path
from geographiclib.geodesic import Geodesic
assert geographiclib.__version__=='2.1'
f32=lambda value:struct.unpack('f',struct.pack('f',value))[0]
rows=[]
for latitude in [-65,-20,20,65]:
 for longitude in [-160,0,160]:
  rings=[[[longitude-12,latitude-5],[longitude+10,latitude-4],[longitude+13,latitude+5],[longitude-11,latitude+7]],[[longitude-2,latitude-1],[longitude+2,latitude-1],[longitude+2,latitude+1],[longitude-2,latitude+1]]]
  rings=[[[f32(value) for value in point] for point in ring] for ring in rings];areas=[]
  for ring in rings:
   polygon=Geodesic.WGS84.Polygon()
   for longitudeValue,latitudeValue in ring:polygon.AddPoint(latitudeValue,longitudeValue)
   areas.append(abs(polygon.Compute()[2]))
  rows.append({'rings':rings,'area':areas[0]-areas[1]})
path=Path(__file__).resolve().parents[1]/'gpu-spatial-analysis/global-coordinates/fixtures/geodesic-area-adaptive.json';path.write_text(json.dumps({'provenance':'geographiclib==2.1 direct Geodesic.WGS84.Polygon.Compute, 12 holed polygons near equator/high latitudes/antimeridian, exact f32 inputs','rows':rows},indent=2)+'\n')
