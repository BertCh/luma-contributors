# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct GeographicLib inverse oracle; exact float32 input coordinates."""
import geographiclib,json,random,struct
from geographiclib.geodesic import Geodesic
from pathlib import Path
assert geographiclib.__version__=='2.1'
def f32(value):return struct.unpack('f',struct.pack('f',value))[0]
randomizer=random.Random(48019);rows=[]
for index in range(256):
 longitude=randomizer.uniform(-170,170);latitude=randomizer.uniform(-80,80)
 if index<128:
  targetLongitude=longitude+180-randomizer.uniform(.01,1.2);targetLatitude=-latitude+randomizer.uniform(-1.2,1.2)
  if targetLongitude>180:targetLongitude-=360
 else:targetLongitude=randomizer.uniform(-180,180);targetLatitude=randomizer.uniform(-89.9,89.9)
 first=list(map(f32,[longitude,latitude]));second=list(map(f32,[targetLongitude,targetLatitude]));result=Geodesic.WGS84.Inverse(first[1],first[0],second[1],second[0]);rows.append({'first':first,'second':second,'distance':result['s12'],'difficult':index<128})
path=Path(__file__).resolve().parents[1]/'gpu-spatial-analysis/global-coordinates/fixtures/difficult-geodesics.json';path.write_text(json.dumps({'provenance':'geographiclib==2.1 direct Geodesic.WGS84.Inverse of exactly f32-representable coordinates; deterministic 128 near-antipodal and 128 global pairs','rows':rows},indent=2)+'\n')
