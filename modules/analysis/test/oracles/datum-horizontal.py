# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct PROJ hgridshift, CTable2 test-grid encoding documented by GDAL's driver.
Format reference: https://github.com/OSGeo/gdal/blob/release/3.10/frmts/raw/ctable2dataset.cpp
No upstream implementation is copied; only the binary interchange layout is emitted.
"""
import json,math,struct,tempfile
from pathlib import Path
import pyproj
from pyproj.enums import TransformDirection
assert pyproj.__version__=='3.7.2'
def create(path,origin,spacing,width,height,base):
 header=bytearray(160);header[:16]=b'CTABLE V2.0     ';struct.pack_into('<ddddii',header,96,*map(math.radians,[*origin,*spacing]),width,height)
 samples=[];payload=bytearray()
 for row in range(height):
  for column in range(width):
   longitude=base+column*1e-7+row*2e-8;latitude=-base+row*2e-7-column*1e-8
   packed=struct.pack('<ff',-math.radians(longitude),math.radians(latitude));payload.extend(packed)
   decoded=struct.unpack('<ff',packed);samples.extend([-math.degrees(decoded[0]),math.degrees(decoded[1]),0])
 path.write_bytes(header+payload)
 return {'origin':origin,'spacing':spacing,'width':width,'height':height,'samples':samples}
def result(values):return list(values) if all(map(math.isfinite,values)) else None
with tempfile.TemporaryDirectory() as temporary:
 first=Path(temporary)/'first.ct2';second=Path(temporary)/'second.ct2'
 grids=[create(first,[179,45],[.001,.001],3,3,1e-7),create(second,[178.999,44.999],[.001,.001],5,5,8e-7)]
 transform=pyproj.Transformer.from_pipeline(f'+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=hgridshift +grids={first},{second} +step +proj=unitconvert +xy_in=rad +xy_out=deg')
 rows=[]
 for point in [[179.000123456789,45.000123456789,100],[179.0015,45.0017,50],[178.9993,44.9995,10],[179,45,0],[179.003,45.003,1],[179.1,45.1,1]]:
  forward=result(transform.transform(*point));inverse=result(transform.transform(*(forward or point),direction=TransformDirection.INVERSE));rows.append({'input':point,'forward':forward,'inverse':inverse})
 # PROJ's target-seeded inverse can fail when a forward image leaves all grid coverage.
 outside=[179.003,45.002,1];outside_forward=result(transform.transform(*outside));outside_inverse=result(transform.transform(*outside_forward,direction=TransformDirection.INVERSE)) if outside_forward else None
path=Path(__file__).resolve().parents[1]/'gpu-spatial-analysis/global-coordinates/fixtures/datum-horizontal.json'
path.write_text(json.dumps({'provenance':f'pyproj==3.7.2 PROJ {pyproj.proj_version_str}, direct hgridshift with two prioritized generated CTable2 grids; samples decode the actual f32 radian file words','grids':grids,'rows':rows,'outsideInverse':{'input':outside,'forward':outside_forward,'inverse':outside_inverse}},indent=2)+'\n')
