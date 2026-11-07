# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Actual PDAL2.8.4 normal and ICP oracle. No production implementation imports.
Run: python generate-pdal-pointcloud.py --pdal /isolated/env/bin/pdal
"""
import argparse,csv,json,subprocess,tempfile
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--pdal',required=True);arguments=parser.parse_args()
version=subprocess.check_output([arguments.pdal,'--version'],text=True).strip()
assert 'pdal 2.8.4 ' in version
fixture=json.loads(Path(__file__).with_name('pointcloud.json').read_text())
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory)
 for name in ['source','target','positions']:
  values=fixture[name]
  with (root/f'{name}.csv').open('w') as output:
   writer=csv.writer(output);writer.writerow(['X','Y','Z']);writer.writerows(zip(values[::3],values[1::3],values[2::3]))
 def run(pipeline,name):
  subprocess.run([arguments.pdal,'pipeline','--stdin','--metadata',str(root/f'{name}-metadata.json')],input=json.dumps(pipeline),text=True,capture_output=True,check=True)
  return json.loads((root/f'{name}-metadata.json').read_text())
 metadata=run([dict(type='readers.text',filename=str(root/'target.csv'),tag='fixed'),dict(type='readers.text',filename=str(root/'source.csv'),tag='moving'),dict(type='filters.icp',inputs=['fixed','moving'],max_iter=20,max_dist=1),dict(type='writers.text',filename=str(root/'icp.csv'))],'icp')['stages']['filters.icp']
 # PDAL's composed matrix includes centroid translation; its raw transform is centroid-relative.
 matrix=[float(value) for value in metadata['composed'].split()]
 transform=[matrix[row*4+column] for column in range(4) for row in range(4)]
 run([str(root/'positions.csv'),dict(type='filters.normal',knn=9,always_up=True,refine=False),dict(type='writers.text',filename=str(root/'normals.csv'),order='NormalX,NormalY,NormalZ',keep_unspecified=False,precision=12)],'normals')
 normals=[[float(row[key]) for key in ['NormalX','NormalY','NormalZ']] for row in csv.DictReader((root/'normals.csv').open())]
 # A separated outlier makes sample-vs-population threshold conventions immaterial.
 outlier_positions=[[float(index % 4),float(index // 4),0.] for index in range(12)]+[[30.,30.,30.]]
 with (root/'outlier.csv').open('w') as stream:
  writer=csv.writer(stream);writer.writerow(['X','Y','Z','Classification']);writer.writerows([*point,1] for point in outlier_positions)
 run([str(root/'outlier.csv'),dict(type='filters.outlier',method='statistical',mean_k=3,multiplier=2),dict(type='writers.text',filename=str(root/'outlier-result.csv'),order='Classification',keep_unspecified=False)],'outlier')
 outlier_mask=[int(float(row['Classification'])!=7) for row in csv.DictReader((root/'outlier-result.csv').open())]
 # Query points share the exact XY of cell-centre ground samples: raster and nearest-one agree.
 height_positions=[[.5,.5,10.],[1.5,.5,20.],[.5,.5,12.],[1.5,.5,19.]]
 with (root/'height.csv').open('w') as stream:
  writer=csv.writer(stream);writer.writerow(['X','Y','Z','Classification']);writer.writerows([*point,2 if index<2 else 1] for index,point in enumerate(height_positions))
 run([str(root/'height.csv'),dict(type='filters.hag_nn',count=1),dict(type='writers.text',filename=str(root/'height-result.csv'),order='HeightAboveGround',keep_unspecified=False)],'height')
 heights=[float(row['HeightAboveGround']) for row in csv.DictReader((root/'height-result.csv').open())]
 output=dict(generator='Actual PDAL CLI',version=version,normalConvention='knn9 includes self; planar normals agree with the leave-self-out covariance fixture, covariance eigenvalues are not compared across differing support',normals=normals,icp=dict(transform=transform,converged=metadata['converged'],fitness=metadata['fitness']),outlier=dict(positions=outlier_positions,capacity=3,multiplier=2,mask=outlier_mask,convention='PDAL sample standard deviation differs from GPU population standard deviation; classifications agree on separated fixture'),height=dict(positions=height_positions,heights=heights,convention='Nearest-one ground reference at identical cell-centre XY; no general hag_nn parity claim'),m3c2='filters.m3c2 absent from this PDAL2.8.4 build; no external M3C2 parity claim')
Path(__file__).with_name('pdal-pointcloud.json').write_text(json.dumps(output,indent=2)+'\n')
