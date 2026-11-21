# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Reproduce: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geometry-predicates/multipart-review.oracle.py
from shapely import wkt
import numpy as np
for first,second in [('MULTILINESTRING ((0 0,1 0),(1 0,2 0))','MULTILINESTRING ((0 0,2 0))'),('MULTILINESTRING ((0 0,1 0),(1 0,2 0))','POINT (1 0)'),('MULTILINESTRING ((0 0,1 0),(3 0,4 0))','MULTILINESTRING ((0 0,2 0))')]:
    first,second=wkt.loads(first),wkt.loads(second)
    matrix=first.relate(second)
    packed=sum({'F':0,'0':1,'1':2,'2':3}[cell] << (2*index) for index,cell in enumerate(matrix))
    print(matrix,packed,[bool(getattr(first,name)(second)) for name in ['equals','intersects','contains','within','covers','covered_by','touches','crosses','overlaps']],first.relate_pattern(second,'T**FF*FF*'))
for geometry in ['MULTIPOINT ((1000000 0),(.03125 0),(.015625 0))','MULTILINESTRING ((1000000 0,.03125 0,.015625 0))','MULTILINESTRING ((0 0,0 0),(9 0,9 0,9 0,9 0))','MULTILINESTRING ((0 0,0 0),(9 0,9 0,9 0,9 0,9 0))']:
    geometry=wkt.loads(geometry)
    print('measure:',geometry.bounds,geometry.length,geometry.centroid.wkt,geometry.minimum_clearance)
empty=wkt.loads('MULTILINESTRING EMPTY')
line=wkt.loads('MULTILINESTRING ((0 0,2 0))')
for first,second in [(empty,empty),(empty,line),(line,empty),(wkt.loads('MULTIPOINT EMPTY'),wkt.loads('MULTIPOINT EMPTY'))]:
    matrix=first.relate(second)
    print('empty:',matrix,sum({'F':0,'0':1,'1':2,'2':3}[cell] << (2*index) for index,cell in enumerate(matrix)),first.equals(second))
for sign in [1,-1]:
    maximum=sign*float(np.finfo(np.float32).max)
    print('float32 bound:',wkt.loads(f'POINT ({maximum} 0)').bounds)
print('clearance tie:',wkt.loads('MULTILINESTRING ((0 0,0 0),(1 0,1 0),(-1 0,-1 0))').minimum_clearance)
# Both partners are distance 1 from vertex row0. The public deterministic tie rule
# chooses original row2 (+1) before row4 (-1), yielding the witness [0,0,1,0].
