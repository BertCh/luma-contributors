# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Execute pinned CGAL 6.0.1/Boost 1.87 reference, with independently installed headers.

Compile (substitute the local dependency paths):
clang++ -std=c++17 -O1 -DCGAL_DISABLE_GMP=1 -I/path/to/CGAL-6.0.1/include \
  -I/path/to/boost_1_87_0 geometry-straight-skeleton.cpp -o /tmp/geometry-cgal-oracle
python3 geometry-straight-skeleton.py --oracle /tmp/geometry-cgal-oracle

CGAL and Boost are external reference dependencies; their source/licensing remains
in the reference environment. This repository contains only the invocation wrapper.
"""
import argparse
import json
import subprocess
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--oracle', default='/tmp/luma-geometry-cgal/oracle')
arguments=parser.parse_args()
cases=[
 ('rectangle', [[(0,0),(4,0),(4,2),(0,2)]]),
 ('triangle', [[(0,0),(6,0),(0,8)]]),
 ('concave-L', [[(0,0),(6,0),(6,2),(2,2),(2,5),(0,5)]]),
 ('split-notch', [[(0,0),(8,0),(8,6),(5,6),(4,2),(3,6),(0,6)]]),
 ('square-hole', [[(0,0),(10,0),(10,10),(0,10)],[(3,3),(3,7),(7,7),(7,3)]]),
 ('asymmetric-hole', [[(0,0),(12,0),(12,10),(0,10)],[(3,2),(3,5),(6,5),(6,2)]]),
 ('two-holes', [[(0,0),(16,0),(16,10),(0,10)],[(2,2),(2,5),(5,5),(5,2)],[(10,3),(10,7),(13,7),(13,3)]]),
 ('concave-C', [[(0,0),(8,0),(8,2),(2,2),(2,6),(8,6),(8,8),(0,8)]]),
 ('reflex-pentagon', [[(0,0),(7,0),(7,6),(3,2),(0,6)]]),
 ('collinear-corners', [[(0,0),(2,0),(4,0),(4,2),(0,2)]])
]
results=[]
for name,rings in cases:
 text=f'{len(rings)} 0.25\n'+'\n'.join(str(len(ring))+' '+' '.join(str(value) for point in ring for value in point) for ring in rings)+'\n'
 result=json.loads(subprocess.check_output([arguments.oracle],input=text.encode()))
 positions=[]; offsets=[0]
 for ring in rings:
  positions.extend(value for point in ring for value in point); offsets.append(len(positions)//2)
 results.append(dict(name=name, positions=positions, ringOffsets=offsets, offsetDistance=0.25, **result))
(Path(__file__).parent/'fixtures'/'geometry-straight-skeleton.json').write_text(json.dumps(dict(generator='geometry-straight-skeleton.py',versions=dict(cgal='6.0.1',boost='1.87.0',kernel='Exact_predicates_inexact_constructions_kernel',numberType='boost multiprecision'),cases=results),indent=2)+'\n')
