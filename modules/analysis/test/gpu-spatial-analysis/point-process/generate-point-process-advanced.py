# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Pinned scikit-learn1.7.2 xi extraction and scipy1.15.3 independent midrank oracle."""
import json
import sys
from pathlib import Path
import numpy as np
import sklearn, scipy
from sklearn.cluster import OPTICS
from scipy.stats import rankdata
# Rank-only regeneration preserves the pinned Xi evidence and records the local SciPy version.
# /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/point-process/generate-point-process-advanced.py --ranks-only --stdout
if '--ranks-only' not in sys.argv:
 assert sklearn.__version__ == '1.7.2'
 assert scipy.__version__ == '1.15.3'
 rng=np.random.default_rng(421)
 positions=np.concatenate([rng.normal([0,0],0.08,(12,2)),rng.normal([2,0],0.2,(12,2)),rng.uniform([-1,-1],[3,1],(8,2))]).astype(np.float32)
 fixtures=[]
 for correction in [False,True]:
  for xi in [0.05,0.2]:
   model=OPTICS(min_samples=3,max_eps=3,cluster_method='xi',xi=xi,min_cluster_size=4,predecessor_correction=correction).fit(positions)
   fixtures.append(dict(xi=xi,predecessorCorrection=correction,positions=positions.tolist(),minimumSamples=3,minimumClusterSize=4,maximumDistance=3,ordering=model.ordering_.tolist(),predecessors=model.predecessor_.tolist(),labels=model.labels_.tolist(),intervals=model.cluster_hierarchy_.tolist()))
 # This seed exercises a predecessor correction that actually removes a candidate interval.
 for correction in [False,True]:
  positions=np.random.default_rng(1).normal(size=(24,2)).astype(np.float32)
  model=OPTICS(min_samples=3,max_eps=5,cluster_method='xi',xi=.05,min_cluster_size=3,predecessor_correction=correction).fit(positions)
  fixtures.append(dict(xi=.05,predecessorCorrection=correction,positions=positions.tolist(),minimumSamples=3,minimumClusterSize=3,maximumDistance=5,ordering=model.ordering_.tolist(),predecessors=model.predecessor_.tolist(),labels=model.labels_.tolist(),intervals=model.cluster_hierarchy_.tolist()))
rankfixtures=[]
for curves,alpha in [([[1,3,1],[1,2,3],[2,2,2],[3,1,3],[0,4,0]],0.4),([[1,1],[1,1],[1,1],[1,1]],0.25),([[0,0],[2,3],[3,2],[4,4],[9,9]],0.2), ([[0],[1],[2],[2],[2],[0]],0.34), ([[1],[2],[3],[3],[0]],0.21), ([[value] for value in range(1,20)]+[[19],[0]],0.05)]:
 array=np.array(curves,dtype=float); count=len(curves)
 lower=rankdata(array,axis=0,method='average'); upper=count+1-lower
 extreme=np.minimum(lower,upper).min(axis=1)
 critical=max(rank for rank in np.unique(extreme) if np.count_nonzero(extreme<rank)/count<=alpha)
 retained=array[extreme>=critical]
 rankfixtures.append(dict(curves=curves,alpha=alpha,ranks=extreme.tolist(),interval=[float(np.mean(extreme<extreme[-1])),float(np.mean(extreme<=extreme[-1]))],lower=retained.min(axis=0).tolist(),upper=retained.max(axis=0).tolist()))
fixture_path=Path(__file__).with_name('point-process-advanced-fixtures.json')
if '--ranks-only' in sys.argv:
 payload=json.loads(fixture_path.read_text())
 payload['rank']=rankfixtures
 payload['reference']['rankScipy']=scipy.__version__
 payload['reference']['rankNumpy']=np.__version__
else:
 payload=dict(reference=dict(sklearn=sklearn.__version__,scipy=scipy.__version__,numpy=np.__version__),xi=fixtures,rank=rankfixtures)
serialized=json.dumps(payload,indent=2)+'\n'
if '--stdout' in sys.argv:
 print(serialized,end='')
else:
 Path(__file__).with_name('point-process-advanced-fixtures.json').write_text(serialized)
