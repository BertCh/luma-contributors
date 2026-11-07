# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Pinned scikit-learn1.7.2 xi extraction and scipy1.15.3 independent midrank oracle."""
import json
from pathlib import Path
import numpy as np
import sklearn, scipy
from sklearn.cluster import OPTICS
from scipy.stats import rankdata
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
for curves,alpha in [([[1,3,1],[1,2,3],[2,2,2],[3,1,3],[0,4,0]],0.4),([[1,1],[1,1],[1,1],[1,1]],0.25),([[0,0],[2,3],[3,2],[4,4],[9,9]],0.2)]:
 array=np.array(curves,dtype=float); count=len(curves)
 lower=rankdata(array,axis=0,method='average'); upper=count+1-lower
 extreme=np.minimum(lower,upper).min(axis=1)
 critical=max(rank for rank in range(1,(count+1)//2+1) if np.count_nonzero(extreme<rank)<=np.floor(alpha*count))
 sortedValues=np.sort(array,axis=0)
 rankfixtures.append(dict(curves=curves,alpha=alpha,ranks=extreme.tolist(),interval=[float(np.mean(extreme<extreme[-1])),float(np.mean(extreme<=extreme[-1]))],lower=sortedValues[critical-1].tolist(),upper=sortedValues[count-critical].tolist()))
Path(__file__).with_name('point-process-advanced-fixtures.json').write_text(json.dumps(dict(reference=dict(sklearn=sklearn.__version__,scipy=scipy.__version__,numpy=np.__version__),xi=fixtures,rank=rankfixtures),indent=2)+'\n')
