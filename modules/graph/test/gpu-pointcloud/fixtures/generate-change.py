# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NumPy2.2.6 / SciPy1.15.3 independent cylindrical-support and correlated-noise oracle."""
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.spatial.distance import cdist
assert np.__version__=='2.2.6' and scipy.__version__=='1.15.3'
reference=np.array([[0,0,-.1],[.4,0,0],[0,.4,.1],[.4,.4,.05],[5,0,9]],dtype=float)
comparison=reference+np.array([0,0,.25])
normal=np.array([0,0,1]);radius=.8;depth=1.;length=.5
selected=[]
for cloud in [reference,comparison]:
 projection=cloud@normal
 radial=cloud-projection[:,None]*normal
 selected.append(cloud[(np.abs(projection)<=depth)&(np.sum(radial**2,axis=1)<=radius**2)])
means=[(cloud@normal).mean() for cloud in selected]
roughness=[(cloud@normal).std(ddof=1) for cloud in selected]
variances=[.04,.09]
sampling=sum(variance*np.exp(-cdist(cloud,cloud,'sqeuclidean')/(2*length**2)).sum()/len(cloud)**2 for variance,cloud in zip(variances,selected))
detection=1.96*np.sqrt(sampling+.01)
output=dict(generator='NumPy2.2.6 / SciPy1.15.3 cdist Gaussian covariance sum',reference=reference.ravel().tolist(),comparison=comparison.ravel().tolist(),radius=radius,halfDepth=depth,correlationLength=length,referenceVariance=variances[0],comparisonVariance=variances[1],registrationVariance=.01,counts=[len(cloud) for cloud in selected],changes=[means[1]-means[0],means[1]-means[0]-detection,means[1]-means[0]+detection],roughness=[*roughness,detection])
Path(__file__).with_name('change.json').write_text(json.dumps(output,indent=2)+'\n')
