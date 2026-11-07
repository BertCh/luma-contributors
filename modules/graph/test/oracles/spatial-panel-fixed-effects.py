# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent dummy-variable IV composition; run with NumPy to regenerate fixture."""
import json
from pathlib import Path
import numpy as np

entities, periods = 5, 4
weights = np.zeros((entities, entities))
for entity in range(entities):
    weights[entity, (entity + 1) % entities] = .7
    weights[entity, (entity + 2) % entities] = .3
predictors = np.array([1,4,2,8,3, 3,9,4,1,7, 2,5,8,6,4, 9,2,7,3,5], dtype=float)
effects = np.tile([2,-1,4,0,3], periods)
noise = np.array([.2,-.3,.1,.4,-.2, -.1,.4,-.2,.3,.1, .3,-.1,.2,-.4,.2, -.2,.1,-.3,.2,-.1])
block = np.kron(np.eye(periods), weights)
response = np.linalg.solve(np.eye(entities*periods) - .25*block, 1.5*predictors + effects + noise)
dummies = np.tile(np.eye(entities), (periods, 1))
regressors = np.column_stack([dummies, predictors, block @ response])
instruments = np.column_stack([dummies, predictors, block @ predictors])
projected = instruments @ np.linalg.lstsq(instruments, regressors, rcond=None)[0]
normal = regressors.T @ projected
coefficients = np.linalg.solve(normal, projected.T @ response)
residuals = response - regressors @ coefficients
degrees = entities*periods - entities - 2
variance = residuals @ residuals / degrees
standard_errors = np.sqrt(np.diag(np.linalg.inv(normal)) * variance)
fixture = dict(metadata=dict(reference="NumPy dummy-variable IV composition; not direct library panel API", numpy=np.__version__, ordering="period-major", covariance="homoskedastic residual df n-N-k-1", degrees=degrees), entities=entities, periods=periods, offsets=[0,2,4,6,8,10], neighbors=[int((i+j)%entities) for i in range(entities) for j in [1,2]], weights=[.7,.3]*entities, predictors=predictors.tolist(), response=response.tolist(), coefficients=coefficients[-2:].tolist(), standardErrors=standard_errors[-2:].tolist(), residuals=residuals.tolist(), variance=float(variance))
Path(__file__).with_name("fixtures").joinpath("spatial-panel-fixed-effects.json").write_text(json.dumps(fixture, indent=2)+"\n")
