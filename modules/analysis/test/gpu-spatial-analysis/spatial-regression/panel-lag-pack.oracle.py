# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/panel-lag-pack.oracle.py

Independent float64 SVD/concentrated Gaussian unit-effects likelihood on the 8-row
hand packing fixture. W and temporal demeaning commute; expectations use the original
float32 observations, never the contributor's packed data or normal equations.
"""
import json
import numpy as np
from scipy.optimize import minimize_scalar

weights = np.array([[0,.5,0,.5],[.5,0,.5,0],[0,.5,0,.5],[.5,0,.5,0]])
predictors = np.array([0,1,2,3,2,-2,1,4], dtype=float)
response = np.tile([3000,-3000,3000,-3000], 2) + np.array([1,2,4,3,-1,-2,-4,-3])*2.0**-12 + 16
within = np.eye(8)-np.kron(np.ones((2,2))/2, np.eye(4))
design = (within@predictors).reshape(8,1)
target = within@response
lag = within@np.kron(np.eye(2),weights)@response

def evaluate(rho):
    estimate = np.linalg.lstsq(design,target-rho*lag,rcond=None)[0]
    residual = target-rho*lag-design@estimate
    variance = residual@residual/8
    determinant = np.linalg.slogdet(np.eye(4)-rho*weights)[1]
    return 4*np.log(variance)-2*determinant, estimate, residual, variance

fit=minimize_scalar(lambda rho:evaluate(rho)[0],bounds=(-.999,.999),method='bounded',options={'xatol':1e-14})
rho=float(fit.x)
objective,estimate,residual,variance=evaluate(rho)
product=weights@np.linalg.inv(np.eye(4)-rho*weights)
lag_prediction=((design@estimate).reshape(2,4)@product.T).ravel()
information_design=np.column_stack((design,lag_prediction))
information=information_design.T@information_design/variance
information[-1,-1]+=2*(np.trace(product@product)+np.sum(product**2))
covariance=np.full((3,3),np.nan)
covariance[1:,1:]=np.linalg.inv(information)
expected=dict(coefficients=[None,float(estimate[0]),rho,0], covariance=[None if not np.isfinite(value) else value for value in covariance.ravel().tolist()],
              summary=[variance,-objective-4*(1+np.log(2*np.pi)+np.log(8))], fitted=(target-residual).tolist(),residuals=residual.tolist())
print(json.dumps(expected,indent=2))
