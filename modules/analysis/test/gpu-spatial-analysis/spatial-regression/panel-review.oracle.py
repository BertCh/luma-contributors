# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/panel-review.oracle.py
Independent SVD projections, conditional Gaussian likelihood and dummy-variable IV.
"""
import json
import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize_scalar

PREDICTORS = np.array([-2,1,1,3,0,-1,2,-2,1,3,-2,0,2,-2,0,1], dtype=float)
RESPONSE = np.array([1,4,2,8,3,-1,6,1,4,5,-2,2,7,-3,4,3], dtype=float)
WEIGHTS = np.array([[0,.5,0,.5],[.5,0,.5,0],[0,.5,0,.5],[.5,0,.5,0]])

def diagnostics(predictors, response=RESPONSE, weights=WEIGHTS):
    design = np.column_stack([np.ones(16), predictors])
    operator = np.kron(np.eye(4), weights)
    fitted = design @ np.linalg.lstsq(design, response, rcond=None)[0]
    residual = response - fitted
    lagged = operator @ fitted
    projected = lagged - design @ np.linalg.lstsq(design, lagged, rcond=None)[0]
    variance = residual @ residual / 14
    ratio = float(projected @ projected / (lagged @ lagged))
    if ratio < 1e-28:
        # W1=1, Wx=-x proves the projected lag is exactly zero; SVD roundoff is not information.
        expected = [None] * 6
    else:
        trace = float(np.trace(operator.T @ operator + operator @ operator))
        information = float(projected @ projected / variance)
        error_score = float(residual @ operator @ residual / variance)
        lag_score = float(residual @ projected / variance)
        full = information + trace
        error_contrast = information / full * error_score - trace / full * lag_score
        statistics = [error_score**2 / trace, lag_score**2 / information,
                      error_contrast**2 / (trace * information / full)]
        expected = [number for value in statistics for number in [value, chi2.sf(value, 1)]]
    return dict(fitted=fitted.tolist(), residuals=residual.tolist(), projected=projected.tolist(),
                variance=float(variance), projectionRatio=ratio, tests=expected)

def unit_lag(predictors, response, weights, periods):
    entities = len(weights)
    predictors = np.asarray(predictors).reshape(periods, entities, -1).astype(float)
    response = np.asarray(response).reshape(periods, entities).astype(float)
    predictors -= predictors.mean(axis=0)
    response -= response.mean(axis=0)
    design = predictors.reshape(periods*entities, -1)
    observed = response.ravel()
    lag = (response @ weights.T).ravel()
    spectrum = np.linalg.eigvalsh(weights)
    rows, columns = design.shape
    def profile(rho):
        beta = np.linalg.lstsq(design, observed-rho*lag, rcond=None)[0]
        residual = observed-rho*lag-design@beta
        return rows/2*np.log(residual@residual/rows)-periods*np.log(1-rho*spectrum).sum()
    optimum = minimize_scalar(profile, bounds=(-.999,.999), method='bounded', options={'xatol':1e-15})
    rho = min([optimum.x, -.999, .999], key=profile)
    beta = np.linalg.lstsq(design, observed-rho*lag, rcond=None)[0]
    fitted = design@beta+rho*lag
    residual = observed-fitted
    variance = residual@residual/rows
    product = weights @ np.linalg.inv(np.eye(entities)-rho*weights)
    lag_prediction = (product @ (design@beta).reshape(periods,entities).T).T.ravel()
    information_design = np.column_stack([design,lag_prediction])
    information = information_design.T@information_design/variance
    information[-1,-1] += periods*np.trace(product@product+product.T@product)
    covariance = np.full((columns+2, columns+2),np.nan)
    covariance[1:,1:] = np.linalg.inv(information)
    return dict(coefficients=[None,*beta,rho,0], covariance=[float(value) if np.isfinite(value) else None for value in covariance.ravel()],
                summary=[float(variance),float(-profile(rho)-rows/2*(1+np.log(2*np.pi))-rows/2*np.log(rows))],
                fitted=fitted.tolist(),residuals=residual.tolist())

def shifted_iv():
    response = (RESPONSE/4096 + np.tile([3000,-3000,3000,-3000],4)).astype(np.float32).astype(float)
    operator = np.kron(np.eye(4),WEIGHTS)
    dummies = np.tile(np.eye(4),(4,1))
    design = np.column_stack([dummies,PREDICTORS,operator@response])
    instruments = np.column_stack([dummies,PREDICTORS,operator@PREDICTORS])
    projected = instruments@np.linalg.lstsq(instruments,design,rcond=None)[0]
    estimate = np.linalg.lstsq(projected,response,rcond=None)[0]
    residual = response-design@estimate
    variance = float(residual@residual/10)
    expected = 1.2862052364361378/4096**2
    assert abs(variance/expected-1) < 1e-7
    return dict(variance=variance,expectedVariance=expected,residuals=residual.tolist())

if __name__ == '__main__':
    zero_fit_response=np.array([-2,5,0,-3]+[0]*12)
    # Every operation below is exact on these small integers. The pooled normal scores are
    # zero, so the fitted vector, its spatial lag and robust LM information are exactly zero.
    assert int(zero_fit_response.sum())==0 and int(PREDICTORS@zero_fit_response)==0
    print('review 2 R2',dict(status=2,tests=[None]*6))
    predictor = np.array([period * value for period in range(1,5) for value in [1,-1,1,-1]], dtype=float)
    nearby = predictor.copy()
    nearby[0] += .25
    assert np.allclose(diagnostics(nearby)['tests'],diagnostics(nearby,RESPONSE+32768)['tests'],rtol=1e-8,atol=1e-8)
    nonuniform=WEIGHTS.copy()
    nonuniform[0]*=.5
    print('review 2 R6 nonuniform',json.dumps(diagnostics(nearby,RESPONSE+16,nonuniform)['tests']))
    print(json.dumps(dict(rankZero=diagnostics(predictor), nearby=diagnostics(nearby)), indent=2))
    offsets=np.array([-2048,2048,-2048,2048])
    design=np.array([[offsets[entity]+[1,-1,1,-1][period],offsets[entity]+[1,1,-1,-1][period]] for period in range(4) for entity in range(4)])
    print('R4',json.dumps(unit_lag(design,RESPONSE,WEIGHTS,4),indent=2))
    extreme_design=np.array([[2**48 if entity==0 else -2**48 if entity==1 else [1,-1,1,-1][period],
                              2**48 if entity==0 else -2**48 if entity==1 else [1,1,-1,-1][period]]
                             for period in range(4) for entity in range(4)],dtype=float)
    print('review 2 R4',json.dumps(unit_lag(extreme_design,RESPONSE,WEIGHTS,4),indent=2))
    boundary_response=np.array([1,2,-3,2,1,-3,-2,3,-1,0,-1,1])
    triangle=(np.ones((3,3))-np.eye(3))*.5
    print('R6',json.dumps(unit_lag(PREDICTORS[:12],boundary_response,triangle,4),indent=2))
    print('R7',json.dumps(shifted_iv()))
