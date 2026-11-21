# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/gwr-mathematical-audit.oracle.py

Independent raw-design NumPy least squares/SVD, not the centered Cholesky in TS/WGSL.
mgwr conventions: self counts in adaptive k, equal distances all receive equal weights;
the GPU's documented 1.00001 expansion replaces mgwr's 1.0000001. sigma2_v1 uses
RSS/(n-tr(S)), never tr(S'S). Local R2 evaluates the focal model at all neighbors.
SciPy supplies Student t inverse survival. The masked ninth row is deliberately extreme.
The shifted copy translates positions and rescales the predictors; fitted values, hat,
scores, local R2 and significance must remain unchanged (intercepts transform).
"""
import json
import numpy as np
from scipy.stats import t

positions = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [40, 40]], dtype=float)
predictors = np.array([[0, 2], [1, 0], [-1, 1], [2, -1], [-2, -2], [1, 2], [-1, -2], [2, 1], [100, -100]], dtype=float)
response = np.array([2, 5, -1, 6, -3, 5, -2, 8, 1000], dtype=float)
mask = np.array([1] * 8 + [0])


def fit(kernel, mode, value, transformed=False):
    locations = positions + ([16, -8] if transformed else [0, 0])
    features = predictors * ([2, 0.5] if transformed else [1, 1]) + ([4, -2] if transformed else [0, 0])
    design = np.column_stack([np.ones(8), features[:8]])
    observed = response[:8]
    coefficients, covariance, fitted, hat, local_r2, condition = [], [], [], [], [], []
    for focus in range(8):
        distances = np.linalg.norm(locations[:8] - locations[focus], axis=1)
        bandwidth = value if mode == 'fixed' else np.sort(distances)[int(value) - 1] * 1.00001
        ratio = distances / bandwidth
        weights = np.exp(-ratio**2 / 2) if kernel == 'gaussian' else np.where(ratio < 1, (1-ratio**2)**2, 0)
        weighted_design = np.sqrt(weights[:, None]) * design
        coefficient = np.linalg.lstsq(weighted_design, np.sqrt(weights) * observed, rcond=None)[0]
        influence = np.linalg.pinv(weighted_design) * np.sqrt(weights)[None, :]
        coefficients.append(coefficient)
        covariance.append(np.diag(influence @ influence.T))
        fitted.append(design[focus] @ coefficient)
        hat.append((design @ influence)[focus, focus])
        weighted_mean = np.average(observed, weights=weights)
        local_r2.append(1 - np.sum(weights*(observed-design@coefficient)**2) / np.sum(weights*(observed-weighted_mean)**2))
        diagnostic_design = weights[:, None] * design
        diagnostic_design /= np.linalg.norm(diagnostic_design, axis=0)
        singular_values = np.linalg.svd(diagnostic_design, compute_uv=False)
        condition.append(singular_values[0] / singular_values[-1])
    residuals = observed - fitted
    residual_sum = float(residuals @ residuals)
    trace = float(np.sum(hat))
    sigma = residual_sum / (8 - trace)
    aicc = (8*np.log(residual_sum/8) + 8*np.log(2*np.pi) + 8*(8+trace)/(8-2-trace)) if trace < 6 else np.nan
    alpha = .05*3/trace
    critical = t.isf(alpha/2, 7)
    errors = np.sqrt(np.array(covariance)*sigma)
    values = np.array(coefficients)/errors
    def padded(values, width=1):
        return np.append(np.asarray(values).reshape(-1), [np.nan]*width).tolist()
    return dict(coefficients=padded(coefficients, 3), standardErrors=padded(errors, 3),
                tValues=padded(values, 3), significant=np.append((np.abs(values)>critical).astype(int).reshape(-1), [0]*3).tolist(),
                localR2=padded(local_r2), fitted=padded(fitted), residuals=padded(residuals), hatDiagonal=padded(hat),
                localConditionNumber=padded(condition), localStatus=[0]*8+[2],
                summary=[residual_sum, trace, aicc, 1-residual_sum/np.sum((observed-observed.mean())**2), 8, 1, 0, sigma, alpha, critical,
                         -4*(np.log(2*np.pi)+np.log(residual_sum/8)+1), 8],
                candidate=[value, aicc, 0, residual_sum, trace, 8, 8])


cases = []
for kernel, mode, bandwidths, transformed in [
        ('gaussian', 'fixed', [1, 3], False),
        ('bisquare', 'fixed', [3, 4], False),
        ('bisquare', 'adaptive', [7, 8], False),
        ('gaussian', 'fixed', [1, 3], True)]:
    fits = [fit(kernel, mode, value, transformed) for value in bandwidths]
    selected = int(np.nanargmin([value['summary'][2] for value in fits]))
    result = fits[selected].copy()
    del result['candidate']
    result.update(bandwidthScores=[value['summary'][2] for value in fits] + [np.nan],
                  bandwidthSingularRowCounts=[0, 0, 0], candidateTable=sum([value['candidate'] for value in fits], []) + [np.nan]*7,
                  selectedBandwidth=[selected, bandwidths[selected]], approximation=[0])
    cases.append(dict(kernel=kernel, bandwidthMode=mode, bandwidths=bandwidths, transformed=transformed, expected=result))

# Two refinement passes for capacity 3: winner is the upper endpoint 3; successive
# interior thirds/quarters are explicitly derived from [1,3], then [2.5,3].
refined_values = [1, 3, 1.5, 2, 2.5, 2.625, 2.75, 2.875]
refined_fits = [fit('gaussian', 'fixed', value) for value in refined_values]
assert np.nanargmin([value['summary'][2] for value in refined_fits]) == 1
result = refined_fits[1].copy()
del result['candidate']
result.update(bandwidthScores=[refined_fits[0]['summary'][2], refined_fits[1]['summary'][2], np.nan],
              bandwidthSingularRowCounts=[0,0,0], selectedBandwidth=[1,3], approximation=[0],
              candidateTable=refined_fits[0]['candidate']+refined_fits[1]['candidate']+[np.nan]*7+sum([value['candidate'] for value in refined_fits[2:]], []))
cases.append(dict(kernel='gaussian', bandwidthMode='fixed', bandwidths=[1,3], transformed=False, refinementPassCount=2, expected=result))

# One golden-section comparison: b=1+0.38197*4, d=5-0.38197*4. No shared search implementation.
first, second = 2.52788, 3.47212
fits = [fit('gaussian', 'fixed', value) for value in [first, second]]
selected = int(np.argmin([value['summary'][2] for value in fits]))
result = fits[selected].copy()
del result['candidate']
result.update(bandwidthScores=[value['summary'][2] for value in fits]+[np.nan], bandwidthSingularRowCounts=[0, 0, 0],
              candidateTable=sum([value['candidate'] for value in fits], []), selectedBandwidth=[selected, [first, second][selected]], approximation=[0])
if selected == 0:
    bracket = [1, 1+.38197*(second-1), second, first, 2, 0]
else:
    bracket = [first, second, 5, 5-.38197*(5-first), 1, 2]
result['searchState'] = bracket+[1, 0, selected, 2, 0, 0]
cases.append(dict(kernel='gaussian', bandwidthMode='fixed', bandwidths=[1, 5], transformed=False, golden=True, expected=result))

def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value

print(json.dumps(sanitize(cases), indent=2))
