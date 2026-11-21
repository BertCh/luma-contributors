# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent QR/SVD OLS audit. Run:
/usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/audit-ols.oracle.py
Prints literals for audit-ols-fixtures.ts; no production code or CPU twin is used.
"""

import json
import numpy as np
from scipy import linalg, stats


def fit(offset, scale):
    # Repeated predictor rows, a masked outlier, missing response, invalid predictor.
    predictors = np.array([[0, 1], [1, 0], [1, 0], [2, 2], [3, -1], [4, 1],
                           [5, 3], [6, -2], [7, 2], [8, 0], [9, 1], [10, np.nan]], float)
    response = np.array([2, 4, 3, 5, 8, 7, 8, 15, 11, 200, np.nan, 5], float)
    predictors = ((predictors + [offset, -offset]) * scale).astype(np.float32).astype(float)
    response = (response + offset).astype(np.float32).astype(float)
    mask = np.array([1] * 9 + [0, 1, 1])
    used = (mask != 0) & np.isfinite(response) & np.isfinite(predictors).all(axis=1)
    design = np.column_stack([np.ones(sum(used)), predictors[used]])
    observed = response[used]
    coefficients = linalg.lstsq(design, observed, lapack_driver='gelsy')[0]
    residual = observed - design @ coefficients
    count, parameter_count = design.shape
    rss = residual @ residual
    tss = np.sum((observed - observed.mean()) ** 2)
    sigma = rss / (count - parameter_count)
    _, upper = linalg.qr(design, mode='economic')
    inverse_root = linalg.solve_triangular(upper, np.eye(parameter_count))
    errors = np.sqrt(sigma * np.diag(inverse_root @ inverse_root.T))
    squared = residual ** 2
    auxiliary = design @ linalg.lstsq(design, squared, lapack_driver='gelsy')[0]
    explained = np.sum((auxiliary - squared.mean()) ** 2)
    koenker = count * explained / np.sum((squared - squared.mean()) ** 2)
    breusch = explained / (2 * (rss / count) ** 2)
    skew = np.mean(residual ** 3) / np.mean(squared) ** 1.5
    kurtosis = np.mean(residual ** 4) / np.mean(squared) ** 2
    jarque = count / 6 * (skew ** 2 + (kurtosis - 3) ** 2 / 4)
    likelihood = -count / 2 * (np.log(2 * np.pi * rss / count) + 1)
    centered = predictors[used] - predictors[used].mean(axis=0)
    normalized = centered / linalg.norm(centered, axis=0)
    correlation = normalized.T @ normalized
    eigenvalues = linalg.eigvalsh(correlation)
    singular = linalg.svdvals(design / linalg.norm(design, axis=0))
    summary = [count, 1-rss/tss, 1-(rss/tss)*(count-1)/(count-parameter_count), sigma,
               likelihood, 2*parameter_count-2*likelihood, parameter_count*np.log(count)-2*likelihood,
               jarque, stats.chi2.sf(jarque, 2), koenker, stats.chi2.sf(koenker, 2), rss, tss,
               skew, kurtosis, 0, breusch, stats.chi2.sf(breusch, 2), singular[0]/singular[-1],
               eigenvalues[-1]/eigenvalues[0], np.max(np.diag(linalg.inv(correlation))), 0]
    fitted = np.full(len(response), np.nan)
    residuals = fitted.copy()
    predictions = fitted.copy()
    fitted[used] = design @ coefficients
    residuals[used] = residual
    finite = np.isfinite(predictors).all(axis=1)
    predictions[finite] = np.column_stack([np.ones(sum(finite)), predictors[finite]]) @ coefficients
    return dict(name=f'offset-{offset}-scale-{scale}', predictors=predictors.ravel().tolist(),
                response=response.tolist(), mask=mask.tolist(),
                inverseNormal=(inverse_root @ inverse_root.T).tolist(),
                cholesky=linalg.cholesky(design.T @ design, lower=True).tolist(),
                singularRange=[linalg.svdvals(design)[-1], linalg.svdvals(design)[0]],
                expected=dict(status=[0],
                coefficients=coefficients.tolist(), standardErrors=errors.tolist(),
                tStatistics=(coefficients/errors).tolist(), summary=summary,
                residuals=residuals.tolist(), fitted=fitted.tolist(), predictions=predictions.tolist()))


if __name__ == '__main__':
    print(json.dumps([fit(0, 1), fit(16, 4)], indent=2))
