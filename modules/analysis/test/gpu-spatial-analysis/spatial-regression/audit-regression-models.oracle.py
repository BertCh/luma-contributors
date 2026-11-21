# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/audit-regression-models.oracle.py
Independent SciPy least squares; prints literals for audit-regression-models-fixtures.ts.
"""

import json
import numpy as np
from scipy import linalg


def regime(offset, scale):
    groups = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 5, 1])
    predictors = np.array([[0,1],[0,0],[1,2],[0,1],[1,0],[2,1],[1,2],[1,0],
                           [2,1],[1,2],[2,2],[3,-2],[3,-1],[4,2],[4,1],[5,-1],
                           [1,1],[2,0],[0,0],[np.nan,0]], float)
    response = np.array([2,5,1,4,4,3,2,3,4,3,5,9,8,2,7,8,100,np.nan,100,3], float)
    predictors = ((predictors + [offset,-offset]) * scale).astype(np.float32).astype(float)
    response = (response + offset).astype(np.float32).astype(float)
    mask = np.ones(20, int)
    mask[16] = 0
    valid = (mask != 0) & np.isfinite(response) & np.isfinite(predictors).all(axis=1)
    counts, status, coefficients = [], [], []
    fitted = np.full(20, np.nan)
    offsets, moments = [0], []
    for group in range(5):
        used = valid & (groups == group)
        count = int(sum(used))
        counts.append(count)
        offsets.append(offsets[-1] + count)
        values = np.column_stack([predictors[used], response[used]])
        mean = values.mean(axis=0).astype(np.float32).astype(float) if count else np.zeros(3)
        centered = values - mean
        gram = centered.T @ centered
        moments.extend(mean.tolist() + [gram[a,b] for a in range(3) for b in range(a,3)] + centered.sum(axis=0).tolist())
        design = np.column_stack([np.ones(count), predictors[used]])
        code = 1 if count < 3 else 2 if np.linalg.matrix_rank(design) < 3 else 0
        status.append(code)
        if code:
            coefficients.extend([np.nan]*3)
        else:
            estimate = linalg.lstsq(design, response[used], lapack_driver='gelsy')[0]
            coefficients.extend(estimate.tolist())
            fitted[used] = design @ estimate
    return dict(name=f'offset-{offset}-scale-{scale}', predictors=predictors.ravel().tolist(),
                response=response.tolist(), regimes=groups.tolist(), mask=mask.tolist(),
                expected=dict(coefficients=coefficients, fitted=fitted.tolist(), status=status,
                              counts=counts), offsets=offsets, moments=moments)


def cross_validation(offset, scale, empty=False):
    predictors = np.array([0,1,1,2,3,4,0,1,2,3,4,5], float)
    response = np.array([2,4,3,5,8,7,4,3,7,8,np.nan,9], float)
    predictors = ((predictors + offset) * scale).astype(np.float32).astype(float)
    response = (response + offset).astype(np.float32).astype(float)
    groups = np.array([0]*6+[1]*6)
    mask = np.ones(12, int)
    mask[5] = 0
    if empty:
        # No fold can score or fit: OLS has zero training rows, R² has zero test variance.
        return dict(name='empty-training-and-testing', predictors=predictors.tolist(),
                    response=response.tolist(), blocks=groups.tolist(), mask=[0]*12,
                    expected=dict(coefficients=[np.nan]*4, predictions=[np.nan]*24,
                                  fitStatus=[2,2], metrics=[0,0,np.nan,np.nan,np.nan,np.nan]*2,
                                  metricStatus=[5,5]),
                    foldExpected=dict(folds=[4294967295]*12,training=[0]*24,testing=[0]*24,
                                      counts=[0]*6,status=[3,3]))
    coefficients, predictions, metrics, codes = [], [], [], []
    training, testing, counts = [], [], []
    for fold in range(2):
        train = (groups != fold) & (mask != 0)
        test = (groups == fold) & (mask != 0)
        training.extend(train.astype(int).tolist())
        testing.extend(test.astype(int).tolist())
        counts.extend([int(sum(train)),int(sum(test)),0])
        used = train & np.isfinite(response)
        design = np.column_stack([np.ones(sum(used)),predictors[used]])
        estimate = linalg.lstsq(design,response[used],lapack_driver='gelsy')[0]
        predicted = np.column_stack([np.ones(12),predictors]) @ estimate
        coefficients.extend(estimate.tolist())
        predictions.extend(predicted.tolist())
        scored = test & np.isfinite(response)
        error = predicted[scored] - response[scored]
        tss = np.sum((response[scored]-response[scored].mean())**2)
        missing = int(sum(test)-sum(scored))
        metrics.extend([int(sum(scored)),missing,np.mean(error**2),np.mean(abs(error)),np.mean(error),1-np.sum(error**2)/tss])
        codes.append(2 if missing else 0)
    return dict(name=f'offset-{offset}-scale-{scale}', predictors=predictors.tolist(),
                response=response.tolist(), blocks=groups.tolist(), mask=mask.tolist(),
                expected=dict(coefficients=coefficients,predictions=predictions,fitStatus=[0,0],
                              metrics=metrics,metricStatus=codes),
                foldExpected=dict(folds=[0]*5+[4294967295]+[1]*6,training=training,testing=testing,counts=counts,status=[0,0]))


if __name__ == '__main__':
    print(json.dumps(dict(regimes=[regime(0,1),regime(16,4)],
                          crossValidation=[cross_validation(0,1),cross_validation(16,4),
                                           cross_validation(0,1,empty=True)]),indent=2))
