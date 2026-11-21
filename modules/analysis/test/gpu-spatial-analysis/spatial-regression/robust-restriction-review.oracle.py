# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent review-case OLS/HC0 calculations.
Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/robust-restriction-review.oracle.py
Uses dense NumPy SVD and SciPy chi2.sf; no TypeScript imports.
"""
import numpy as np
from scipy.stats import chi2
import contextlib
import io
import runpy
from pathlib import Path

regimes = np.arange(8) % 2
response = np.array([0, 1, 4, 9, 3, 12, 10, 10], dtype=float)
for name, predictor, varying in [
    ('R3', [0, 0, 0, 1, 0, 2, 0, 3], False),
    ('R5', np.array([0, 0, 1, 1, 0, 0, 1, 1])/16, True)
]:
    predictor = np.array(predictor, dtype=float)
    design = np.column_stack([regimes == 0, regimes == 1, *([predictor*(regimes == 0), predictor*(regimes == 1)] if varying else [predictor])])
    inverse = np.linalg.pinv(design)
    estimate = inverse @ response
    residual = response - design @ estimate
    scores = inverse * residual
    covariance = scores @ scores.T
    restriction = np.array([[-1,1,0,0], [0,0,-1,1]]) if varying else np.array([[-1,1,0]])
    difference = restriction @ estimate
    projected = restriction @ scores
    statistic = difference @ np.linalg.solve(projected @ projected.T, difference)
    print(name, 'coefficients', estimate.tolist(), 'covariance', covariance.ravel().tolist())
    print(name, 'joint', [statistic, len(restriction), chi2.sf(statistic, len(restriction))])
    print(name, 'fitted', (design @ estimate).tolist())
# R2: K=11' makes meat (X'e)(X'e)' identically zero by the normal equations.
small_design = np.array([[1,0,0],[0,1,0],[1,0,1],[0,1,1]], dtype=float)
small_response = np.array([0,1,4,9], dtype=float)
small_inverse = np.linalg.pinv(small_design)
small_scores = small_inverse*(small_response-small_design@(small_inverse@small_response))
print('R3 no unrestricted df', 'covariance', (small_scores@small_scores.T).ravel().tolist(), 'tail', chi2.sf(9,1))
# R9: doubling identical rows doubles X'X and meat, halves HC0 covariance,
# and doubles Wald chi-square. chi2.sf(2*14.08897835587404,4) = 1.14787416e-5.
print('R9', 2*14.08897835587404, chi2.sf(2*14.08897835587404, 4))

# Recompute the doubled fixture from its independent generator, without TypeScript imports.
with contextlib.redirect_stdout(io.StringIO()):
    base = runpy.run_path(str(Path(__file__).with_name('robust-restriction.oracle.py')))
design = base['design'](base['predictors'], [False, True, False])
observed = base['response'][base['used']].astype(float)
design = np.vstack([design, design])
observed = np.tile(observed, 2)
inverse = np.linalg.pinv(design)
estimate = inverse @ observed
scores = inverse * (observed-design @ estimate)
restriction = np.array([[-1,1,0,0,0,0,0],[-1,0,1,0,0,0,0],[0,0,0,0,-1,1,0],[0,0,0,0,-1,0,1]])
projected = restriction @ scores
difference = restriction @ estimate
for name, keep in [('all tiles', np.tile(base['used'],2)), ('wrong first tile only', np.tile(base['used'],2) & (np.arange(66)<64))]:
    kept = keep[np.tile(base['used'],2)]
    covariance = scores[:,kept] @ scores[:,kept].T
    contrast_covariance = projected[:,kept] @ projected[:,kept].T
    statistic = difference @ np.linalg.solve(contrast_covariance,difference)
    print('R9',name,'covariance[0]',covariance[0,0],'joint',[statistic,4,chi2.sf(statistic,4)])
print('R9 tail covariance diagonal contribution', (scores[:,-1]**2).tolist())
