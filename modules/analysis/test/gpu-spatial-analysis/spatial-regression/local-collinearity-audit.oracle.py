# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/local-collinearity-audit.oracle.py

VIF=diag(inv(weighted centered predictor correlation)); condition=smax/smin
of its square-root design. mgwr design CN and Belsley proportions use w*X,
not sqrt(w)*X, raw [1,X], column unit norms. NumPy LAPACK SVD is independent
of the GPU streaming QR and one-sided Jacobi. Empty, masked, insufficient,
rank-deficient and shifted/rescaled neighborhoods exercise status and tails.
"""
import json
import numpy as np
features = np.array([[0,0],[1,2],[2,1],[3,3],[4,0],[-1,1],[2,-2],[2,2],[100,-100]], dtype=float)
neighborhoods = [list(range(9)), list(range(6)), [], [0,1], [0,3,7], [0,1,2,6], [0,1,2], [8], list(range(8))]
local_weights = [[1,2,3,1,2,1,2,1,100],[2,1,1,3,2,1],[],[1,1],[1,1,1],[1,2,1,3],[1,2,3],[1],[1]*8]
cases=[]
for transformed in [False, True]:
    predictors = features*([2,.5] if transformed else [1,1])+([4,-2] if transformed else [0,0])
    inflation, condition, design_condition, decomposition, status = [], [], [], [], []
    for neighbors, weights in zip(neighborhoods, local_weights):
        retained = [index for index,row in enumerate(neighbors) if row != 8]
        design = predictors[np.array([neighbors[index] for index in retained], dtype=int)]
        weight = np.array([weights[index] for index in retained], dtype=float)
        if len(design) <= 2:
            inflation += [None]*2; condition.append(None); design_condition.append(None); decomposition += [None]*3; status.append(1); continue
        centered = design - np.average(design, axis=0, weights=weight)
        weighted = np.sqrt(weight[:,None])*centered
        weighted /= np.linalg.norm(weighted,axis=0)
        singular_values = np.linalg.svd(weighted,compute_uv=False)
        if singular_values[-1] <= singular_values[0]*1e-7:
            inflation += [None]*2; condition.append(3e38); design_condition.append(None); decomposition += [None]*3; status.append(2); continue
        inflation += np.diag(np.linalg.inv(weighted.T@weighted)).tolist()
        condition.append(singular_values[0]/singular_values[-1])
        raw = np.column_stack([np.ones(len(design)),design])*weight[:,None]
        raw /= np.linalg.norm(raw,axis=0)
        _, singular_values, right_transpose = np.linalg.svd(raw,full_matrices=False)
        design_condition.append(singular_values[0]/singular_values[-1])
        components = (right_transpose/singular_values[:,None])**2
        decomposition += (components[-1]/components.sum(axis=0)).tolist()
        status.append(0)
    cases.append(dict(transformed=transformed,predictors=predictors.reshape(-1).tolist(),varianceInflationFactors=inflation,conditionNumbers=condition,designConditionNumbers=design_condition,varianceDecomposition=decomposition,status=status))
print(json.dumps(cases,indent=2))
