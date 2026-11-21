# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-regression/gwr-nonstationarity-audit.oracle.py

Conditional fixed-bandwidth Monte Carlo GWR: uncentered NumPy SVD solves;
population SD (ddof=0), >= exceedance, add-one p, population simulated SD.
Assignments are explicit experimental inputs: the seed-12345 purpose-GWR-N
swap-or-not bijections for eight locations. This oracle audits the fits and
Monte Carlo statistics conditional on those assignments, not the shared RNG.
"""
import json
import sys
import numpy as np

locations = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1]], dtype=float)
features = np.array([[0, 2], [1, 0], [-1, 1], [2, -1], [-2, -2], [1, 2], [-1, -2], [2, 1]], dtype=float)
response = np.array([2, 5, -1, 6, -3, 5, -2, 8], dtype=float)
assignments = [list(range(8)), [1,3,6,7,5,0,2,4], [5,4,3,6,1,7,2,0], [3,0,4,2,1,7,5,6], [5,3,6,0,1,7,4,2], [1,2,4,5,6,0,3,7]]

cases = []
for adaptive in [False, True]:
    for transformed in [False, True]:
        positions = locations + ([16, -8] if transformed else [0, 0])
        predictors = features * ([2, .5] if transformed else [1, 1]) + ([4, -2] if transformed else [0, 0])
        coefficients, spreads = [], []
        for assignment in assignments:
            design = np.column_stack([np.ones(8), predictors[assignment]])
            observed = response[assignment]
            local_coefficients = []
            for focus in range(8):
                distances = np.linalg.norm(positions-positions[focus], axis=1)
                bandwidth = np.sort(distances)[7]*1.00001 if adaptive else 3
                ratio = distances/bandwidth
                weights = np.where(ratio < 1, (1-ratio**2)**2, 0) if adaptive else np.exp(-ratio**2/2)
                local_coefficients.append(np.linalg.lstsq(np.sqrt(weights[:, None])*design, np.sqrt(weights)*observed, rcond=None)[0])
            coefficients.append(np.array(local_coefficients).reshape(-1).tolist())
            spreads.append(np.std(local_coefficients, axis=0))
        distribution = np.array(spreads)
        exceedances = (distribution[1:] >= distribution[0]).sum(axis=0)
        mean = distribution[1:].mean(axis=0)
        deviation = distribution[1:].std(axis=0)
        table = np.column_stack([distribution[0], (exceedances+1)/6, exceedances, mean, deviation, (distribution[0]-mean)/deviation])
        cases.append(dict(adaptive=adaptive, transformed=transformed, coefficients=coefficients[0], standardDeviations=distribution.reshape(-1).tolist(), table=table.reshape(-1).tolist(), summary=[5,8,8,0]))
if '--paired-refits' in sys.argv:
    # Each separated pair has positive local weights. Observed pairs 0 and 1 have x=0,1
    # and fit [intercept,slope]=[2,-1],[5,-2]. Pairs 2 and 3 have constant x and fail.
    # Both fixed assignments map equal x values into EVERY pair: their spreads are undefined.
    paired_positions = np.array([[0,0],[.25,0],[10,0],[10.25,0],[20,0],[20.25,0],[30,0],[30.25,0]])
    paired_predictors = np.array([1,0,1,0,1,1,0,0])
    paired_response = np.array([1,2,3,5,8,13,21,34])
    paired_coefficients, paired_spreads, paired_counts = [], [], []
    for assignment in assignments[:3]:
        design = np.column_stack([np.ones(8),paired_predictors[assignment]])
        estimates = np.full((8,2),np.nan)
        for focus in range(8):
            distances = np.linalg.norm(paired_positions-paired_positions[focus],axis=1)
            weights = np.where(distances < 1,(1-distances**2)**2,0)
            weighted = np.sqrt(weights[:,None])*design
            if np.linalg.matrix_rank(weighted) == 2:
                estimates[focus] = np.linalg.lstsq(weighted,np.sqrt(weights)*paired_response[assignment],rcond=None)[0]
        complete = estimates[np.isfinite(estimates).all(axis=1)]
        paired_counts.append(len(complete))
        paired_coefficients.append(estimates)
        paired_spreads.append(complete.std(axis=0) if len(complete) else np.full(2,np.nan))
    def finite_or_none(values):
        return [float(value) if np.isfinite(value) else None for value in np.asarray(values).reshape(-1)]
    print(json.dumps(dict(positions=paired_positions.reshape(-1).tolist(),predictors=paired_predictors.tolist(),response=paired_response.tolist(),
                          coefficients=finite_or_none(paired_coefficients[0]),
                          standardDeviations=finite_or_none(paired_spreads)+[None,None],
                          table=[float(paired_spreads[0][0]),None,None,None,None,None,float(paired_spreads[0][1]),None,None,None,None,None],
                          summary=[2,8,paired_counts[0],sum(8-count for count in paired_counts[1:])]),indent=2))
elif '--undefined-spreads' in sys.argv:
    # Complete null distribution is required for inference. Available simulated moments
    # remain diagnostics, never an imputed non-exceedance for an undefined simulation.
    edge_cases = []
    for name, distribution, counts in [
        ('complete', [[3,2],[1,4],[5,1]], [8,8,8]),
        ('partly undefined', [[3,2],[1,4],[None,None]], [8,8,0]),
        ('wholly undefined', [[3,2],[None,None],[None,None]], [8,0,0]),
        ('undefined observed', [[None,2],[1,4],[5,1]], [0,8,8]),
        ('ties at zero', [[0,0],[0,0],[0,0]], [8,8,8]),
        ('one undefined coefficient', [[3,2],[1,4],[None,1]], [8,8,8]),
    ]:
        values = np.array(distribution, dtype=float)
        table = []
        for column in range(2):
            observed = values[0, column]
            simulated = values[1:, column]
            valid = simulated[np.isfinite(simulated)]
            complete = np.isfinite(observed) and len(valid) == 2
            greater = int(np.sum(valid >= observed)) if complete else None
            mean = float(valid.mean()) if len(valid) else None
            deviation = float(valid.std(ddof=0)) if len(valid) else None
            table += [None if not np.isfinite(observed) else float(observed),
                      (greater + 1) / 3 if complete else None, greater,
                      mean, deviation,
                      (float(observed) - mean) / deviation if complete and deviation > 0 else None]
        edge_cases.append(dict(name=name,
                               standardDeviations=[item for row in distribution for item in row]+[None,None],
                               fitCounts=counts+[None], table=table,
                               summary=[2,8,counts[0],sum(8-count for count in counts[1:])]))
    # Invalid requests execute and publish no simulated draws or moments.
    edge_cases.append(dict(name='invalid permutation count', permutationCount=0,
                           standardDeviations=[3,2,1,4,5,1,None,None],fitCounts=[8,8,8,None],
                           table=[3,None,None,None,None,None,2,None,None,None,None,None],
                           summary=[None,8,8,0]))
    edge_cases.append(dict(name='permutation count above capacity',permutationCount=4,
                           standardDeviations=[3,2,1,4,5,1,2,3],fitCounts=[8,8,8,8],
                           table=[3,None,None,None,None,None,2,None,None,None,None,None],
                           summary=[None,8,8,0]))
    print(json.dumps(edge_cases, indent=2))
else:
    print(json.dumps(cases, indent=2))
