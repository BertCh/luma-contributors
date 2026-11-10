# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent fixtures: numpy 2.2.6, scipy 1.15.3, shapely 2.1.2.
Run with /tmp/luma-coverage-oracles/bin/python from repository root.
Equations: covariance constrained kriging; SciPy RBFInterpolator; Shapely Voronoi areas.
"""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares
from shapely import MultiPoint, Point, box, voronoi_polygons

positions = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [.6, .7], [1.4, 1.2]], dtype=float)
values = np.array([1, 3, 2, 4, 2.2, 1.5])
queries = np.array([[.8, 1.1], [1.1, .4], [.6, .7], [2.3, 1.1]])
errors = np.array([.1, .2, .05, .3, .15, .25])
covariance = dict(model='exponential', range=3., sill=2., nugget=.1)

def covariance_matrix(first, second):
    separation = np.linalg.norm(first[:, None] - second[None], axis=2)
    return 2 * np.exp(-separation) + .1 * (separation == 0)

def kriging(mode, block=False, co=False):
    results=[]
    factors=np.array([[1., 0.], [.4, .8]])
    coregionalization=factors @ factors.T
    variables=np.array([0,1,0,1,0,1]) if co else np.zeros(6,dtype=int)
    target=1 if co else 0
    means=np.array([1.2,2.3])
    matrix=covariance_matrix(positions, positions)
    matrix*=coregionalization[variables[:,None],variables[None]] if co else 1
    matrix+=np.diag(errors)
    for query in queries:
        supports=np.array([query+[-.1,0],query+[.1,0]]) if block else np.array([query])
        cross=covariance_matrix(positions,supports).mean(axis=1)
        cross*=coregionalization[variables,target] if co else 1
        variance=covariance_matrix(supports,supports).mean()*(coregionalization[target,target] if co else 1)
        design=None
        if mode=='ordinary':design=np.ones((6,1));target_design=np.ones(1)
        if mode=='universal':design=np.c_[np.ones(6),positions];target_design=np.r_[1,query]
        if mode=='external-drift':design=np.c_[np.ones(6),positions[:,0]**2];target_design=np.r_[1,(supports[:,0]**2).mean()]
        if design is not None:
            count=design.shape[1]
            system=np.block([[matrix,design],[design.T,np.zeros((count,count))]])
            right=np.r_[cross,target_design]
            weights=np.linalg.solve(system,right)
            prediction=weights[:6]@values
            variance-=weights@right
        else:
            weights=np.linalg.solve(matrix,cross)
            prediction=means[target]+weights@(values-means[variables])
            variance-=weights@cross
        results.append([prediction,variance])
    return results

old_cells=voronoi_polygons(MultiPoint(positions),extend_to=box(-20,-20,20,20),ordered=True)
new_sites=np.vstack([positions,queries[0]])
new_cells=voronoi_polygons(MultiPoint(new_sites),extend_to=box(-20,-20,20,20),ordered=True)
query_cell=new_cells.geoms[-1]
areas=np.array([query_cell.intersection(cell).area for cell in old_cells.geoms])
weights=areas/areas.sum()
base=covariance_matrix(positions,positions)+np.diag(errors)
cross=covariance_matrix(positions,queries[:2])
conditional_mean=1.2+cross.T@np.linalg.solve(base,values-1.2)
conditional_covariance=covariance_matrix(queries[:2],queries[:2])-cross.T@np.linalg.solve(base,cross)
fixture=dict(positions=positions.reshape(-1).tolist(),values=values.tolist(),queries=queries.reshape(-1).tolist(),errors=errors.tolist(),covariance=covariance,
             kriging={mode:kriging(mode) for mode in ['simple','ordinary','universal','external-drift']},block=kriging('ordinary',True),co=kriging('simple',co=True),
             rbf={kernel:RBFInterpolator(positions,values,kernel=kernel,epsilon=.8,smoothing=.02,neighbors=6)(queries).tolist() for kernel in ['thin_plate_spline','cubic','gaussian','linear']},
             natural=dict(weights=weights.tolist(),value=float(weights@values)),simulation=dict(mean=conditional_mean.tolist(),covariance=conditional_covariance.tolist()))
variogram_distances = np.arange(1, 21) * .2
variogram_counts = np.arange(10, 30)
variogram_fits = {}
for family in ['spherical', 'exponential', 'gaussian']:
    def shape(scaled):
        if family == 'spherical': return np.where(scaled >= 1, 1, 1.5 * scaled - .5 * scaled ** 3)
        if family == 'exponential': return 1 - np.exp(-3 * scaled)
        return 1 - np.exp(-3 * scaled ** 2)
    semivariances = .2 + 1.5 * shape(variogram_distances / 2.2) + .015 * np.sin(np.arange(20) * 1.7)
    def residual(parameters):
        return np.sqrt(variogram_counts) * (semivariances - parameters[0] - parameters[1] * shape(variogram_distances / parameters[2]))
    fitted = least_squares(residual, [.1, 1, 2], bounds=([0, 0, .02], [np.inf, np.inf, 12]), xtol=1e-13, ftol=1e-13, gtol=1e-13)
    variogram_fits[family] = dict(semivariances=semivariances.tolist(), parameters=fitted.x.tolist(), residual=float(residual(fitted.x) @ residual(fitted.x)))
fixture['variogram'] = dict(distances=variogram_distances.tolist(), pairCounts=variogram_counts.tolist(), fits=variogram_fits)
text='// luma.gl\n// SPDX-License-Identifier: MIT\n// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors\n\n// Generated by test/oracles/interpolation-coverage.py with pinned independent scientific libraries.\nexport const INTERPOLATION_COVERAGE_REFERENCE = '+json.dumps(fixture,indent=2)+' as const;\n'
Path('modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/interpolation-coverage-reference.ts').write_text(text)
