# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geographic-distribution/geographic-distribution-audit.oracle.py
"""Population weighted covariance via numpy.cov(aweights=..., ddof=0), eigh,
and circular unit-vector moments. Medians are analytic: symmetry fixes group 0
at the origin; the midpoint is the mean-start convention for group 1's tied
two-point median. Empty groups publish NaN except zero counts/weights/flags.
"""
import json
import numpy as np

positions = np.array([[-2, 0], [2, 0], [0, -1], [0, 1], [0, 0], [10, 0], [14, 0], [999, 999], [np.nan, 0]])
ends = np.array([[0, 0], [2, 3], [np.nan, 0], [0, 1], [-4, 0], [11, 0], [13, 0], [1000, 1000], [0, 0]])
weights = np.array([1, 1, 1, 1, 4, 1, 1, 1, 1])
groups = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0])
mask = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=bool)
result = {port: [] for port in ['counts', 'weightSums', 'meanCenters', 'medianCenters', 'medianConverged', 'standardDistances', 'ellipses', 'directionalMeans', 'ellipseVertices', 'circleVertices']}
for group in range(3):
    included = (groups == group) & mask & np.isfinite(positions).all(axis=1)
    points = positions[included]
    mass = weights[included]
    result['counts'].append(len(points))
    result['weightSums'].append(float(mass.sum()))
    result['medianConverged'].append(int(len(points) > 0))
    if len(points) == 0:
        for port, count in [('meanCenters', 2), ('medianCenters', 2), ('standardDistances', 1), ('ellipses', 3), ('directionalMeans', 3), ('ellipseVertices', 8), ('circleVertices', 8)]:
            result[port].extend([float('nan')] * count)
        continue
    center = np.average(points, axis=0, weights=mass)
    covariance = np.cov(points.T, aweights=mass, ddof=0)
    variances, vectors = np.linalg.eigh(covariance)
    short_direction = vectors[:, 0]
    angle = np.arctan2(short_direction[1], short_direction[0])
    if angle > np.pi/2:
        angle -= np.pi
    if angle <= -np.pi/2:
        angle += np.pi
    axes = np.sqrt(variances)
    distance = np.sqrt(np.trace(covariance))
    result['meanCenters'].extend(center)
    result['medianCenters'].extend(center)
    result['standardDistances'].append(distance)
    result['ellipses'].extend([angle, *axes])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for parameter in np.arange(4) * np.pi/2:
        unit = np.array([np.cos(parameter), np.sin(parameter)])
        result['ellipseVertices'].extend(center + rotation @ (axes * unit))
        result['circleVertices'].extend(center + distance * unit)
    lines = ends[included] - points
    lengths = np.linalg.norm(lines, axis=1)
    valid = np.isfinite(lengths) & (lengths > 0)
    unit = lines[valid] / lengths[valid, None]
    resultant = np.average(unit, axis=0, weights=mass[valid])
    direction = np.arctan2(resultant[1], resultant[0]) if np.linalg.norm(resultant) > 0 else float('nan')
    result['directionalMeans'].extend([direction, 1-np.linalg.norm(resultant), np.average(lengths[valid], weights=mass[valid])])
print(json.dumps(result))
print('axial directionalMeans:', [0, 1/3, 3.5, 0, 0, 1, float('nan'), float('nan'), float('nan')])
# Coincident-median fixture: positions [0,10,-5], weights [3,1,2].
# Mean=0; noncoincident pull is 1-2=-1, coincident mass=3 >= abs(pull),
# hence zero is the exact weighted geometric median and one corrected step is zero.
print('coincident median:', [0, 0], 'converged:', [1])

# An integer-coordinate similarity transform: sqrt(2) times a +45 degree rotation.
# Population covariance/eigh independently check a positive nonzero cross-moment.
rotation = np.array([[1, -1], [1, 1]])
for group in [0, 1]:
    included = (groups == group) & mask & np.isfinite(positions).all(axis=1)
    transformed = positions[included] @ rotation.T
    covariance = np.cov(transformed.T, aweights=weights[included], ddof=0)
    variances, vectors = np.linalg.eigh(covariance)
    angle = np.arctan2(vectors[1, 0], vectors[0, 0])
    if angle > np.pi/2:
        angle -= np.pi
    if angle <= -np.pi/2:
        angle += np.pi
    print('rotated group', group, 'ellipse:', [angle, *np.sqrt(variances)], 'standardDistance:', np.sqrt(np.trace(covariance)))
# [10,14] with weights [1,3] has mean 13. Pull -1+3=2 and inverse-distance
# mass 1/3+3=10/3 give the first Weiszfeld iterate 13+2/(10/3)=13.6.
print('asymmetric one-step median:', [13.6, 0], 'mean:', [13, 0], 'converged:', [0])
