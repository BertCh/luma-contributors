# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run with /usr/bin/python3; independent references for the sibling live lens spec."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from shapely.geometry import Polygon
from shapely.ops import unary_union

data = Path(__file__).resolve().parents[5] / 'modules/lenses/src/computation-lenses/data'
rings = json.loads((data / 's3-chicago-community-areas.json').read_text())
points = np.array(rings['positionsProjected'], dtype=np.float32).reshape(-1, 2)
offsets = rings['ringOffsets']
polygons = [Polygon(points[first:last]) for first, last in zip(offsets, offsets[1:])]
union = unary_union(polygons)
print('area sum, union:', sum(p.area for p in polygons), union.area)
print('union type, holes, perimeter:', union.geom_type, len(union.interiors), union.length)
# Polygon, 0 holes, 80373.20290219603 m perimeter (uncancelled sum: 220150.9001818201 m).

tracts = json.loads((data / 's1-chicago-tracts.json').read_text())
names = ['diabetes', 'poverty150Pct', 'noVehiclePct', 'uninsuredPct']
values = np.array([tracts['columns'][name] for name in names], dtype=np.float32).T
kept = np.isfinite(values).all(axis=1)
vertices = np.zeros((len(tracts['deltas']) // 2, 2), dtype=np.float32)
deltas = np.array(tracts['deltas']).reshape(-1, 2)
for first, last in zip(tracts['ringOffsets'], tracts['ringOffsets'][1:]):
    vertices[first:last] = np.array(tracts['origin']) + np.cumsum(deltas[first:last], axis=0) * tracts['quantum']
# The loader uploads float32 coordinates. Default queen tolerance=0 means shared exact vertices.
sets = [set(map(tuple, vertices[a:b])) for a, b in zip(tracts['ringOffsets'], tracts['ringOffsets'][1:])]
sets = [s for s, keep in zip(sets, kept) if keep]
adjacency = np.array([[i != j and bool(a & b) for j, b in enumerate(sets)] for i, a in enumerate(sets)], dtype=float)
degrees = adjacency.sum(axis=1)[:, None]
weights = np.divide(adjacency, degrees, out=np.zeros_like(adjacency), where=degrees != 0)
response = values[kept, 0].astype(float)
design = np.column_stack((np.ones(sum(kept)), values[kept, 1:])).astype(float)
lag = weights @ response
beta0 = np.linalg.lstsq(design, response, rcond=None)[0]
beta1 = np.linalg.lstsq(design, lag, rcond=None)[0]
e0, e1 = response - design @ beta0, lag - design @ beta1
powers = np.arange(1, 129)
traces = np.real((np.linalg.eigvals(weights)[:, None] ** powers).sum(axis=0))

def objective(rho):
    residuals = e0 - rho * e1
    return len(response) / 2 * np.log(residuals @ residuals / len(response)) + np.sum(rho ** powers * traces / powers)

fit = minimize_scalar(objective, bounds=(-0.9, 0.9), method='bounded', options={'xatol': 1e-12})
print('rows, directed edges, rho, beta:', sum(kept), adjacency.sum(), fit.x, beta0 - fit.x * beta1)
# 788 rows, 5168 edges; rho=0.6499915484241714;
# beta=[0.78834312, 0.14601426, -0.01152027, -0.00772924]. The optimum is interior.
