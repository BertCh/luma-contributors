# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent analytical, SciPy 1.15.3, Shapely 2.1.2, NetworkX 3.5 fixtures.
Run: /tmp/luma-coverage-oracles/bin/python modules/analysis/test/oracles/urban-ecology.py
"""
import json
import math
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import solve
import shapely
from shapely.geometry import LineString, Point, Polygon
import networkx as nx

edges = [(0, 1), (1, 2), (0, 2)]
conductances = [1, 1, 0.5]
laplacian = np.zeros((3, 3))
for (first, second), conductance in zip(edges, conductances):
    laplacian[first, first] += conductance
    laplacian[second, second] += conductance
    laplacian[first, second] -= conductance
    laplacian[second, first] -= conductance
voltages = np.r_[solve(laplacian[:2, :2], [1, 0]), 0]
box = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])
endpoints = []
for ray in range(8):
    angle = ray * math.tau / 8
    intersection = box.boundary.intersection(LineString([(0, 0), (10 * math.cos(angle), 10 * math.sin(angle))]))
    endpoints.append(list(intersection.coords)[0])
visibility = nx.path_graph(3)
cohesion = 100 * (1 - 8 / (8 * math.sqrt(4))) / (1 - 1 / math.sqrt(8))
contagion = 100 * (1 + sum(value * math.log(value) for value in [.4, .1, .1, .4]) / (2 * math.log(2)))
fixture = {
    'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'shapely': shapely.__version__, 'networkx': nx.__version__},
    'landscape': {'classMetrics': [4, 8, 2, cohesion] * 2, 'classAdjacency': [8, 2, 2, 8], 'aggregation': [100, 100], 'landscape': [contagion, 4, 8, 2]},
    'resistance': {'edges': np.array(edges).flatten().tolist(), 'conductances': conductances, 'potentials': voltages.tolist(), 'currents': [conductance * (voltages[first] - voltages[second]) for (first, second), conductance in zip(edges, conductances)], 'resistance': float(voltages[0])},
    'isovist': {'endpoints': np.array(endpoints).flatten().tolist(), 'measures': [Polygon(endpoints).area, Polygon(endpoints).length, max(Point(point).distance(Point(0, 0)) for point in endpoints), 1]},
    'visibility': {'choice': list(nx.betweenness_centrality(visibility, normalized=False).values()), 'metrics': [[visibility.degree(vertex), 2, sum(nx.single_source_shortest_path_length(visibility, vertex).values()), nx.closeness_centrality(visibility, vertex)] for vertex in visibility]},
    'decomposition': {'populations': [[9,1,1,9],[8,2,2,8],[7,3,3,7],[5,5,5,5]], 'indices': [.8,.6,.4,0], 'decomposition': [-.8,-.3,-.5,0]},
    'tessellation': {'enclosure': [-3,-2,3,-2,3,2,-3,2], 'segments': [-2,-.5,-1,-.5, -1,-.5,-1,.5, -1,.5,-2,.5, -2,.5,-2,-.5, 1,-.5,2,-.5, 2,-.5,2,.5, 2,.5,1,.5, 1,.5,1,-.5], 'owners':[0,0,0,0,1,1,1,1], 'areas':[12,12]}
}
Path(__file__).with_name('fixtures').joinpath('urban-ecology.json').write_text(json.dumps(fixture, indent=2) + '\n')
