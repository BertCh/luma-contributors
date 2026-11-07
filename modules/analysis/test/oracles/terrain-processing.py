# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent composed references. Run with numpy==2.2.6, networkx==3.5."""
import json
from pathlib import Path
import networkx as network
import numpy as numpy

assert numpy.__version__ == '2.2.6'
assert network.__version__ == '3.5'

diamond = network.DiGraph()
diamond.add_weighted_edges_from([(0, 1, 0.25), (0, 2, 0.75), (1, 3, 1), (2, 3, 1)])
lengths = {(0, 1): 1, (0, 2): 3, (1, 3): 1, (2, 3): 1}
injection = [8, 0, 0, 0]
transport = injection.copy()
for cell in network.topological_sort(diamond):
    for receiver in diamond.successors(cell):
        transport[receiver] += transport[cell] * diamond[cell][receiver]['weight']
paths = list(network.all_simple_paths(diamond, 0, 3))
path_lengths = [sum(lengths[edge] for edge in zip(path[:-1], path[1:])) for path in paths]
path_probabilities = [float(numpy.prod([diamond[first][second]['weight'] for first, second in zip(path[:-1], path[1:])])) for path in paths]
residuals = numpy.array([2, 2, 2, 100, 5.0])
stable = residuals[:4]
bias = float(numpy.median(stable))
nmad = float(numpy.median(numpy.abs(stable - bias)) * 1.4826022)
correlation_length = 2.0
coordinates = numpy.arange(5)
covariance = numpy.exp(-numpy.abs(coordinates[:, None] - coordinates[None, :]) / correlation_length)
result = {
    'provenance': {'kind': 'composed independent oracle', 'numpy': numpy.__version__, 'networkx': network.__version__, 'claims': 'DAG path enumeration, median/NMAD and covariance arithmetic; not upstream hydrology parity'},
    'diamond': {'transport': transport, 'expected': sum(length * probability for length, probability in zip(path_lengths, path_probabilities)), 'minimum': min(path_lengths), 'maximum': max(path_lengths)},
    'vertical': {'bias': bias, 'nmad': nmad, 'corrected': (residuals-bias).tolist(), 'volume': float(numpy.sum(residuals-bias)), 'independentStandardDeviation': float(numpy.sqrt(5)), 'correlatedStandardDeviation': float(numpy.sqrt(numpy.sum(covariance)))},
    'nested': {'elevation': [0, 2, 1], 'stages': [1, 3], 'rootCurves': [2, 1, 3, 6]}
}
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
