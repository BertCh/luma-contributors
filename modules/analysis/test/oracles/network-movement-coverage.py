# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Independent bounded optimization and Gaussian fixtures; never imports production kernels.

Reproduce: python -m pip install networkx==3.5 numpy==2.2.6
          python modules/graph/test/oracles/network-movement-coverage.py
"""
import json
from pathlib import Path
import random

import networkx as networkx
import numpy as numpy

assert networkx.__version__ == '3.5'
assert numpy.__version__ == '2.2.6'
random_generator = random.Random(1729)
flow_cases = []
for case_index in range(8):
    node_count = 3 + case_index % 4
    graph = networkx.DiGraph()
    graph.add_nodes_from(range(node_count))
    for source in range(node_count):
        for destination in range(node_count):
            if source != destination and random_generator.random() < 0.28:
                graph.add_edge(source, destination, capacity=random_generator.randrange(1, 6), weight=random_generator.randrange(0, 9))
    if not graph.edges:
        graph.add_edge(0, 1, capacity=1, weight=0)
    maximum_flow = networkx.maximum_flow_value(graph, 0, node_count - 1)
    minimum_cost_flow = networkx.max_flow_min_cost(graph, 0, node_count - 1)
    flow_cases.append({
        'name': f'directed-integral-{case_index}',
        'nodeCount': node_count,
        'edges': [[source, destination] for source, destination in graph.edges],
        'capacities': [data['capacity'] for _, _, data in graph.edges(data=True)],
        'costs': [data['weight'] for _, _, data in graph.edges(data=True)],
        'maximumFlow': maximum_flow,
        'minimumCost': networkx.cost_of_flow(graph, minimum_cost_flow),
    })

# Direct NetworkX modularity oracle on undirected self-loop/disconnected graphs.
communities = []
for edges in [[(0, 1, 1), (2, 3, 1)], [(0, 0, 2), (0, 1, 1), (1, 2, 3), (2, 3, 3)]]:
    graph = networkx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_weighted_edges_from(edges)
    partition = networkx.community.louvain_communities(graph, seed=0)
    matrix = networkx.to_numpy_array(graph)
    numpy.fill_diagonal(matrix, numpy.diag(matrix) * 2)
    communities.append({'weights': matrix.ravel().tolist(), 'modularity': networkx.community.modularity(graph, partition), 'partition': [sorted(group) for group in partition]})

# Independent dense linear-Gaussian forward and RTS reference, isotropic axes.
times = [0, 0.5, 2, 3]
observations = [[0, 1], [1, 1.2], [3, 1.5], [3.5, 1.8]]
noise, measurement_variance, velocity_variance = 0.2, 0.5, 2
states = []
covariances = []
state = numpy.array([observations[0][0], observations[0][1], 0, 0], dtype=float)
covariance = numpy.diag([measurement_variance, velocity_variance])
for observation_index, position in enumerate(observations):
    if observation_index:
        delta = times[observation_index] - times[observation_index - 1]
        transition = numpy.array([[1, delta], [0, 1]])
        process_covariance = noise * numpy.array([[delta**3 / 3, delta**2 / 2], [delta**2 / 2, delta]])
        state[:2] += delta * state[2:]
        covariance = transition @ covariance @ transition.T + process_covariance
        gain = covariance[:, 0] / (covariance[0, 0] + measurement_variance)
        innovation = numpy.array(position) - state[:2]
        state[:2] += gain[0] * innovation
        state[2:] += gain[1] * innovation
        correction = numpy.eye(2) - numpy.outer(gain, [1, 0])
        covariance = correction @ covariance @ correction.T + measurement_variance * numpy.outer(gain, gain)
    states.append(state.copy())
    covariances.append(covariance.copy())
smoothed_states = [state.copy() for state in states]
smoothed_covariances = [covariance.copy() for covariance in covariances]
for observation_index in range(len(times) - 2, -1, -1):
    delta = times[observation_index + 1] - times[observation_index]
    transition = numpy.array([[1, delta], [0, 1]])
    process_covariance = noise * numpy.array([[delta**3 / 3, delta**2 / 2], [delta**2 / 2, delta]])
    predicted_covariance = transition @ covariances[observation_index] @ transition.T + process_covariance
    gain = numpy.linalg.solve(predicted_covariance, transition @ covariances[observation_index]).T
    predicted_state = states[observation_index].copy()
    predicted_state[:2] += delta * predicted_state[2:]
    residual = smoothed_states[observation_index + 1] - predicted_state
    for axis in range(2):
        corrected = gain @ residual[[axis, axis + 2]]
        smoothed_states[observation_index][[axis, axis + 2]] += corrected
    smoothed_covariances[observation_index] += gain @ (smoothed_covariances[observation_index + 1] - predicted_covariance) @ gain.T

fixture = {
    'provenance': {
        'generator': 'network-movement-coverage.py', 'networkx': networkx.__version__, 'numpy': numpy.__version__,
        'flow': 'direct maximum_flow_value and max_flow_min_cost/cost_of_flow',
        'communities': 'direct louvain_communities(seed=0) and modularity; compare objectives/invariants, not labels',
        'kalman': 'independent NumPy dense matrix forward/Joseph/RTS composition; not a MovingPandas call',
        'limits': 'No throughput/scaling evidence; RAAM/pgRouting/MobilityDB/LCSS parity is not established.'
    },
    'flow': flow_cases, 'communities': communities,
    'kalman': {'times': times, 'positions': observations, 'parameters': [noise, measurement_variance, velocity_variance, 10],
               'filteredState': numpy.array(states).ravel().tolist(), 'filteredCovariance': numpy.array(covariances).reshape(-1).tolist(),
               'smoothedState': numpy.array(smoothed_states).ravel().tolist(), 'smoothedCovariance': numpy.array(smoothed_covariances).reshape(-1).tolist()}
}
destination = Path(__file__).with_name('fixtures') / 'network-movement-coverage.json'
destination.write_text(json.dumps(fixture, indent=2) + '\n')
print(destination)
