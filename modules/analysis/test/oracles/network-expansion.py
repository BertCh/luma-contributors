# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Generate independent small graph and facility reference fixtures.

Requires NetworkX 3.x; p-center uses exhaustive itertools combinations rather than
sharing the GPU subset-ranking implementation. Float inputs are exactly representable.
"""
import itertools
import json
import math
from pathlib import Path
import networkx as network

undirected = network.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
undirected.add_node(5)
directed = network.DiGraph([(0, 1), (1, 0), (2, 1), (2, 3), (3, 4), (4, 3)])
directed.add_node(5)
labels = list(range(6))
for component in network.strongly_connected_components(directed):
    for vertex in component:
        labels[vertex] = min(component)
routes = network.DiGraph()
routes.add_weighted_edges_from([(0, 1, 1), (0, 2, 2), (0, 3, 5), (1, 2, 0.5), (1, 3, 2), (2, 3, 1)])
paths = list(itertools.islice(network.shortest_simple_paths(routes, 0, 3, weight='weight'), 3))
chain = network.DiGraph([(0, 1), (1, 2)])
costs = [[1, 4, 5], [2, 1, 4], [5, 3, 1]]
subsets = list(itertools.combinations(range(3), 2))
objectives = [max(min(row[facility] for facility in subset) for row in costs) for subset in subsets]
best = subsets[objectives.index(min(objectives))]
random_cases = []
for seed in range(8):
    graph = network.gnp_random_graph(8, 0.28, seed=seed)
    offsets = [0]
    neighbors = []
    slots = {}
    for vertex in range(8):
        for neighbor in sorted(graph.neighbors(vertex)):
            slots[(vertex, neighbor)] = len(neighbors)
            neighbors.append(neighbor)
        offsets.append(len(neighbors))
    reverse = [0] * len(neighbors)
    bridges = [0] * len(neighbors)
    blocks = [0xffffffff] * len(neighbors)
    for (first, second), slot in slots.items():
        reverse[slot] = slots[(second, first)]
    for first, second in network.bridges(graph):
        bridges[slots[(first, second)]] = bridges[slots[(second, first)]] = 1
    for component in network.biconnected_component_edges(graph):
        component_slots = [slots[pair] for edge in component for pair in [edge, edge[::-1]]]
        for slot in component_slots:
            blocks[slot] = min(component_slots)
    random_cases.append({'offsets': offsets, 'neighbors': neighbors, 'reverse': reverse,
                         'bridges': bridges, 'blocks': blocks,
                         'articulation': [int(vertex in set(network.articulation_points(graph))) for vertex in range(8)]})
lcss_cases = []
for first, second, tolerance in [
    ([0, 1, 2], [0, 1, 2], 0),
    ([0, 2, 4], [0, 1, 2, 3, 4], 0),
    ([0, 1, 2], [2, 1, 0], 0),
    ([0, 0, 1], [0, 1, 1], 0),
    ([0, 1], [0.25, 2], 0.5),
    ([0, 1], [4, 5], 0),
]:
    # Independent combinatorial oracle: matching pairs are a DAG; a longest chain
    # is an LCSS, without copying the GPU dynamic-program recurrence.
    matches = [(left, right) for left in range(len(first)) for right in range(len(second))
               if math.fabs(first[left] - second[right]) <= tolerance]
    graph = network.DiGraph()
    graph.add_nodes_from(matches)
    graph.add_edges_from((left, right) for left in matches for right in matches
                         if left[0] < right[0] and left[1] < right[1])
    length = len(network.dag_longest_path(graph)) if matches else 0
    lcss_cases.append({'first': first, 'second': second, 'tolerance': tolerance,
                       'similarity': length / min(len(first), len(second))})
result = {
    'reference': {'networkx': network.__version__, 'facility': 'Python itertools exhaustive subsets'},
    'lcss': lcss_cases,
    'randomUndirected': random_cases,
    'strongComponents': labels,
    'articulationPoints': sorted(network.articulation_points(undirected)),
    'routeCosts': sorted(sum(routes[first][second]['weight'] for first, second in zip(path, path[1:])) for path in paths),
    'edgeBetweenness': list(network.edge_betweenness_centrality(chain, normalized=False).values()),
    'pCenter': {'selection': list(best), 'objective': min(objectives)},
    # Exhaust all assignments independently of the GPU's demand-order greedy rule.
    # Greedy serves the first unit at facility 0 and then pays 100 for the second;
    # swapping the assignments gives the exact objective 3.
    'capacitated': {
        'costs': [1, 2, 1, 100], 'demand': [1, 1], 'capacities': [1, 1],
        'exactObjective': min(
            sum([[1, 2], [1, 100]][row][facility] for row, facility in enumerate(assignment))
            for assignment in itertools.product(range(2), repeat=2)
            if all(assignment.count(facility) <= 1 for facility in range(2))
        ),
    },
    # Parallel-edge fixture: 3--4 has two distinct edges, so neither is a bridge.
    'multigraphBridges': [[2, 3]],
}
Path(__file__).with_name('fixtures').joinpath('network-expansion.json').write_text(json.dumps(result, indent=2) + '\n')
