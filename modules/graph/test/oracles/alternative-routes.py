# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NetworkX 3.5 independent simple EDGE path oracle; dyadic costs are exact f32.

all_simple_edge_paths handles parallel edge keys. Sort complete enumerations by
cost then CSR edge sequence, independently of the GPU priority queue.
"""
import json
from pathlib import Path
import networkx as network


def fixture(name, offsets, neighbors, weights, source, target):
    graph = network.MultiDiGraph()
    graph.add_nodes_from(range(len(offsets) - 1))
    for vertex in graph:
        for edge in range(offsets[vertex], offsets[vertex + 1]):
            graph.add_edge(vertex, neighbors[edge], key=edge, weight=weights[edge])
    paths = [[]] if source == target else list(network.all_simple_edge_paths(graph, source, target))
    routes = []
    for path in paths:
        edges = [edge[2] for edge in path]
        vertices = [source] + [edge[1] for edge in path]
        routes.append(dict(edges=edges, vertices=vertices, cost=sum(weights[edge] for edge in edges)))
    routes.sort(key=lambda route: (route['cost'], route['edges']))
    return dict(name=name, offsets=offsets, neighbors=neighbors, weights=weights,
                source=source, target=target, routes=routes)

cases = [
    fixture('zero-cycle-parallel-ties', [0, 3, 6, 8, 8], [1, 1, 2, 0, 2, 3, 1, 3], [0, 0, 1, 0, 1, 2, 0, 1], 0, 3),
    fixture('updated-costs', [0, 3, 6, 8, 8], [1, 1, 2, 0, 2, 3, 1, 3], [4, 0, 0.5, 0, 1, 2, 0, 1], 0, 3),
    fixture('identity', [0, 1, 2], [1, 0], [0, 0], 0, 0),
    fixture('unreachable', [0, 1, 1, 1], [1], [1], 0, 2),
    fixture('expensive-first-layered', [0, 3, 5, 7, 9, 11, 13, 15, 16, 17, 17],
            [1, 2, 9, 3, 4, 3, 4, 5, 6, 5, 6, 7, 8, 7, 8, 9, 9],
            [100, 100, 1] + [1] * 14, 0, 9)
]
result = dict(reference='networkx.all_simple_edge_paths; complete enumeration then cost/CSR-edge lexicographic sort',
              version=network.__version__, cases=cases)
Path(__file__).with_name('fixtures').joinpath('alternative-routes.json').write_text(json.dumps(result, indent=2) + '\n')
