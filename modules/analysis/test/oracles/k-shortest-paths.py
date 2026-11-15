# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Generate independent Yen fixtures using NetworkX shortest_simple_paths.

Each CSR edge becomes an intermediate vertex, since NetworkX's k-shortest
routine does not accept multigraphs. This preserves parallel edges as distinct
simple routes and keeps original graph vertices unique in every route.
Run: /usr/bin/python3 modules/analysis/test/oracles/k-shortest-paths.py
"""
import itertools
import json
from pathlib import Path
import random
import networkx as network


def fixture(name, node_count, arcs, source, target):
    arcs = sorted(arcs, key=lambda arc: arc[0])
    offsets = [0]
    for vertex in range(node_count):
        offsets.append(offsets[-1] + sum(arc[0] == vertex for arc in arcs))
    graph = network.DiGraph()
    graph.add_nodes_from(range(node_count))
    for edge, (first, second, cost) in enumerate(arcs):
        graph.add_edge(first, node_count + edge, weight=cost)
        graph.add_edge(node_count + edge, second, weight=0)
    routes = []
    try:
        for path in itertools.islice(network.shortest_simple_paths(graph, source, target, weight='weight'), 128):
            edges = [vertex - node_count for vertex in path if vertex >= node_count]
            routes.append(dict(vertices=[vertex for vertex in path if vertex < node_count], edges=edges,
                               cost=sum(arcs[edge][2] for edge in edges)))
    except network.NetworkXNoPath:
        pass
    return dict(name=name, offsets=offsets, neighbors=[arc[1] for arc in arcs],
                weights=[arc[2] for arc in arcs], source=source, target=target, routes=routes)


cases = [
    fixture('parallel-zero-cycle-ties', 5, [(0, 1, 0), (0, 1, 0), (0, 2, 1), (1, 0, 0), (1, 2, 1), (1, 4, 2), (2, 1, 0), (2, 4, 1)], 0, 4),
    fixture('persistent-candidates', 7, [(0, 1, 1), (0, 2, 2), (0, 6, 30), (1, 3, 1), (1, 4, 3), (2, 3, 2), (2, 5, 2), (3, 6, 1), (4, 6, 4), (5, 6, 3)], 0, 6),
    fixture('identity', 3, [(0, 1, 0), (1, 0, 0)], 0, 0),
    fixture('unreachable', 3, [(0, 1, 1)], 0, 2),
    fixture('empty-slots', 4, [], 0, 3),
    fixture('large-branch-network', 700, [(0, 1, 1), (0, 350, 2), (1, 699, 1), (350, 699, 2)], 0, 699),
    fixture('long-chain', 300, [(vertex, vertex + 1, 1) for vertex in range(299)], 0, 299)
]
random_generator = random.Random(7281)
for case in range(4):
    arcs = [(first, second, random_generator.randrange(1, 17) / 4)
            for first in range(7) for second in range(7)
            if first != second and random_generator.random() < 0.35]
    cases.append(fixture(f'random-{case}', 7, arcs, 0, 6))
result = dict(reference='networkx.shortest_simple_paths on edge-subdivided CSR', version=network.__version__, cases=cases)
Path(__file__).with_name('fixtures').joinpath('k-shortest-paths.json').write_text(json.dumps(result, indent=2) + '\n')
