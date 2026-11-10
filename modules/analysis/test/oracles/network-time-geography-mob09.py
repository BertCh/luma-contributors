# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference network potential path areas and static travel times for MOB-09 from networkx.

Writes fixtures/network-time-geography-mob09.json next to this script (consumed by
test/gpu-network/network-time-geography/gpu-network-time-geography.spec.ts). Run from anywhere with
    <venv>/bin/python -I modules/analysis/test/oracles/network-time-geography-mob09.py
in a venv with numpy and networkx (versions are recorded in the fixture).

Reference: networkx.single_source_dijkstra_path_length from the origin anchor on the network and
from the destination anchor on its reverse. An edge (u, v) is in the prism when
d_a(u) + w + d_b(v) <= budget, a node when d_a(v) + d_b(v) <= budget. The generator moves each
budget so that no edge or node sum lies within 1e-3 of it, so the float32 sums of the GPU cannot
flip a member. The static travel times are d_a(v) of the same network (the travel time of every
departure when the edge durations do not depend on time). Weights are float32 values.
"""

import json
import os

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'network-time-geography-mob09.json')
MARGIN = 1e-3


def build_scene(seed, node_count, edge_count):
    random = np.random.default_rng(seed)
    edges = {}
    while len(edges) < edge_count:
        source, target = (int(value) for value in random.integers(0, node_count, 2))
        if source != target:
            edges[(source, target)] = float(np.float32(random.uniform(1, 5)))
    # CSR order: by source, then insertion order.
    ordered = sorted(edges.items(), key=lambda item: item[0][0])
    return [(source, target, weight) for (source, target), weight in ordered]


def costs(node_count, edges, origin, reverse):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(node_count))
    for source, target, weight in edges:
        if reverse:
            graph.add_edge(target, source, weight=weight)
        else:
            graph.add_edge(source, target, weight=weight)
    lengths = nx.single_source_dijkstra_path_length(graph, origin, weight='weight')
    return [lengths.get(node) for node in range(node_count)]


def prism(node_count, edges, forward, backward, budget):
    edge_flags = []
    sums = []
    for source, target, weight in edges:
        if forward[source] is None or backward[target] is None:
            edge_flags.append(0)
            continue
        total = forward[source] + weight + backward[target]
        sums.append(total)
        edge_flags.append(int(total <= budget))
    node_flags = []
    for node in range(node_count):
        if forward[node] is None or backward[node] is None:
            node_flags.append(0)
            continue
        total = forward[node] + backward[node]
        sums.append(total)
        node_flags.append(int(total <= budget))
    return edge_flags, node_flags, sums


def main():
    cases = []
    for seed, node_count, edge_count in ((1, 40, 160), (2, 80, 300), (3, 120, 360)):
        edges = build_scene(seed, node_count, edge_count)
        origin, destination = 0, node_count - 1
        forward = costs(node_count, edges, origin, False)
        backward = costs(node_count, edges, destination, True)
        shortest = forward[destination]
        assert shortest is not None, f'seed {seed}: destination unreachable'
        # A budget 40% above the shortest path, moved off any sum.
        budget = shortest * 1.4
        _, _, sums = prism(node_count, edges, forward, backward, budget)
        while any(abs(total - budget) <= MARGIN for total in sums):
            budget += 2 * MARGIN
        budget = float(np.float32(budget))
        edge_flags, node_flags, sums = prism(node_count, edges, forward, backward, budget)
        assert all(abs(total - budget) > MARGIN / 2 for total in sums)
        assert 0 < sum(edge_flags) < len(edges), 'the prism must be a proper part of the network'
        cases.append(
            {
                'seed': seed,
                'nodeCount': node_count,
                'edges': [[source, target, weight] for source, target, weight in edges],
                'origin': origin,
                'destination': destination,
                'budget': budget,
                'forward': forward,
                'backward': backward,
                'edgeInPrism': edge_flags,
                'nodeInPrism': node_flags,
            }
        )
    with open(OUTPUT, 'w') as handle:
        json.dump(
            {
                'versions': {'networkx': nx.__version__, 'numpy': np.__version__},
                'cases': cases,
            },
            handle,
        )


main()
