# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Generate sparse Louvain fixtures with NetworkX, independently of the TypeScript twin.

Run from repository root with /usr/bin/python3. NumPy is used to check the Newman
modularity matrix against NetworkX's partition score before committing each fixture.
"""
import json
from pathlib import Path
import networkx as networkx
import numpy as numpy


def make_case(name, group_count, group_size, directed=False, parallel=False, isolates=0):
    node_count = group_count * group_size + isolates
    rows = [[] for _ in range(node_count)]
    for source in range(group_count * group_size):
        for destination in range(group_count * group_size):
            if source != destination and source // group_size == destination // group_size:
                weight = 2 + source % 3 if directed else 3
                if parallel:
                    rows[source].extend([(destination, weight / 4), (destination, weight * 3 / 4)])
                else:
                    rows[source].append((destination, weight))
    for group in range(group_count - 1):
        source = (group + 1) * group_size - 1
        destination = source + 1
        rows[source].append((destination, 0.125))
        rows[destination].append((source, 0.25 if directed else 0.125))
    rows[0].append((0, 0.75))  # directed loop once / undirected loop twice
    if parallel:
        rows[0].append((0xffffffff, -123))  # missing target, ignored even with invalid weight
    return finish_case(name, node_count, rows, directed, padding=8 if parallel else 0)


def finish_case(name, node_count, rows, directed=False, padding=0):
    offsets = [0]
    neighbors = []
    weights = []
    matrix = numpy.zeros((node_count, node_count))
    for source, row in enumerate(rows):
        for destination, weight in row:
            neighbors.append(destination)
            weights.append(weight)
            if destination < node_count:
                matrix[source, destination] += weight
        offsets.append(len(neighbors))
    neighbors.extend([0] * padding)
    weights.extend([-999] * padding)  # allocated tail is outside offsets[-1]
    graph = networkx.DiGraph() if directed else networkx.Graph()
    graph.add_nodes_from(range(node_count))
    for source in range(node_count):
        for destination in range(node_count):
            weight = matrix[source, destination]
            if not weight or (not directed and destination < source):
                continue
            graph.add_edge(source, destination, weight=float(weight / 2 if not directed and source == destination else weight))
    partition = networkx.community.louvain_communities(graph, seed=17, threshold=1e-7)
    score = networkx.community.modularity(graph, partition)
    labels = numpy.empty(node_count, dtype=int)
    for community, members in enumerate(partition):
        labels[list(members)] = community
    total = matrix.sum()
    modularity_matrix = matrix / total - numpy.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / total ** 2
    numpy_score = modularity_matrix[labels[:, None] == labels[None, :]].sum()
    if not numpy.isclose(score, numpy_score, atol=1e-12):
        raise RuntimeError(f'{name}: NetworkX {score} != NumPy {numpy_score}')
    return dict(name=name, nodeCount=node_count, directed=directed, offsets=offsets,
                neighbors=neighbors, weights=weights, resolution=1, modularity=score,
                communities=labels.tolist())


cases = [
    make_case('96-weighted', 8, 12),
    make_case('48-directed', 8, 6, directed=True),
    make_case('parallel-padding-isolates', 10, 4, parallel=True, isolates=4),
    finish_case('coarsening', 4, [
        [(1, 2), (2, 1.25), (3, 1.25)], [(0, 2), (2, 1.25), (3, 1.25)],
        [(0, 1.25), (1, 1.25), (3, 2)], [(0, 1.25), (1, 1.25), (2, 2)]
    ])
]
Path('modules/analysis/test/oracles/fixtures/network-louvain-sparse.json').write_text(
    json.dumps(dict(networkxVersion=networkx.__version__, cases=cases), indent=2) + '\n'
)
for case in cases:
    print(case['name'], case['nodeCount'], len(case['neighbors']), case['modularity'])
