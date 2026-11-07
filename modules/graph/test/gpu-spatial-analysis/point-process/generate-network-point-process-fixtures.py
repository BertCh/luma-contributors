# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NetworkX routing plus independently enumerated quarter-length network locations.

All fixtures have integral edge lengths and integral event distances. Quarter-length
samples therefore contain every metric-sphere crossing exactly; vertices have one identity.
No production kernel or its edge-root formulas are used to generate expected K values.
Run with networkx 3.5. The formula follows spatstat.linnet linearKinhom (normalise=FALSE).
"""
from fractions import Fraction
import json
from pathlib import Path
import networkx

cases = [
    dict(name='junction-disconnected', nodeCount=6, edges=[[0, 1, 2], [0, 2, 3], [0, 3, 4], [4, 5, 2]], events=[0, 1, 3, 4], intensity=[0.5, 1, 2, 0.25], radii=[0, 1, 2, 3, 4, 6, 10]),
    dict(name='cycle-branch', nodeCount=4, edges=[[0, 1, 2], [1, 2, 2], [2, 0, 2], [2, 3, 1]], events=[0, 1, 3], intensity=[1, 2, 0.5], radii=[0, 1, 2, 3, 4]),
    dict(name='coincident-events', nodeCount=2, edges=[[0, 1, 2]], events=[0, 0, 1], intensity=[1, 2, 1], radii=[0, 1, 2, 3])
]
for case in cases:
    graph = networkx.Graph()
    graph.add_nodes_from(range(case['nodeCount']))
    graph.add_weighted_edges_from(case['edges'])
    distances = dict(networkx.all_pairs_dijkstra_path_length(graph))
    event_distances = [distances[event] for event in case['events']]
    case['nodeDistances'] = [row.get(node) for row in event_distances for node in range(case['nodeCount'])]
    case['shellCounts'] = []
    case['numeratorNone'] = []
    case['numeratorAng'] = []
    for source, source_node in enumerate(case['events']):
        locations = {('vertex', node): Fraction(distance) for node, distance in distances[source_node].items()}
        for edge_index, (first, second, length) in enumerate(case['edges']):
            if first not in distances[source_node]:
                continue
            for quarter in range(1, length * 4):
                offset = Fraction(quarter, 4)
                locations[('edge', edge_index, quarter)] = min(distances[source_node][first] + offset, distances[source_node][second] + length - offset)
        for destination in case['events']:
            radius = distances[source_node].get(destination)
            case['shellCounts'].append(sum(value == radius for value in locations.values()) if radius is not None else 0)
    for radius in case['radii']:
        uncorrected = Fraction(0)
        corrected = Fraction(0)
        for source, source_node in enumerate(case['events']):
            for destination, destination_node in enumerate(case['events']):
                distance = distances[source_node].get(destination_node)
                if source == destination or distance is None or distance > radius:
                    continue
                weight = 1 / (Fraction(case['intensity'][source]) * Fraction(case['intensity'][destination]))
                uncorrected += weight
                corrected += weight / case['shellCounts'][source * len(case['events']) + destination]
        case['numeratorNone'].append(float(uncorrected))
        case['numeratorAng'].append(float(corrected))
    length = sum(edge[2] for edge in case['edges'])
    inverse_sum = sum(1 / Fraction(value) for value in case['intensity'])
    case['normalizations'] = [float(length), float(inverse_sum), float(inverse_sum * inverse_sum / length)]
    case['kAng'] = [value / length for value in case['numeratorAng']]
    case['kNone'] = [value / length for value in case['numeratorNone']]
Path(__file__).with_name('network-point-process-fixtures.json').write_text(json.dumps({'reference': {'networkx': networkx.__version__, 'sphereGridStep': '1/4', 'exactForFixtures': True}, 'cases': cases}, indent=2) + '\n')
