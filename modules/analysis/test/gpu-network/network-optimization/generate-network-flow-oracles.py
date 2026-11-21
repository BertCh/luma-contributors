#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Recompute the three external flow records in a separate output directory.

See network-flow-oracles.md for versions, deterministic construction and seed provenance.
The explicit primal-dual input manifest is independent of the expected-value records.
"""

import argparse
import json
from pathlib import Path
import platform

import networkx
import numpy
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow


FIXTURE_DIRECTORY = Path(__file__).resolve().parent


def createScipyRecord(nodeCount, sources, targets):
    capacities = csr_matrix(
        (numpy.ones(len(sources), dtype=numpy.int64), (sources, targets)),
        shape=(nodeCount, nodeCount),
    )
    result = maximum_flow(capacities, 0, nodeCount - 1, method='dinic')
    residual = (capacities - result.flow).tocsr()
    # Reachability uses strictly positive residual capacity, including reverse residual arcs.
    reached = numpy.zeros(nodeCount, dtype=bool)
    reached[0] = True
    queue = [0]
    for source in queue:
        for slot in range(residual.indptr[source], residual.indptr[source + 1]):
            target = residual.indices[slot]
            if residual.data[slot] > 0 and not reached[target]:
                reached[target] = True
                queue.append(int(target))
    cutCapacity = sum(
        1 for source, target in zip(sources, targets) if reached[source] and not reached[target]
    )
    assert cutCapacity == result.flow_value
    return {
        'maximumFlow': int(result.flow_value),
        'cutCapacity': int(cutCapacity),
        'sourceReachable': numpy.flatnonzero(reached).tolist(),
    }


def createFanRecord():
    nodeCount = 100000
    leaves = numpy.arange(1, nodeCount - 1, dtype=numpy.int32)
    # Match the spec's original edge rows: all source->leaf rows, then all leaf->sink rows.
    sources = numpy.concatenate((numpy.zeros(len(leaves), dtype=numpy.int32), leaves))
    targets = numpy.concatenate((leaves, numpy.full(len(leaves), nodeCount - 1, dtype=numpy.int32)))
    return {
        'oracle': 'SciPy scipy.sparse.csgraph.maximum_flow (Dinic), unit capacities',
        'nodeCount': nodeCount,
        'edgeCount': len(sources),
        **createScipyRecord(nodeCount, sources, targets),
    }


def createGridRecord():
    rows, columns = 250, 400
    nodeCount = rows * columns
    sources, targets = [], []
    # Row-major vertices, outgoing right edge before outgoing down edge. No wraparound.
    for source in range(nodeCount):
        row, column = divmod(source, columns)
        if column + 1 < columns:
            sources.append(source)
            targets.append(source + 1)
        if row + 1 < rows:
            sources.append(source)
            targets.append(source + columns)
    return {
        'oracle': 'SciPy scipy.sparse.csgraph.maximum_flow, integral unit capacities',
        'rows': rows,
        'columns': columns,
        'nodeCount': nodeCount,
        'edgeCount': len(sources),
        **createScipyRecord(nodeCount, sources, targets),
    }


def createPrimalDualRecord():
    record = json.loads((FIXTURE_DIRECTORY / 'network-flow-primal-dual-oracle.inputs.json').read_text())
    for scene in record['scenes']:
        nodeCount = scene['nodeCount']
        # Aggregate only for maximum-flow value: NetworkX flow routines do not accept MultiDiGraph.
        aggregate = networkx.DiGraph()
        aggregate.add_nodes_from(range(nodeCount))
        multigraph = networkx.MultiDiGraph()
        multigraph.add_nodes_from(range(nodeCount), demand=0)
        for edge, (capacity, cost) in enumerate(zip(scene['capacities'], scene['costs'])):
            source, target = scene['edges'][2 * edge:2 * edge + 2]
            previous = aggregate.get_edge_data(source, target, {}).get('capacity', 0)
            aggregate.add_edge(source, target, capacity=previous + capacity)
            # Every original row has its own key, including parallel edges and self-loops.
            multigraph.add_edge(source, target, key=edge, capacity=capacity, weight=cost)
        maximumValue = networkx.maximum_flow_value(aggregate, 0, nodeCount - 1)
        required = scene['required']
        value = maximumValue if required == 0xffffffff else min(required, maximumValue)
        multigraph.nodes[0]['demand'] = -value
        multigraph.nodes[nodeCount - 1]['demand'] = value
        objective, flow = networkx.network_simplex(multigraph)
        # Independently check the external solver's edge bounds, conservation and keyed objective.
        balance = [0] * nodeCount
        checkedObjective = 0
        for source, target, key, attributes in multigraph.edges(keys=True, data=True):
            amount = flow[source][target][key]
            assert 0 <= amount <= attributes['capacity']
            balance[source] -= amount
            balance[target] += amount
            checkedObjective += amount * attributes['weight']
        assert balance == [-value] + [0] * (nodeCount - 2) + [value]
        assert checkedObjective == objective
        scene['expectedValue'] = int(value)
        scene['expectedCost'] = int(objective)
        scene['expectedUnserved'] = 0 if required == 0xffffffff else int(required - value)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', required=True, type=Path)
    arguments = parser.parse_args()
    outputDirectory = arguments.output_directory.resolve()
    if outputDirectory == FIXTURE_DIRECTORY:
        parser.error('write fresh scratch output, not the committed fixture directory')
    assert networkx.__version__ == '3.2.1', networkx.__version__
    assert scipy.__version__ == '1.13.1', scipy.__version__
    outputDirectory.mkdir(parents=True, exist_ok=True)
    print(f'Python {platform.python_version()}, NetworkX {networkx.__version__}, SciPy {scipy.__version__}, NumPy {numpy.__version__}')
    for name, record in [
        ('fan', createFanRecord()), ('grid', createGridRecord()), ('primal-dual', createPrimalDualRecord())
    ]:
        filename = f'network-flow-{name}-oracle.json'
        content = json.dumps(record, indent=2) + '\n'
        (outputDirectory / filename).write_text(content)
        committed = (FIXTURE_DIRECTORY / filename).read_text()
        assert json.loads(committed) == record, f'{filename}: record differs'
        assert committed == content, f'{filename}: bytes differ'
        print(f'{filename}: fresh output matches committed bytes')


if __name__ == '__main__':
    main()
