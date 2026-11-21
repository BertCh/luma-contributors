# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""External references for the first STX-08 group: geographical detector q, REDCAP single linkage,
the correlogram over contiguity orders and the spatial entropy.

- Geographical detector: `geodetector` 0.2.0 (MIT) `q_statistic` and `q_significance_test`, loaded
  from its `_stats.py` (numpy and scipy only). F and lambda are formed with that package's own
  formulas (population variance in lambda). Strata all have at least two rows, because the package
  drops smaller strata.
- REDCAP single linkage: the constrained single-linkage definition, evaluated in binary64 with
  SciPy: `scipy.sparse.csgraph.minimum_spanning_tree` of the contiguity graph under squared
  Euclidean dissimilarity of the standardised attributes, the `k - c` heaviest tree edges removed,
  `connected_components` for the labels. No REDCAP package was run (spopt 0.8.0 has none).
- Correlogram over contiguity orders: binary64 NumPy. Rings by breadth-first search (the
  `libpysal` `higher_order` definition), row-standardised weights, Moran's I with the `esda.Moran`
  expectation and the normality and randomization variances. libpysal and esda were not installed.
- Spatial entropy: binary64 NumPy Shannon category entropy and Leibovici (2009)
  co-occurrence entropy. The mutual information computed here is
  H(origin) + H(destination) - H(ordered pair), in nats, for the category at each
  end of an ordered link. It is distinct from Altieri, Cocchi and Roli's (2018)
  decomposition over unordered category pairs and distance classes, which is
  not implemented here. The R `spatentropy` package (GPL) was not run.

Every value array is rounded to float32 first, the data the GPU kernels read.

Run: STX08_GEODETECTOR_SOURCE=<dir containing geodetector/_stats.py> <venv>/bin/python -I \
  modules/analysis/test/oracles/stx08-spatial-methods.py
"""
import importlib.util
import json
import os
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'fixtures' / 'stx08-spatial-methods.json'
GEODETECTOR_VERSION = '0.2.0'


def load_stats():
    path = Path(os.environ['STX08_GEODETECTOR_SOURCE']) / 'geodetector' / '_stats.py'
    spec = importlib.util.spec_from_file_location('geodetector_stats', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_float32(values):
    return np.asarray(values, dtype=np.float64).astype(np.float32).astype(np.float64)


def lattice(width, height):
    """Rook contiguity CSR of a width x height lattice, neighbors ascending."""
    offsets = [0]
    neighbors = []
    for row in range(width * height):
        x, y = row % width, row // width
        candidates = []
        if y > 0:
            candidates.append(row - width)
        if x > 0:
            candidates.append(row - 1)
        if x < width - 1:
            candidates.append(row + 1)
        if y < height - 1:
            candidates.append(row + width)
        neighbors.extend(candidates)
        offsets.append(len(neighbors))
    return offsets, neighbors


def adjacency(offsets, neighbors):
    return [neighbors[offsets[row]:offsets[row + 1]] for row in range(len(offsets) - 1)]


def geographical_detector_cases(stats, generator):
    cases = []
    for name, rows, strata, effect in [
        ('strong', 60, 4, 3.0),
        ('moderate', 200, 7, 1.0),
        ('weak', 150, 5, 0.2),
        ('offset', 90, 3, 1.5)
    ]:
        labels = np.arange(rows) % strata
        generator.shuffle(labels)
        means = generator.normal(0, effect, strata)
        offset = 50.0 if name == 'offset' else 0.0
        values = to_float32(offset + means[labels] + generator.normal(0, 1, rows))
        q = stats.q_statistic(values, labels)
        sst = float(np.sum((values - values.mean()) ** 2))
        variance = float(np.var(values, ddof=1))
        means_by_stratum = [values[labels == label].mean() for label in range(strata)]
        counts = [int(np.sum(labels == label)) for label in range(strata)]
        m1 = sum(mean * mean for mean in means_by_stratum)
        m2 = sum(np.sqrt(count) * mean for count, mean in zip(counts, means_by_stratum)) ** 2 / rows
        non_centrality = (m1 - m2) / (variance * (rows - 1) / rows)
        f_statistic = ((rows - strata) * q) / ((strata - 1) * (1 - q))
        cases.append({
            'name': name,
            'values': values.tolist(),
            'labels': labels.tolist(),
            'strataCount': strata,
            'q': q,
            'fStatistic': float(f_statistic),
            'nonCentrality': float(non_centrality),
            'pValue': stats.q_significance_test(values, labels),
            'totalSumOfSquares': sst
        })
    return cases


def standardize(table):
    table = np.asarray(table, dtype=np.float64)
    return (table - table.mean(axis=0)) / table.std(axis=0)


def canonical_labels(labels):
    """Relabel each component by its smallest row index (the label the GPU writes)."""
    smallest = {}
    for row, label in enumerate(labels):
        smallest.setdefault(int(label), row)
    return [smallest[int(label)] for label in labels]


def redcap_cases(generator):
    cases = []
    graphs = []
    offsets, neighbors = lattice(7, 7)
    graphs.append(('lattice-7x7-two-columns', offsets, neighbors, 2))
    offsets, neighbors = lattice(9, 5)
    graphs.append(('lattice-9x5-one-column', offsets, neighbors, 1))
    # Two disconnected 4x4 lattices: a forest of two trees.
    left_offsets, left_neighbors = lattice(4, 4)
    offsets = left_offsets[:-1] + [offset + len(left_neighbors) for offset in left_offsets]
    neighbors = left_neighbors + [neighbor + 16 for neighbor in left_neighbors]
    graphs.append(('two-lattices-4x4', offsets, neighbors, 2))
    for name, offsets, neighbors, column_count in graphs:
        rows = len(offsets) - 1
        values = to_float32(generator.normal(0, 1, (rows, column_count)))
        table = standardize(values)
        edges = []
        for row in range(rows):
            for neighbor in neighbors[offsets[row]:offsets[row + 1]]:
                if neighbor > row:
                    cost = float(np.sum((table[row] - table[neighbor]) ** 2))
                    edges.append((row, neighbor, cost))
        sparse = coo_matrix(
            ([cost for _, _, cost in edges], ([a for a, _, _ in edges], [b for _, b, _ in edges])),
            shape=(rows, rows)
        )
        tree = minimum_spanning_tree(sparse).tocoo()
        tree_edges = sorted(zip(tree.data, tree.row, tree.col), key=lambda entry: -entry[0])
        costs = sorted(entry[0] for entry in tree_edges)
        assert len(set(costs)) == len(costs), 'tree costs must be distinct'
        trees = rows - len(tree_edges)
        outcomes = []
        for region_count in [1, 2, 3, 5, 8, 15, rows + 3]:
            cut = min(max(region_count - trees, 0), len(tree_edges))
            kept = tree_edges[cut:]
            graph = coo_matrix(
                ([1] * len(kept), ([int(r) for _, r, _ in kept], [int(c) for _, _, c in kept])),
                shape=(rows, rows)
            )
            count, labels = connected_components(graph, directed=False)
            outcomes.append({
                'regionCount': region_count,
                'labels': canonical_labels(labels),
                'resultingRegionCount': int(count)
            })
        cases.append({
            'name': name,
            'offsets': offsets,
            'neighbors': neighbors,
            'columnCount': column_count,
            'values': values.reshape(-1).tolist(),
            'treeEdgeCount': len(tree_edges),
            'outcomes': outcomes
        })
    return cases


def rings(adjacency_lists, maximum_order):
    """Rows at exactly each order (breadth-first), per row."""
    result = [[] for _ in range(maximum_order)]
    for start in range(len(adjacency_lists)):
        distance = {start: 0}
        frontier = [start]
        for order in range(1, maximum_order + 1):
            next_frontier = []
            for node in frontier:
                for neighbor in adjacency_lists[node]:
                    if neighbor not in distance:
                        distance[neighbor] = order
                        next_frontier.append(neighbor)
            frontier = next_frontier
        for order in range(maximum_order):
            result[order].append(sorted(node for node, d in distance.items() if d == order + 1))
    return result


def moran(ring_lists, values):
    """Moran's I with row-standardised weights; islands are excluded (esda variances)."""
    active = [row for row, neighbors in enumerate(ring_lists) if neighbors]
    n = len(active)
    x = np.array([values[row] for row in active])
    z = x - x.mean()
    index = {row: position for position, row in enumerate(active)}
    w = np.zeros((n, n))
    for row in active:
        for neighbor in ring_lists[row]:
            if neighbor in index:
                w[index[row], index[neighbor]] = 1.0
        w[index[row]] /= w[index[row]].sum()
    s0 = w.sum()
    s1 = 0.5 * np.sum((w + w.T) ** 2)
    s2 = np.sum((w.sum(axis=1) + w.sum(axis=0)) ** 2)
    statistic = n / s0 * (z @ w @ z) / (z @ z)
    expected = -1.0 / (n - 1)
    variance_norm = (n * n * s1 - n * s2 + 3 * s0 * s0) / (s0 * s0 * (n * n - 1)) - expected**2
    kurtosis = (np.sum(z**4) / n) / ((np.sum(z**2) / n) ** 2)
    variance_rand = (
        n * ((n * n - 3 * n + 3) * s1 - n * s2 + 3 * s0 * s0)
        - kurtosis * ((n * n - n) * s1 - 2 * n * s2 + 6 * s0 * s0)
    ) / ((n - 1) * (n - 2) * (n - 3) * s0 * s0) - expected**2
    return {
        'statistic': float(statistic),
        'expected': float(expected),
        'varianceNormality': float(variance_norm),
        'varianceRandomization': float(variance_rand),
        'count': n,
        'islandCount': len(ring_lists) - n
    }


def correlogram_cases(generator):
    cases = []
    for width, height in [(8, 8), (10, 6)]:
        offsets, neighbors = lattice(width, height)
        rows = width * height
        x = np.arange(rows) % width
        y = np.arange(rows) // width
        values = to_float32(np.sin(x / 2.5) + np.cos(y / 3.0) + generator.normal(0, 0.4, rows))
        lists = adjacency(offsets, neighbors)
        maximum_order = 5
        by_order = rings(lists, maximum_order)
        cases.append({
            'name': f'lattice-{width}x{height}',
            'offsets': offsets,
            'neighbors': neighbors,
            'values': values.tolist(),
            'maximumOrder': maximum_order,
            'orders': [moran(by_order[order], values) for order in range(maximum_order)]
        })
    return cases


def entropy_cases(generator):
    cases = []
    for width, height, category_count in [(10, 10, 3), (12, 8, 5)]:
        offsets, neighbors = lattice(width, height)
        rows = width * height
        x = np.arange(rows) % width
        y = np.arange(rows) // width
        field = np.sin(x / 2.0) + np.cos(y / 2.5) + generator.normal(0, 0.5, rows)
        edges = np.quantile(field, np.linspace(0, 1, category_count + 1)[1:-1])
        categories = np.searchsorted(edges, field).astype(int)
        # One row without a category to exercise the missing value.
        missing_row = 7
        categories_with_missing = categories.copy()
        categories_with_missing[missing_row] = 0xFFFFFFFF
        valid = categories_with_missing < category_count
        counts = np.bincount(categories_with_missing[valid], minlength=category_count)
        pairs = np.zeros((category_count, category_count), dtype=np.int64)
        for row in range(rows):
            if not valid[row]:
                continue
            for neighbor in neighbors[offsets[row]:offsets[row + 1]]:
                if valid[neighbor]:
                    pairs[categories_with_missing[row], categories_with_missing[neighbor]] += 1

        def entropy(count_array):
            count_array = np.asarray(count_array, dtype=np.float64).reshape(-1)
            probability = count_array[count_array > 0] / count_array.sum()
            return float(-np.sum(probability * np.log(probability)))

        origin = entropy(pairs.sum(axis=1))
        destination = entropy(pairs.sum(axis=0))
        joint = entropy(pairs)
        cases.append({
            'name': f'lattice-{width}x{height}-{category_count}-categories',
            'offsets': offsets,
            'neighbors': neighbors,
            'categories': categories_with_missing.tolist(),
            'categoryCount': category_count,
            'pairCounts': pairs.reshape(-1).tolist(),
            'categoryRowCounts': counts.tolist(),
            'shannon': entropy(counts),
            'originEntropy': origin,
            'destinationEntropy': destination,
            'cooccurrence': joint,
            'mutualInformation': origin + destination - joint,
            'linkCount': int(pairs.sum()),
            'rowCount': int(valid.sum())
        })
    return cases


def main():
    stats = load_stats()
    generator = np.random.default_rng(20261109)
    fixture = {
        'description': 'STX-08 first group references: geographical detector, REDCAP single linkage, contiguity-order correlogram, spatial entropy.',
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'geodetector': GEODETECTOR_VERSION
        },
        'geographicalDetector': geographical_detector_cases(stats, generator),
        'redcap': redcap_cases(generator),
        'correlogram': correlogram_cases(generator),
        'entropy': entropy_cases(generator)
    }
    OUTPUT.write_text(json.dumps(fixture, indent=1) + '\n')
    print(f'wrote {OUTPUT}')


if __name__ == '__main__':
    main()
