# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/redcap-regions/redcap-audit-b.oracle.py
"""Kruskal plus a sorted-edge cut, with SciPy connected-component labels.

This is the class's documented constrained single-linkage rule, not REDCAP's
top-down SSD objective. A zero-cost graph checks exact edge-slot tie conventions.
The isolated ninth row can be NaN without changing any edge or region.
"""
import numpy as np
import sklearn
import warnings
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
         (4, 5), (4, 6), (5, 6), (5, 7), (6, 7)]
slots = [0, 1, 3, 4, 7, 10, 11, 13, 14, 17]
values = np.array([0, 2, 10, 14, 1, 3, 20, 26, 99], dtype=float)
for tied in [False, True]:
    table = values * 0 if tied else values
    parents = list(range(9))

    def root(row):
        while parents[row] != row:
            row = parents[row]
        return row

    ordered = sorted(zip(edges, slots), key=lambda item: ((table[item[0][0]] - table[item[0][1]]) ** 2, item[1]))
    tree = []
    for (first, second), slot in ordered:
        if root(first) != root(second):
            parents[max(root(first), root(second))] = min(root(first), root(second))
            tree.append((first, second))
    for target in ([1, 4, 5, 10] if not tied else [6, 8]):
        kept = tree[:max(0, len(tree) - max(0, min(target, 9) - 3))]
        rows = [first for first, _ in kept]
        columns = [second for _, second in kept]
        graph = csr_matrix((np.ones(len(kept)), (rows, columns)), shape=(9, 9))
        count, labels = connected_components(graph, directed=False)
        canonical = [int(np.flatnonzero(labels == label).min()) for label in labels]
        print('tied', tied, 'target', target, 'labels', canonical, 'count', count)

# Verify the reference claim on the documented six-row example plus two islands.
# Values are first rounded to float32, exactly as in the caller's feature table.
path_values = np.array([0, 1, 2.1, 3.3, 4.6, 6, 77, 88], dtype=np.float32).astype(float)
path_rows = np.arange(5)
path_columns = path_rows + 1
path = csr_matrix((np.ones(5), (path_rows, path_columns)), shape=(8, 8))
path = path + path.T
print('sklearn version', sklearn.__version__)
for connected in [False, True]:
    connectivity = path.copy().tolil()
    if connected:
        connectivity[5, 6] = connectivity[6, 5] = 1
        connectivity[6, 7] = connectivity[7, 6] = 1
    for target in [1, 4]:
        with warnings.catch_warnings(record=True) as emitted:
            result = AgglomerativeClustering(n_clusters=target, linkage='single',
                                             connectivity=connectivity.tocsr()).fit_predict(path_values[:, None])
        canonical = [int(np.flatnonzero(result == label).min()) for label in result]
        print('sklearn connected', connected, 'target', target, 'labels', canonical,
              'warnings', [str(item.message) for item in emitted])


def centered_ssd(table):
    return float(np.sum((table - table.mean()) ** 2))


# Exhaustive enumeration of the five possible one-edge cuts on the only nontrivial tree.
# The two islands contribute zero SSD. This verifies the SSD definition, not a Guo-paper claim.
whole = centered_ssd(path_values[:6])
for boundary in range(1, 6):
    within = centered_ssd(path_values[:boundary]) + centered_ssd(path_values[boundary:6])
    print('path boundary', boundary, 'canonical slot', 2 * (boundary - 1),
          'within SSD', within, 'gain', whole - within)

# Review R5: sklearn's single-linkage tree replaces zero distances with machine epsilon.
# With tiny positive distances, this changes the strict order even on connected inputs.
cycle_values = np.array([0, 0, 1e-18, 3e-18], dtype=np.float32).astype(float)
cycle_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
cycle_graph = csr_matrix((np.ones(8), ([0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3])), shape=(4, 4))
cycle_result = AgglomerativeClustering(n_clusters=3, linkage='single', connectivity=cycle_graph).fit_predict(cycle_values[:, None])
canonical = [int(np.flatnonzero(cycle_result == label).min()) for label in cycle_result]
print('R5 sklearn cycle raw', cycle_result.tolist(), 'canonical', canonical, 'squared edge costs', [(float((cycle_values[a]-cycle_values[b])**2)) for a, b in cycle_edges])
print('R5 strict zero-preserving single-linkage cut', [0, 0, 2, 3])
