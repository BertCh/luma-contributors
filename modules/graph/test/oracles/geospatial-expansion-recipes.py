# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Regenerate recipe references with NumPy and NetworkX; no luma implementation imported."""
import json
from pathlib import Path
import numpy as np
import networkx as nx

zones = np.array([0, 0, 0, 1, 1, 1, 2, 2])
values = np.array([1, 3, 3, 4, np.nan, 8, np.nan, np.inf])
probabilities = [0, 0.25, 0.5, 0.75, 1]
distributions = []
for zone in range(4):
    selected = values[(zones == zone) & np.isfinite(values)]
    unique, counts = np.unique(selected, return_counts=True)
    distributions.append(dict(count=len(selected), median=float(np.median(selected)) if len(selected) else None,
                              mode=float(unique[np.argmax(counts)]) if len(selected) else None,
                              unique=len(unique), quantiles=np.quantile(selected, probabilities).tolist() if len(selected) else [None]*5))
network = nx.DiGraph()
network.add_weighted_edges_from([(0, 1, 1), (1, 2, 2), (2, 3, 1)])
network.add_node(4)
scenario = network.copy()
scenario.remove_edge(1, 2)
def distances(graph):
    return [[nx.shortest_path_length(graph, origin, target, weight='weight') if nx.has_path(graph, origin, target) else None for target in range(5)] for origin in [0, 2]]
reference = dict(provenance=dict(numpy=np.__version__, networkx=nx.__version__),
                zones=zones.tolist(), values=[float(value) if np.isfinite(value) else None for value in values], probabilities=probabilities,
                distributions=distributions, baseline=distances(network), scenario=distances(scenario))
Path(__file__).with_name('fixtures').joinpath('geospatial-expansion-recipes.json').write_text(json.dumps(reference, indent=2)+'\n')
