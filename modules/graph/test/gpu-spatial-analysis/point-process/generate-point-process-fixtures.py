# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent sklearn OPTICS oracle; run with sklearn 1.7.2 / numpy 2.2.6."""
import json
from pathlib import Path
import numpy
import sklearn
from sklearn.cluster import OPTICS

positions = [[0, 0], [0.1, 0], [0.2, 0], [3, 0], [3.1, 0], [3.2, 0], [8, 0]]
model = OPTICS(min_samples=2, max_eps=0.5, cluster_method='dbscan', eps=0.3).fit(positions)
def values(array):
    return [None if not numpy.isfinite(value) else float(value) for value in array]
fixture = {
    'reference': {'sklearn': sklearn.__version__, 'numpy': numpy.__version__},
    'positions': positions, 'minimumSamples': 2, 'maximumDistance': 0.5,
    'extractionDistance': 0.3,
    'ordering': model.ordering_.tolist(), 'predecessors': model.predecessor_.tolist(),
    'coreDistances': values(model.core_distances_), 'reachability': values(model.reachability_),
    'labels': model.labels_.tolist()
}
Path(__file__).with_name('point-process-fixtures.json').write_text(json.dumps(fixture, indent=2) + '\n')
