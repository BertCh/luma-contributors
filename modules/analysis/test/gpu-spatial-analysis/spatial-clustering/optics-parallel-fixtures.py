# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent OPTICS/xi fixture above the old 4096-row GPU limit.

Run with /usr/bin/python3. All distances are integers, so float32 and float64 ties agree.
scikit-learn chooses the first unprocessed row of minimal reachability (input-index ties),
and changes predecessors only on strict improvement. This is also the sequential GPU rule.
Disconnected components exercise infinity ties; duplicates exercise zero ties. The shuffled
ordering, xi areas and labels cross both 64-rank extraction batches and 256-lane reductions.
"""
import json
from pathlib import Path

import numpy
import sklearn
from sklearn.cluster import OPTICS, cluster_optics_xi

random = numpy.random.default_rng(20261114)
coordinates = numpy.array([group * 32 + offset for group in range(684) for offset in [0, 0, 1, 2, 4, 8]], dtype=numpy.float32)
coordinates = coordinates[random.permutation(len(coordinates))]
positions = numpy.column_stack([coordinates, numpy.zeros(len(coordinates))])
fitted = OPTICS(min_samples=2, max_eps=10, algorithm='kd_tree').fit(positions)
extractions = []
for correction in [True, False]:
    labels, intervals = cluster_optics_xi(reachability=fitted.reachability_, predecessor=fitted.predecessor_, ordering=fitted.ordering_, min_samples=2, min_cluster_size=3, xi=0.2, predecessor_correction=correction)
    extractions.append({'predecessorCorrection': correction, 'labels': [int(value) if value >= 0 else 0xffffffff for value in labels], 'intervals': intervals.tolist()})
fixture = {
    'generator': Path(__file__).name,
    'versions': {'numpy': numpy.__version__, 'sklearn': sklearn.__version__},
    'positions': positions.flatten().tolist(),
    'options': {'minimumSamples': 2, 'maximumDistance': 10, 'extractionMethod': 'xi', 'xi': 0.2, 'minimumClusterSize': 3},
    'ordering': fitted.ordering_.tolist(),
    'coreDistances': fitted.core_distances_.tolist(),
    'reachability': [float(value) if numpy.isfinite(value) else 'Infinity' for value in fitted.reachability_],
    'predecessors': [int(value) if value >= 0 else 0xffffffff for value in fitted.predecessor_],
    'xi': extractions,
}
Path(__file__).with_suffix('.json').write_text(json.dumps(fixture, separators=(',', ':'), allow_nan=False) + '\n')
