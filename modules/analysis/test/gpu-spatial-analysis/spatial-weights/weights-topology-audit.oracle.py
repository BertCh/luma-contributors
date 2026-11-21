# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-weights/weights-topology-audit.oracle.py
from shapely.geometry import box

polygons = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 1, 3, 2), box(10, 10, 11, 11)]
for criterion in ['rook', 'queen']:
    offsets, neighbors = [0], []
    for row, polygon in enumerate(polygons):
        for other, candidate in enumerate(polygons):
            if row == other:
                continue
            contact = polygon.boundary.intersection(candidate.boundary)
            if (contact.length > 0 if criterion == 'rook' else not contact.is_empty):
                neighbors.append(other)
        offsets.append(len(neighbors))
    print(criterion, 'offsets', offsets, 'neighbors', neighbors, 'weights', [1] * len(neighbors))
