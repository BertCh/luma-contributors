# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Reproduce: /usr/bin/python3 modules/analysis/test/gpu-raster/remote-sensing/remote-output-oracles.oracle.py
from shapely.geometry import box

destination = box(1.25, 0, 2.5, 1)
intersections = [(row, destination.intersection(box(row, 0, row+1, 1)).area) for row in range(4)]
print('source rows and areas:', [(row, area) for row, area in intersections if area > 0])
for values in [[100, 2, 4, 100], [100, 10, 14, 100]]:
    print('destination-area mean:', sum(values[row]*area for row, area in intersections)/destination.area)
