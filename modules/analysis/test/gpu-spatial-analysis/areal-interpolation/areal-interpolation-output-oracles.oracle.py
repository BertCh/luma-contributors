# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/areal-interpolation/areal-interpolation-output-oracles.oracle.py
from shapely.geometry import box, Point

sources = [box(0, 0, 2, 2), box(2, 0, 4, 2)]
target = box(1, 0, 3, 2)
areas = [source.intersection(target).area for source in sources]
print('areas:', areas)
print('extensive:', [area / source.area for area, source in zip(areas, sources)])
print('intensive:', [area / sum(areas) for area in areas])

# Explicit north-up grid: top-left (0,2), unit cells, four columns and two rows.
# Its centres lie at y=1.5,.5. The old default origin (0,0) instead covered y=-2..0.
centres = [Point(column + .5, 2 - row - .5) for row in range(2) for column in range(4)]
print('source raster:', [next((index for index, source in enumerate(sources) if source.contains(point)), 0xffffffff) for point in centres])
print('target raster:', [0 if target.contains(point) else 0xffffffff for point in centres])
# Raster crossingCapacity counts edge/row records across the entire batch.
# Each of two vertical edges crosses two rows; horizontal edges cross none.
print('source/target crossings:', [2 * 2 * len(sources), 2 * 2])
