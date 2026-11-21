# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/polygon-overlay/polygon-overlay-output-oracles.oracle.py
from shapely import intersection
from shapely.geometry import box

surface = intersection(box(0, 0, 4, 4), box(2, 2, 6, 6), grid_size=0.25)
print({"area": surface.area, "rings": 1 + len(surface.interiors),
       "vertices": sorted(surface.exterior.coords[:-1])})
