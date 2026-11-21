# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Reproduce: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geometry-refinement/multipart-offset-components.oracle.py
from shapely import offset_curve, wkt

geometry = wkt.loads('MULTILINESTRING ((0 0,2 0),(5 0,7 0),(10 0,12 0))')
print(offset_curve(geometry, 1).wkt)
# MULTILINESTRING ((0 1, 2 1), (5 1, 7 1), (10 1, 12 1))
# Feature offsets [0,0,2,3] assign the three parts to rows [1,1,2].
# Every part has two vertices and one edge, whose global start rows are [0,2,4].
