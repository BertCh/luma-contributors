# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Reproduce: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/spatial-weights/multipart-polygon-contiguity.oracle.py
from shapely.geometry import box, MultiPolygon

features = [MultiPolygon([box(0, 0, 1, 1), box(4, 0, 5, 1)]), box(1, 0, 2, 1), box(5.25, 0, 6.25, 1), MultiPolygon([])]
for row, first in enumerate(features):
    print(row, [(other, first.distance(second)) for other, second in enumerate(features) if other != row and not first.is_empty and not second.is_empty and first.distance(second) <= 0.25])
# 0 [(1, 0.0), (2, 0.25)]
# 1 [(0, 0.0)]
# 2 [(0, 0.25)]
# 3 []
