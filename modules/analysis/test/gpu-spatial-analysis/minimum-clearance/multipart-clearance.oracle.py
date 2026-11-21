# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Reproduce: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/minimum-clearance/multipart-clearance.oracle.py
from shapely import wkt, minimum_clearance

geometry = wkt.loads('MULTILINESTRING ((0 0,2 0),(0 0.25,2 0.25))')
print(minimum_clearance(geometry))  # 0.25
print([minimum_clearance(part) for part in geometry.geoms])  # [2.0, 2.0]
# All endpoint-to-opposite-part distances tie at .25. The contributor's documented lowest
# vertex-row rule chooses vertex 0 (0,0), whose unique closest point is (0,.25).
