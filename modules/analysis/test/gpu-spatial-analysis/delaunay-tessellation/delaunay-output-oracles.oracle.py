# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/delaunay-tessellation/delaunay-output-oracles.oracle.py
import numpy as np
from scipy.spatial import Delaunay
points = np.array([[0,0],[4,0],[3,2],[0,3]])
print(sorted(sorted(triangle) for triangle in Delaunay(points).simplices.tolist()))
