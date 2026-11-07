# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct Rasterio/GDAL oracle, zero smoothing; run with rasterio and numpy installed."""
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.fill import fillnodata

cases = []
for name, width, height, distance in [('asymmetric-holes', 7, 6, 3), ('border-ties', 4, 5, 2), ('bounded-original-donors', 9, 1, 2), ('empty', 3, 3, 1)]:
    values = np.array([((index * 17 + 13) % 71) - 25 for index in range(width * height)], dtype='float32').reshape(height, width)
    mask = np.array([(index * 7 + index // width * 3) % 11 < 4 for index in range(width * height)], dtype='uint8').reshape(height, width)
    if name == 'bounded-original-donors':
        mask[:] = 0
        mask[0, 0] = 1
    if name == 'empty':
        mask[:] = 0
    values[mask == 0] = -9999
    expected = fillnodata(values.copy(), mask=mask, max_search_distance=distance, smoothing_iterations=0)
    cases.append(dict(name=name, width=width, height=height, maximumDistance=distance, values=values.ravel().tolist(), mask=mask.ravel().tolist(), expected=expected.ravel().tolist()))
Path(__file__).with_suffix('.json').write_text(json.dumps(dict(rasterio=rasterio.__version__, gdal=rasterio.__gdal_version__, smoothingIterations=0, cases=cases), indent=2) + '\n')
