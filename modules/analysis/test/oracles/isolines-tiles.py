# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""R6.3 independent whole-domain skimage contours and Shapely vector-boundary overlays.

Run with /usr/bin/python3. No contributor/CPU twin is imported. The checkerboard's bilinear
saddles are exact determinant ties, so skimage's high-connected rule agrees with the inclusive
asymptotic rule. Other fields have no saddle ambiguity. Negation adapts equality-high.
skimage omits zero-length corner bridges; those are checked against hand-derived cases in TS.
All samples are float32 before the float64 oracle. Row zero advances toward larger world y.
"""
import json
from pathlib import Path
import numpy as np
import skimage
from skimage.measure import find_contours
import shapely
from shapely.geometry import LineString, Polygon

OUTPUT = Path(__file__).resolve().parents[1] / 'gpu-raster/isolines/fixtures/isolines-tiles.json'


def make_case(name, values, levels, mask=None):
    values = np.asarray(values, dtype=np.float32)
    height, width = values.shape
    if mask is None:
        mask = np.ones_like(values, dtype=bool)
    # Deliberately nonzero projected extent and common local origin, non-square samples.
    extent = [600000, 4600000, 600000 + width * 2, 4600000 + height * 3]
    segments = []
    polygon = Polygon([(3.25, 2.25), (width * 2 - 2.25, 4.25),
                       (width * 2 - 3.25, height * 3 - 2.25), (2.25, height * 3 - 3.25)])
    for level_index, level in enumerate(levels):
        contours = find_contours(-values.astype(float), -level, fully_connected='low', mask=mask)
        for contour in contours:
            for first, second in zip(contour[:-1], contour[1:]):
                first = [(float(first[1]) + .5) * 2, (float(first[0]) + .5) * 3]
                second = [(float(second[1]) + .5) * 2, (float(second[0]) + .5) * 3]
                segments.append([level_index, *first, *second])
    # Vector overlay is the line/polygon-boundary intersection, retaining each owning contour
    # segment and external boundary edge. Points on seam vertices may belong to two segments.
    overlay = []
    polygon_positions = list(polygon.exterior.coords)[:-1]
    for segment_index, segment in enumerate(segments):
        for boundary_index in range(len(polygon_positions)):
            intersection = LineString([segment[1:3], segment[3:5]]).intersection(
                LineString([polygon_positions[boundary_index],
                            polygon_positions[(boundary_index + 1) % len(polygon_positions)]]))
            if not intersection.is_empty:
                if intersection.geom_type != 'Point':
                    raise RuntimeError('fixture requires point intersections')
                overlay.append([segment[0], boundary_index, intersection.x, intersection.y])
    return dict(name=name, width=width, height=height, values=values.ravel().tolist(),
                mask=mask.astype(int).ravel().tolist(), levels=levels, extent=extent,
                origin=extent[:2], segments=segments, polygon=polygon_positions, overlay=overlay)


rows, columns = np.mgrid[:7, :8]
plane = columns + 2 * rows
hill = (columns - 3) ** 2 + (rows - 3) ** 2
mask = np.ones((7, 8), dtype=bool)
mask[3, 3] = False
checker_rows, checker_columns = np.mgrid[:6, :7]
checker = np.where((checker_rows + checker_columns) % 2 == 0, 1, -1)
cases = [make_case('plane', plane, [3.25, 7.25, 11.25]),
         make_case('masked-hill', hill, [2.5, 6.5, 12.5], mask),
         make_case('saddle-ties', checker, [0])]
OUTPUT.write_text(json.dumps(dict(versions=dict(numpy=np.__version__, skimage=skimage.__version__,
                                              shapely=shapely.__version__), cases=cases), indent=2) + '\n')
print(f'Wrote {OUTPUT}: {sum(len(case["segments"]) for case in cases)} segments')
