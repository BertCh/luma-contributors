# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/geometry-topology/make-valid-output-oracles.oracle.py
"""Shapely 2.0.6 / GEOS linework oracle; printed values are pasted into the specs."""
import json
from shapely import make_valid
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize_full
from shapely.validation import explain_validity

scenes = {
    "hole": [Polygon([(0, 0), (4, 0), (4, 4), (0, 4)],
                     [[(1, 1), (1, 3), (3, 3), (3, 1)]])],
    "squares": [Polygon([(1, 2), (3, 2), (3, 4), (1, 4)]),
                Polygon([(6, 1), (9, 1), (9, 3), (6, 3)])],
    "bow-tie": [Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])],
    "dangle": [Polygon([(1, 2), (3, 2), (1, 2)])],
    "bridge": [Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0),
                        (-2, 0), (-4, 0), (-4, 2), (-2, 2), (-2, 0)])],
}

def polygons(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [polygon for part in getattr(geometry, "geoms", []) for polygon in polygons(part)]

for name, inputs in scenes.items():
    repaired = [polygon for geometry in inputs for polygon in polygons(make_valid(geometry))]
    print(json.dumps({"scene": name, "areas": sorted(polygon.area for polygon in repaired),
                      "rings": [1 + len(polygon.interiors) for polygon in repaired],
                      "coordinates": sorted([sorted(set(polygon.exterior.coords)) for polygon in repaired]),
                      "validity": [explain_validity(make_valid(geometry)) for geometry in inputs]}))

# Provenance is a host contract, not a Shapely output: eight original square edges have
# parameters [0,1], feature IDs 70/80, ring rows 0/1, edge rows 0..7. Each becomes two
# oppositely oriented directed sides. A bow tie splits edges 0 and 2 at t=1/2, producing six
# pieces. A two-vertex closed ring traverses its isolated segment twice: two source dangles.
# Its edge-start rows are [0,1], with both traversals parameterized from 0 to 1 on their own
# source edge. The two segments share a midpoint, which must not replace source ownership.
# Uncertainty fixture (predicate-contract derivation, not a GEOS diagnostic): implicitly
# close [(1e-30,1e-30),(2e-30,2e-30)] and [(1.5e-30,1.5e-30),(1e38,1e38)]. Each
# segment is traversed twice. All 2*2=4 cross-ring pairs span over 200 binary orders,
# beyond the exact predicate certificate; adjacent reversed same-ring edges are excluded.
# Thus uncertainCount=4 and candidateOverflow=1, with or without a counter destination.
# Deduplicate undirected linework before polygonize_full, whose diagnostics count geometric
# edges rather than the contributor's source rows (the bridge/dangle each occurs twice).
for name in ("dangle", "bridge"):
    coordinates = list(scenes[name][0].exterior.coords)
    edges = {tuple(sorted((first, second))) for first, second in zip(coordinates, coordinates[1:]) if first != second}
    faces, cuts, dangles, invalid = polygonize_full([LineString(edge) for edge in edges])
    print(json.dumps({"scene": name, "polygonize_full": {
        "faces": len(faces.geoms), "cuts": len(cuts.geoms),
        "dangles": len(dangles.geoms), "invalid": len(invalid.geoms)}}))
