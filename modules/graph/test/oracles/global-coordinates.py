# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Regenerate with s2sphere==0.2.5, geographiclib==2.1, pyproj==3.7.2, shapely==2.1.2; direct upstream calls, no GPU formulas."""
import json
import math
import struct
import tempfile
import pyproj
import shapely
from shapely.geometry import Polygon
from pathlib import Path
import geographiclib
import s2sphere
from geographiclib.geodesic import Geodesic

assert s2sphere.__version__ == '0.2.5'
assert geographiclib.__version__ == '2.1'
assert pyproj.__version__ == '3.7.2'
assert shapely.__version__ == '2.1.2'

def words(value):
    return [value & 0xffffffff, value >> 32]

rows = []
for level in [0, 1, 2, 15, 30]:
    if level < 3:
        identifiers = list(s2sphere.CellId.walk(level))
    else:
        identifiers = [s2sphere.CellId.from_lat_lng(s2sphere.LatLng.from_degrees(latitude, longitude)).parent(level) for latitude in [-90, -45, 0, 45, 90] for longitude in [-180, -90, 0, 90]]
    for identifier in identifiers:
        cell = s2sphere.Cell(identifier)
        rows.append({'cell': words(identifier.id()), 'level': level, 'neighbors': [words(neighbor.id()) for neighbor in identifier.get_edge_neighbors()], 'area': cell.exact_area(), 'perimeter': sum(cell.get_vertex(edge).angle(cell.get_vertex((edge + 1) % 4)) for edge in range(4))})

pairs = []
for origin, target in [([0, 0], [90, 0]), ([0, 89], [179, 89]), ([179, 10], [-179, 11]), ([0, 0], [179.7, 0.2])]:
    inverse = Geodesic.WGS84.Inverse(origin[1], origin[0], target[1], target[0])
    pairs.append({'origin': origin, 'target': target, 'distance': inverse['s12']})

# Gnomonic projection maps great circles exactly to straight lines. Topology comes from
# Shapely; spherical area comes independently from GeographicLib on a unit sphere.
first_polygon = Polygon([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
second_polygon = Polygon([(0, -0.7), (0.7, 0), (0, 0.7), (-0.7, 0)])
intersection = first_polygon.intersection(second_polygon)
def directions(polygon):
    return [[x / math.sqrt(x*x+y*y+1), y / math.sqrt(x*x+y*y+1), 1 / math.sqrt(x*x+y*y+1)] for x, y in list(polygon.exterior.coords)[:-1]]
def spherical_area(polygon):
    accumulator = Geodesic(1, 0).Polygon()
    for x, y, z in directions(polygon):
        accumulator.AddPoint(math.degrees(math.asin(z)), math.degrees(math.atan2(y, x)))
    return abs(accumulator.Compute()[2])
overlap = {'first': directions(first_polygon), 'second': directions(second_polygon), 'intersection': directions(intersection), 'area': spherical_area(intersection), 'sourceFraction': spherical_area(intersection) / spherical_area(first_polygon), 'targetFraction': spherical_area(intersection) / spherical_area(second_polygon)}

# GTX is a simple big-endian decoded test grid, generated locally and used directly by PROJ.
with tempfile.TemporaryDirectory() as temporary:
    grid_path = Path(temporary) / 'vertical.gtx'
    grid_path.write_bytes(struct.pack('>ddddii4f', 0, 0, 1, 1, 2, 2, -10, -20, -30, -40))
    transform = pyproj.Transformer.from_pipeline(f'+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=vgridshift +grids={grid_path} +multiplier=1 +step +proj=unitconvert +xy_in=rad +xy_out=deg')
    vertical = {'input': [0.5, 0.5, 100], 'output': list(transform.transform(0.5, 0.5, 100))}

# Direct S2 cap/cell intersection for the opt-in uniform-cover performance workload.
def cap_cells(radius):
    cap = s2sphere.Cap.from_axis_angle(s2sphere.Point(1, 0, 0), s2sphere.Angle.from_radians(radius))
    return [words(identifier.id()) for identifier in s2sphere.CellId.walk(4) if cap.may_intersect(s2sphere.Cell(identifier))]
covering = {'level': 4, 'initialRadius': 0.17, 'updatedRadius': 0.31, 'initialCells': cap_cells(0.17), 'updatedCells': cap_cells(0.31)}

output = Path(__file__).resolve().parents[1] / 'gpu-spatial-analysis/global-coordinates/fixtures/global-coordinates.json'
output.write_text(json.dumps({'provenance': {'s2': 's2sphere==0.2.5 direct CellId.get_edge_neighbors/Cell.exact_area', 'geodesics': 'geographiclib==2.1 direct Geodesic.WGS84.Inverse', 'overlap': 'shapely==2.1.2 gnomonic intersection; geographiclib==2.1 unit-sphere polygon area', 'vertical': 'pyproj==3.7.2 direct vgridshift on generated GTX, multiplier=1', 'projVersion': pyproj.proj_version_str}, 'cells': rows, 'geodesics': pairs, 'overlap': overlap, 'vertical': vertical, 'covering': covering}, indent=2) + '\n')
