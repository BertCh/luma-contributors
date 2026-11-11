# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""GEOS coverage validity with a gap width, for gaps between staggered segments.

Writes fixtures/coverage-validity-gaps.json next to this script. Run from anywhere with the
pinned interpreter of requirements.txt:
    <venv>/bin/python -I modules/analysis/test/oracles/coverage-validity-gaps.py

Oracle: shapely.coverage_invalid_edges and shapely.coverage_is_valid (GEOS CoverageValidator) at
several gap widths. Consumed by
test/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-topology.spec.ts.

Review finding polygon-topology-05 (math correctness review 2026-11-10): the gap rule of
GPUCoverageValidity asked for both ends of the shorter segment to lie within the gap width of the
longer one, so two boundaries that run side by side over only part of their length (staggered
segments, the usual shape of a gap between separately digitised polygons) were not reported. GEOS
reports them. The scenes here have that shape:

1. curated: the review's two rectangles, the same with an overlap along the gap at and below the
   gap width, a rotated pair on a 3-4-5 frame, a pair at projected-metre coordinates
   (500,000 / 4,500,000) and a row of three staggered neighbours;
2. disjoint pairs: random star-shaped polygons closer than 0.6, where every flag is a gap flag;
3. jittered triangulations: Delaunay triangles of a jittered grid, some of them moved or shrunk, so
   that gaps, overlaps and unmatched edges occur together; once near the origin and once at
   projected-metre coordinates.

Every coordinate is a float32 value (multiples of 1/256 near the origin, of 0.5 at the projected
coordinates). Segments are written with the lexicographically smaller end first.
"""

import json
import math
import os
import platform
import random

import numpy as np
import scipy
import shapely
from scipy.spatial import Delaunay
from shapely import Polygon, box

HERE = os.path.dirname(os.path.abspath(__file__))
DESTINATION = os.path.join(HERE, 'fixtures', 'coverage-validity-gaps.json')
QUANTUM = 1.0 / 256.0
DISJOINT_PAIR_COUNT = 30
DISJOINT_PAIR_SEED = 20261111
TRIANGULATION_COUNT = 10
TRIANGULATION_SEED = 20261112
PROJECTED_SEED = 20261113


def quantize(value):
    return round(value / QUANTUM) * QUANTUM


def get_ring(polygon):
    return [[float(x), float(y)] for x, y in list(polygon.exterior.coords)[:-1]]


def normalize_segment(first, second):
    low, high = sorted([(float(first[0]), float(first[1])), (float(second[0]), float(second[1]))])
    return [low[0], low[1], high[0], high[1]]


def get_case(polygons, gap_width):
    geometries = np.array(polygons, dtype=object)
    edges = shapely.coverage_invalid_edges(geometries, gap_width=gap_width)
    invalid = []
    for polygon_index, edge in enumerate(edges):
        segments = set()
        for part in shapely.get_parts(edge):
            coordinates = list(part.coords)
            for index in range(len(coordinates) - 1):
                segments.add(tuple(normalize_segment(coordinates[index], coordinates[index + 1])))
        invalid.extend([polygon_index, *segment] for segment in sorted(segments))
    return {
        'gapWidth': gap_width,
        'isValid': bool(shapely.coverage_is_valid(geometries, gap_width=gap_width)),
        'invalid': invalid,
    }


def get_scene(name, family, polygons, gap_widths):
    for polygon in polygons:
        assert polygon.is_valid and not polygon.interiors, name
        for x, y in polygon.exterior.coords:
            assert np.float32(x) == x and np.float32(y) == y, (name, x, y)
    return {
        'name': name,
        'family': family,
        # One shell per polygon, without the closing vertex.
        'polygons': [[get_ring(polygon)] for polygon in polygons],
        'cases': [get_case(polygons, gap_width) for gap_width in gap_widths],
    }


def get_frame_box(origin, along, across, along_range, across_range):
    """A rectangle in the frame (along, across) at origin; exact for dyadic ranges."""
    corners = [
        (along_range[0], across_range[0]),
        (along_range[1], across_range[0]),
        (along_range[1], across_range[1]),
        (along_range[0], across_range[1]),
    ]
    return Polygon(
        [
            (
                origin[0] + s * along[0] + t * across[0],
                origin[1] + s * along[1] + t * across[1],
            )
            for s, t in corners
        ]
    )


def get_curated_scenes():
    scenes = []
    lower = box(0, -5, 10, 0)
    # The review's pair: side by side over x in [5, 10] only, 0.0625 apart.
    scenes.append(
        get_scene('staggered', 'curated', [lower, box(5, 0.0625, 15, 5)], [0.0, 0.03125, 0.0625, 0.125])
    )
    # Side by side over 0.125 only: not longer than a gap width of 0.125, longer than one of 0.0625.
    scenes.append(
        get_scene(
            'staggered-overlap-at-limit',
            'curated',
            [lower, box(9.875, 0.0625, 15, 5)],
            [0.0625, 0.125, 0.25],
        )
    )
    scenes.append(
        get_scene(
            'staggered-short-overlap',
            'curated',
            [lower, box(9.9375, 0.03125, 15, 5)],
            [0.03125, 0.0625, 0.125],
        )
    )
    # The upper box starts past the end of the lower edge: nothing projects.
    scenes.append(
        get_scene('staggered-past-the-end', 'curated', [lower, box(10.0625, 0.0625, 15, 5)], [0.125, 0.5])
    )
    # A 3-4-5 frame keeps the rotated coordinates dyadic. The gap is 5 / 64 = 0.078125.
    along = (4.0, 3.0)
    across = (-3.0, 4.0)
    scenes.append(
        get_scene(
            'staggered-rotated',
            'curated',
            [
                get_frame_box((1.0, 2.0), along, across, (0.0, 2.0), (-1.0, 0.0)),
                get_frame_box((1.0, 2.0), along, across, (1.0, 3.0), (1.0 / 64.0, 1.0)),
            ],
            [0.0625, 0.078125, 0.125],
        )
    )
    # Projected metres: the float32 step is 0.5 at the northing. The gap is 1 m.
    east = 500000.0
    north = 4500000.0
    scenes.append(
        get_scene(
            'staggered-projected',
            'curated',
            [box(east, north - 50, east + 100, north), box(east + 40, north + 1, east + 160, north + 50)],
            [0.5, 1.0, 2.0],
        )
    )
    # Three neighbours above one long edge, each staggered against the next.
    scenes.append(
        get_scene(
            'staggered-row',
            'curated',
            [
                box(0, -5, 12, 0),
                box(-2, 0.0625, 3, 4),
                box(3, 0.125, 8, 4),
                box(8, 0.0625, 14, 4),
            ],
            [0.0, 0.0625, 0.125, 0.25],
        )
    )
    return scenes


def get_star_polygon(generator, center, radius, vertex_count):
    angles = sorted(generator.uniform(0, 2 * math.pi) for _ in range(vertex_count))
    return Polygon(
        [
            (
                quantize(center[0] + radius * generator.uniform(0.6, 1) * math.cos(angle)),
                quantize(center[1] + radius * generator.uniform(0.6, 1) * math.sin(angle)),
            )
            for angle in angles
        ]
    )


def get_disjoint_pair_scenes(count, seed):
    generator = random.Random(seed)
    gap_widths = [0.125, 0.25, 0.5, 0.75]
    scenes = []
    while len(scenes) < count:
        first = get_star_polygon(generator, (0, 0), 4, generator.randint(3, 6))
        second = get_star_polygon(
            generator, (generator.uniform(3, 8), generator.uniform(-3, 3)), 4, generator.randint(3, 6)
        )
        if not (first.is_valid and second.is_valid) or first.area < 0.5 or second.area < 0.5:
            continue
        separation = first.distance(second)
        if separation == 0 or separation > 0.6:
            continue
        scene = get_scene(f'disjoint-pair-{len(scenes)}', 'disjoint-pair', [first, second], gap_widths)
        # Keep pairs where the gap rule decides something.
        if all(case['isValid'] for case in scene['cases']):
            continue
        scenes.append(scene)
    return scenes


def get_jittered_triangulation_scenes(count, family, seed, origin, scale, quantum, gap_widths):
    """Delaunay triangles of a jittered 5 x 5 grid; about a third are moved or shrunk.

    Coordinates are origin + scale * (unit layout), rounded to multiples of quantum.
    """
    generator = random.Random(seed)

    def place(value, axis):
        return origin[axis] + round(scale * value / quantum) * quantum

    scenes = []
    while len(scenes) < count:
        points = np.array(
            [
                [
                    place(2 * column + generator.uniform(-0.6, 0.6), 0),
                    place(2 * row + generator.uniform(-0.6, 0.6), 1),
                ]
                for row in range(5)
                for column in range(5)
            ]
        )
        polygons = []
        for simplex in Delaunay(points).simplices:
            triangle = points[simplex].copy()
            choice = generator.random()
            if choice < 0.15:
                # Moved as a whole: a gap on one side, an overlap on the other.
                triangle += [
                    round(scale * generator.uniform(-0.2, 0.2) / quantum) * quantum,
                    round(scale * generator.uniform(-0.2, 0.2) / quantum) * quantum,
                ]
            elif choice < 0.3:
                # Shrunk towards its centroid: gaps all around.
                centroid = triangle.mean(axis=0)
                factor = generator.uniform(0.02, 0.1)
                triangle = np.round((triangle + (centroid - triangle) * factor) / quantum) * quantum
            polygon = Polygon(triangle)
            if polygon.area > 0.01 * scale * scale:
                polygons.append(polygon)
        scenes.append(get_scene(f'{family}-{len(scenes)}', family, polygons, gap_widths))
    return scenes


def main():
    scenes = (
        get_curated_scenes()
        + get_disjoint_pair_scenes(DISJOINT_PAIR_COUNT, DISJOINT_PAIR_SEED)
        + get_jittered_triangulation_scenes(
            TRIANGULATION_COUNT,
            'jittered-triangulation',
            TRIANGULATION_SEED,
            (0.0, 0.0),
            1.0,
            QUANTUM,
            [0.0, 0.125, 0.25, 0.75],
        )
        # The same construction in projected metres: 64 m triangles on the 0.5 m float32 grid of
        # the northing, gap widths in metres.
        + get_jittered_triangulation_scenes(
            TRIANGULATION_COUNT,
            'projected-triangulation',
            PROJECTED_SEED,
            (500000.0, 4500000.0),
            32.0,
            0.5,
            [0.0, 4.0, 8.0, 24.0],
        )
    )
    fixture = {
        'generator': 'coverage-validity-gaps.py',
        'versions': {
            'python': platform.python_version(),
            'shapely': shapely.__version__,
            'geos': shapely.geos_version_string,
            'numpy': np.__version__,
            'scipy': scipy.__version__,
        },
        'description': (
            'shapely.coverage_invalid_edges and coverage_is_valid per scene and gap width; '
            'invalid rows are [polygonIndex, x0, y0, x1, y1] with the smaller end first'
        ),
        'scenes': scenes,
    }
    with open(DESTINATION, 'w') as file:
        json.dump(fixture, file, separators=(',', ':'))
        file.write('\n')
    case_count = sum(len(scene['cases']) for scene in scenes)
    invalid_count = sum(len(case['invalid']) for scene in scenes for case in scene['cases'])
    print(f'{len(scenes)} scenes, {case_count} cases, {invalid_count} invalid segments -> {DESTINATION}')


if __name__ == '__main__':
    main()
