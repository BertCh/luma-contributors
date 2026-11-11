# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Valid polygons whose rings touch at single points, for GPUPolygonTriangulation.

Writes fixtures/polygon-triangulation-touching-rings.json next to this script. Run from anywhere
with the pinned interpreter of requirements.txt:
    <venv>/bin/python -I modules/analysis/test/oracles/polygon-triangulation-touching-rings.py

Consumed by
test/gpu-spatial-analysis/polygon-triangulation/gpu-polygon-triangulation-touching-rings.spec.ts.

Review fixes of the closing pass of the math correctness review (2026-11-11,
dev-docs/research/math-correctness-review-2026-11-10/fix-review-closing-geometry.md, findings
frc-geometry-01 to 04). A hole may touch the shell or another hole at single points and the
polygon stays valid (OGC, Shapely `is_valid`) as long as its interior stays connected. The touch
can be vertex to vertex, or a vertex of one ring inside an edge of another.

Oracle: Shapely. Every polygon is built with `shapely.Polygon(shell, holes)` and kept only when
`is_valid`; `area` is `Polygon.area` (exact here: integer coordinates below 2^23, so every
shoelace product and sum is an integer below 2^53). The spec requires a `valid = 1` triangulation
to tile the polygon, which it checks exactly, and the triangle areas to sum to `area`.

Two lists:

- `review`: the inputs named in the review, each a polygon the kernel as it was before these
  fixes flagged or triangulated wrongly on one of its two paths.
- `families`: seeded random polygons on an integer lattice (seed 101): star-shaped or
  rectangle-union shells, small holes translated so that one vertex lands on a vertex or inside
  an edge of the shell or of an earlier hole (or apart from every ring), then written as they are
  and as variants: collinear vertices added, vertices repeated, either winding, and lattice shears
  that make slivers with coordinates up to six million. Of the 7,938 polygons generated, the
  fixture keeps every eighth and the 59 that the kernel as it was before these fixes flagged or
  got wrong on the test GPU (`FAILED_BEFORE`: 49 flagged and 5 wrong on the Earcut path, 5 wrong
  on the monotone path). The indices in `FAILED_BEFORE` belong to this generator's output with
  the library versions the fixture records.
"""

import json
import math
import os
import platform
import random
from math import gcd

import shapely
from shapely import LinearRing, Polygon, box, unary_union
from shapely.affinity import translate

HERE = os.path.dirname(os.path.abspath(__file__))
DESTINATION = os.path.join(HERE, 'fixtures', 'polygon-triangulation-touching-rings.json')

SEED = 101
TARGET_PER_FAMILY = 400
MAXIMUM_HOLES = 6
SAMPLE_STEP = 8

# Indices, in generation order, of the polygons that the kernel flagged or triangulated wrongly
# before the review fixes (GPU run of 2026-11-11, headless Chromium on Apple Metal).
FAILED_BEFORE = [
    298, 720, 721, 927, 928, 1493, 1695, 1696, 1895, 1912, 2034, 2075, 2076, 2271, 2272, 2559,
    2767, 2796, 2973, 3076, 3077, 3130, 3327, 3428, 3455, 3592, 3593, 3639, 3665, 3697, 3698,
    3750, 3758, 3827, 3828, 4141, 4145, 4198, 4278, 4279, 4303, 4304, 4368, 4621, 4622, 4623,
    4779, 4865, 4866, 4944, 4945, 4946, 5355, 5356, 6500, 6942, 7111, 7166, 7733,
]  # fmt: skip

# Inputs named in fix-review-closing-geometry.md. Shell first, then holes.
REVIEW_INPUTS = [
    (
        'frc-geometry-01 hole vertex inside an edge of another hole',
        [[(7, 2), (8, 0), (8, 7), (3, 12), (0, 11)], [(5, 9), (3, 9), (5, 10)], [(7, 4), (4, 9), (7, 5)]],
    ),
    (
        'frc-geometry-01 hole vertex inside a shell edge',
        [
            [(0, 7), (1, 2), (7, 7), (11, 1), (22, 14), (12, 13), (24, 17), (22, 22), (15, 18), (11, 17), (3, 22), (5, 15)],
            [(15, 16), (19, 17), (15, 14)],
        ],
    ),
    (
        'frc-geometry-01 mirrored in x',
        [[(-7, 2), (-8, 0), (-8, 7), (-3, 12), (0, 11)], [(-5, 9), (-3, 9), (-5, 10)], [(-7, 4), (-4, 9), (-7, 5)]],
    ),
    (
        'frc-geometry-01 mirrored in y',
        [[(7, -2), (8, 0), (8, -7), (3, -12), (0, -11)], [(5, -9), (3, -9), (5, -10)], [(7, -4), (4, -9), (7, -5)]],
    ),
    (
        'frc-geometry-02 monotone path, vertex-to-vertex touches',
        [
            [(0, 6), (0, 3), (4, 3), (4, 5), (6, 5), (6, 10), (2, 10), (2, 6)],
            [(4, 6), (5, 6), (4, 5), (3, 6)],
            [(4, 6), (4, 8), (5, 8)],
        ],
    ),
    (
        'frc-geometry-02 default configuration, base of the 4,106-vertex polygon',
        [
            [(10, 12), (22, 12), (22, 23), (20, 23), (20, 24), (17, 24), (17, 23), (10, 23)],
            [(16, 22), (18, 20), (18, 18), (13, 16)],
            [(18, 19), (22, 22), (21, 19)],
            [(18, 17), (20, 19), (18, 15)],
        ],
    ),
    (
        'frc-geometry-03 tie between collinear bridge candidates',
        [
            [(3, 5), (2, 2), (5, 3), (8, 2), (9, 1), (4, 10), (4, 9), (2, 8), (2, 5), (5, 6)],
            [(4, 5), (6, 6), (6, 5), (5, 5)],
            [(5, 5), (6, 4), (6, 3)],
        ],
    ),
    (
        'frc-geometry-03 repeated leftmost vertex shared by two holes',
        [[(8, 7), (7, 7), (10, 12), (3, 9), (5, 3)], [(6, 7), (4, 6), (4, 7), (5, 8)], [(4, 6), (4, 6), (5, 6), (5, 5)]],
    ),
    (
        'frc-geometry-04 rings apart, hole on the row of a shell vertex',
        [
            [(21, 13), (15, 15), (3, 0), (12, 8), (14, 6), (19, 0), (22, 2), (21, 11)],
            [(20, 6), (19, 6), (21, 9)],
            [(18, 2), (19, 4), (19, 2)],
        ],
    ),
    (
        'frc-geometry-04 with a repeated vertex and a third hole',
        [
            [(21, 13), (15, 15), (3, 0), (12, 8), (14, 6), (19, 0), (22, 2), (21, 11)],
            [(20, 6), (19, 6), (21, 9)],
            [(18, 2), (19, 4), (19, 4), (19, 2)],
            [(13, 8), (13, 10), (15, 10), (15, 8)],
        ],
    ),
    ('square, hole on the floor edge', [[(0, 0), (8, 0), (8, 8), (0, 8)], [(4, 0), (5, 3), (3, 3)]]),
    ('square, hole under the ceiling edge', [[(0, 0), (8, 0), (8, 8), (0, 8)], [(4, 8), (3, 5), (5, 5)]]),
    ('square, hole on the left wall', [[(0, 0), (8, 0), (8, 8), (0, 8)], [(0, 4), (3, 3), (3, 5)]]),
    ('square, hole on the right wall', [[(0, 0), (8, 0), (8, 8), (0, 8)], [(8, 4), (5, 5), (5, 3)]]),
    (
        'square, hole under the floor edge of a hole, leftmost vertex touches',
        [[(0, 0), (12, 0), (12, 12), (0, 12)], [(3, 6), (9, 6), (6, 9)], [(5, 6), (7, 3), (8, 4)]],
    ),
    (
        'square, hole on the top edge of a hole, leftmost vertex touches',
        [[(0, 0), (12, 0), (12, 12), (0, 12)], [(3, 6), (6, 3), (9, 6)], [(5, 6), (8, 8), (7, 9)]],
    ),
]


def get_ring_points(ring):
    return [(int(x), int(y)) for x, y in list(ring.coords)[:-1]]


def get_star_shell(generator, grid, vertex_count):
    center_x = grid / 2 + generator.uniform(-1, 1)
    center_y = grid / 2 + generator.uniform(-1, 1)
    points = set()
    while len(points) < vertex_count:
        points.add((generator.randint(0, grid), generator.randint(0, grid)))
    points = sorted(points, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
    return Polygon(points)


def get_rectangle_union_shell(generator, grid, rectangle_count):
    rectangles = []
    for _ in range(rectangle_count):
        low_x = generator.randint(0, grid - 2)
        low_y = generator.randint(0, grid - 2)
        rectangles.append(
            box(
                low_x,
                low_y,
                generator.randint(low_x + 1, min(grid, low_x + grid // 2)),
                generator.randint(low_y + 1, min(grid, low_y + grid // 2)),
            )
        )
    union = unary_union(rectangles)
    parts = list(union.geoms) if union.geom_type == 'MultiPolygon' else [union]
    return max(parts, key=lambda part: part.area)


def get_small_shape(generator, grid):
    kind = generator.choice(['tri', 'tri', 'rect', 'quad', 'thin'])
    x = generator.randint(1, grid - 1)
    y = generator.randint(1, grid - 1)
    radius = generator.randint(1, max(1, grid // 5))
    if kind == 'rect':
        return box(x, y, x + generator.randint(1, radius), y + generator.randint(1, radius))
    if kind == 'thin':
        delta_x = generator.randint(-grid // 2, grid // 2)
        delta_y = generator.randint(-grid // 2, grid // 2)
        return Polygon(
            [
                (x, y),
                (x + delta_x, y + delta_y),
                (x + delta_x + generator.choice([0, 1]), y + delta_y + generator.choice([1, 0, -1])),
            ]
        )
    count = 3 if kind == 'tri' else 4
    points = [(x + generator.randint(-radius, radius), y + generator.randint(-radius, radius)) for _ in range(count)]
    center_x = sum(point[0] for point in points) / count
    center_y = sum(point[1] for point in points) / count
    points.sort(key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
    return Polygon(points)


def get_lattice_points_inside_edges(ring_points):
    """Lattice points strictly inside the edges of a ring."""
    points = []
    count = len(ring_points)
    for index in range(count):
        start = ring_points[index]
        end = ring_points[(index + 1) % count]
        steps = gcd(abs(end[0] - start[0]), abs(end[1] - start[1]))
        for step in range(1, steps):
            points.append(
                (start[0] + (end[0] - start[0]) // steps * step, start[1] + (end[1] - start[1]) // steps * step)
            )
    return points


def touches_only_at_points(first, second):
    intersection = first.intersection(second)
    return intersection.is_empty or intersection.geom_type in ('Point', 'MultiPoint')


def build_polygon(generator, grid, shell_kind, hole_count, touch_modes):
    """A shell and holes placed by `touch_modes`, or None; returns the polygon and the modes used."""
    shell = (
        get_star_shell(generator, grid, generator.randint(4, 14))
        if shell_kind == 'star'
        else get_rectangle_union_shell(generator, grid, generator.randint(2, 6))
    )
    if not shell.is_valid or shell.area < grid * grid / 6 or shell.geom_type != 'Polygon':
        return None
    natural_holes = [Polygon(ring) for ring in shell.interiors]
    shell = Polygon(shell.exterior)
    shell_points = get_ring_points(shell.exterior)
    holes = list(natural_holes)
    tags = set(['natural-hole'] if natural_holes else [])
    attempts = 0
    while len(holes) < hole_count + len(natural_holes) and attempts < 60:
        attempts += 1
        shape = get_small_shape(generator, grid)
        if not shape.is_valid or shape.area == 0:
            continue
        mode = generator.choice(touch_modes)
        points = get_ring_points(shape.exterior)
        own_vertex = generator.choice(points)
        target = None
        if mode == 'shell-vertex':
            target = generator.choice(shell_points)
        elif mode == 'shell-edge':
            candidates = get_lattice_points_inside_edges(shell_points)
            target = generator.choice(candidates) if candidates else None
        elif mode == 'hole-vertex' and holes:
            target = generator.choice(get_ring_points(generator.choice(holes).exterior))
        elif mode == 'hole-edge' and holes:
            candidates = get_lattice_points_inside_edges(get_ring_points(generator.choice(holes).exterior))
            target = generator.choice(candidates) if candidates else None
        elif mode == 'row-of-shell-vertex':
            # The leftmost vertex of the hole goes on the row of a shell vertex.
            shell_vertex = generator.choice(shell_points)
            leftmost = min(points)
            target = None
            shape = translate(shape, 0, shell_vertex[1] - leftmost[1])
            own_vertex = None
        if mode not in ('none', 'row-of-shell-vertex'):
            if target is None:
                continue
            shape = translate(shape, target[0] - own_vertex[0], target[1] - own_vertex[1])
        if not shell.covers(shape):
            continue
        if not touches_only_at_points(shape, shell.exterior):
            continue
        if any(not touches_only_at_points(shape, hole) for hole in holes):
            continue
        holes.append(shape)
        if mode != 'none':
            tags.add(mode)
    if len(holes) == 0:
        return None
    polygon = Polygon(shell_points, [get_ring_points(hole.exterior) for hole in holes])
    if not polygon.is_valid:
        return None
    return polygon, tags


def add_collinear_vertices(generator, points):
    result = []
    count = len(points)
    for index in range(count):
        start = points[index]
        end = points[(index + 1) % count]
        result.append(start)
        steps = gcd(abs(end[0] - start[0]), abs(end[1] - start[1]))
        for step in range(1, steps):
            if generator.random() < 0.5:
                result.append(
                    (start[0] + (end[0] - start[0]) // steps * step, start[1] + (end[1] - start[1]) // steps * step)
                )
    return result


def add_repeated_vertices(generator, points):
    result = []
    for point in points:
        result.append(point)
        if generator.random() < 0.25:
            result.append(point)
    return result


def orient_ring(points, counter_clockwise):
    count = len(points)
    area = sum(
        points[index][0] * points[(index + 1) % count][1] - points[(index + 1) % count][0] * points[index][1]
        for index in range(count)
    )
    return points if (area > 0) == counter_clockwise else points[::-1]


def rotate_ring(generator, points):
    shift = generator.randrange(len(points))
    return points[shift:] + points[:shift]


# Lattice shears: they keep every incidence and turn the polygons into slivers.
SHEARS = {
    'shear-64': lambda point: (point[0] + 64 * point[1], point[1]),
    'shear-4096': lambda point: (point[0] + 4096 * point[1], point[1]),
    'diag-4096': lambda point: (4096 * point[0] + point[1], 4095 * point[0] + point[1]),
    'diag-2^18': lambda point: (262144 * point[0] + point[1], 262143 * point[0] + point[1]),
}

FAMILIES = [
    ('touch-shell-vertex', ['shell-vertex', 'none']),
    ('touch-shell-edge', ['shell-edge', 'none']),
    ('touch-hole-vertex', ['hole-vertex', 'none', 'none']),
    ('touch-hole-edge', ['hole-edge', 'none', 'none']),
    ('touch-mixed', ['shell-vertex', 'shell-edge', 'hole-vertex', 'hole-edge', 'none']),
    ('ray-through-shell-vertex', ['row-of-shell-vertex']),
    ('plain-small-grid', ['none']),
]


def get_entry(name, rings):
    """A fixture entry; asserts that Shapely finds the polygon valid."""
    polygon = Polygon(rings[0], rings[1:])
    assert polygon.is_valid, (name, shapely.is_valid_reason(polygon))
    for ring in rings:
        for x, y in ring:
            assert float(x) == x and float(y) == y and abs(x) < 2**23 and abs(y) < 2**23, (name, x, y)
    linear_rings = [LinearRing(ring) for ring in rings]
    touching = any(
        linear_rings[first].intersects(linear_rings[second])
        for first in range(len(linear_rings))
        for second in range(first + 1, len(linear_rings))
    )
    return {
        'name': name,
        # Rings without the closing vertex, shell first; any winding.
        'rings': [[[int(x), int(y)] for x, y in ring] for ring in rings],
        'area': polygon.area,
        'touching': touching,
    }


def get_family_entries():
    generator = random.Random(SEED)
    entries = []

    def emit(family, polygon, transform=None, collinear=False, repeated=False, shell_counter_clockwise=True, holes_counter_clockwise=False):
        rings = [get_ring_points(polygon.exterior)] + [get_ring_points(ring) for ring in polygon.interiors]
        if collinear:
            rings = [add_collinear_vertices(generator, ring) for ring in rings]
        if transform:
            rings = [[transform(point) for point in ring] for ring in rings]
        if repeated:
            rings = [add_repeated_vertices(generator, ring) for ring in rings]
        rings = [rotate_ring(generator, orient_ring(rings[0], shell_counter_clockwise))] + [
            rotate_ring(generator, orient_ring(ring, holes_counter_clockwise)) for ring in rings[1:]
        ]
        entries.append(get_entry(f'{family} #{len(entries)}', rings))

    for family, modes in FAMILIES:
        made = 0
        while made < TARGET_PER_FAMILY:
            grid = generator.choice([8, 12, 16, 24])
            built = build_polygon(
                generator, grid, generator.choice(['star', 'rect']), generator.randint(1, MAXIMUM_HOLES), modes
            )
            if built is None:
                continue
            polygon, tags = built
            wanted = [mode for mode in modes if mode != 'none']
            if wanted and not (tags & set(wanted)):
                continue
            made += 1
            emit(family, polygon)
            variant = made % 6
            if variant == 0:
                emit(family + '+collinear', polygon, collinear=True)
            if variant == 1:
                emit(family + '+repeated', polygon, repeated=True)
            if variant == 2:
                emit(family + '+clockwise-shell', polygon, shell_counter_clockwise=False)
            if variant == 3:
                emit(family + '+counter-clockwise-holes', polygon, holes_counter_clockwise=True)
            if variant == 4:
                emit(
                    family + '+clockwise-shell-collinear',
                    polygon,
                    collinear=True,
                    shell_counter_clockwise=False,
                    holes_counter_clockwise=True,
                )
            shear = generator.choice(list(SHEARS))
            emit(f'{family}+{shear}', polygon, transform=SHEARS[shear], collinear=(variant == 5))
    return entries


def main():
    generated = get_family_entries()
    failed_before = set(FAILED_BEFORE)
    families = []
    for index, entry in enumerate(generated):
        if index % SAMPLE_STEP == 0 or index in failed_before:
            families.append({**entry, 'failedBefore': index in failed_before})
    fixture = {
        'generator': 'polygon-triangulation-touching-rings.py',
        'versions': {
            'python': platform.python_version(),
            'shapely': shapely.__version__,
            'geos': shapely.geos_version_string,
        },
        'description': (
            'Polygons that shapely.Polygon(...).is_valid accepts; area is Polygon.area; touching is '
            'whether two rings intersect; failedBefore marks the polygons that the kernel flagged or '
            'triangulated wrongly before the review fixes of 2026-11-11'
        ),
        'generatedCount': len(generated),
        'review': [get_entry(name, rings) for name, rings in REVIEW_INPUTS],
        'families': families,
    }
    with open(DESTINATION, 'w') as file:
        json.dump(fixture, file, separators=(',', ':'))
        file.write('\n')
    print('generated', len(generated), 'kept', len(families), 'failed before', sum(entry['failedBefore'] for entry in families))
    print('touching', sum(entry['touching'] for entry in families), 'review', len(fixture['review']))
    print(f'-> {DESTINATION}')


if __name__ == '__main__':
    main()
