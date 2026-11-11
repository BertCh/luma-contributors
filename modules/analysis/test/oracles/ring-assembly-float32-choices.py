# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""GEOS hole ownership and unions for rings that defeat float32 areas and float32 turn keys.

Writes fixtures/ring-assembly-float32-choices.json next to this script. Run from anywhere with the
pinned interpreter of requirements.txt:
    <venv>/bin/python -I modules/analysis/test/oracles/ring-assembly-float32-choices.py

Consumed by
test/gpu-spatial-analysis/ring-assembly/gpu-segment-ring-assembly-float32-choices.spec.ts, which
runs the rings through GPUSegmentRingAssembly and GPUCoverageDissolve.

Closing pass of the math correctness review (2026-11-11), two float32 choices of
GPUSegmentRingAssembly:

1. Hole ownership. The innermost shell of a hole was chosen by comparing float32 ring areas. The
   `wedges` scenes are thin wedges with the apex near the origin and a far side a few float32
   steps tall, nested two or four deep (shell, hole, shell, hole), each ring one step inside the
   one before it, with a vertex at every step of the far side. The areas of consecutive rings
   differ by a few units in the last place of the products that make them up, so the float32
   areas come out in the wrong order. `bands` holds the same nesting on the float32 grid of
   projected metres (easting 500,000, northing 4,500,000), where the areas stay in order.
   Oracle: shapely.build_area of the rings (GEOS BuildArea: even-odd nesting of the linework),
   which returns each shell with its holes; `shellOfRing` is read off that result by matching
   vertex sets.
2. Turn order at a vertex matched within the tolerance. `turns` holds two triangles whose tips
   are a quarter unit apart, so the ring assembly matches them as one vertex, and whose edges
   out of that vertex differ in direction by 2^-43 radians (lattice directions (N - 1, N - 2) and
   (N, N - 1), N = 2^21, cross product 1): the scene of
   gpu-segment-ring-assembly-exact-turns.spec.ts with the second triangle moved. Oracle:
   shapely.union_all of the two triangles (they are disjoint): two parts.
3. The same choice when the float32 direction vectors round (review fixes of 2026-11-11,
   finding frc-geometry-06 of fix-review-closing-geometry.md). `roundedTurns` holds two
   triangles that meet within the tolerance near (1, 1), with far vertices one to five million
   away and edges out of the tips about 1e-7 radians apart, so that `end - start` of a long edge
   is not a float32 value. Each scene records the successors that the exact directions give
   (computed with `fractions`) and those that the float32 differences give. Oracle of the rings:
   shapely.union_all of the two triangles has two parts.
4. The same scenes as their mirror images (re-review of 2026-11-11, finding frc-geometry-07).
   `polarTurns` holds 240 such pairs in degrees beside the antimeridian and the pole, three of
   them with a candidate exactly opposite to the incoming segment. The mirror image of a scene
   with the interior on the other side must give the mirror image of the rings. Oracle:
   shapely.union_all of each pair has two parts.

Every coordinate is a float32 value; the script checks it.
"""

import json
import math
import os
import platform
import random
from fractions import Fraction

import numpy as np
import shapely
from shapely import GeometryCollection, LinearRing, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
DESTINATION = os.path.join(HERE, 'fixtures', 'ring-assembly-float32-choices.json')


def to_float32(value):
    return float(np.float32(value))


def get_step(value):
    """One float32 step at value."""
    return 2.0 ** (math.floor(math.log2(abs(value))) - 23)


def get_orientation(a, b, c):
    """Exact sign of cross(b - a, c - a)."""
    cross = (Fraction(b[0]) - Fraction(a[0])) * (Fraction(c[1]) - Fraction(a[1])) - (
        Fraction(b[1]) - Fraction(a[1])
    ) * (Fraction(c[0]) - Fraction(a[0]))
    return (cross > 0) - (cross < 0)


def is_strictly_inside(point, triangle):
    return all(
        get_orientation(triangle[index], triangle[(index + 1) % 3], point) > 0 for index in range(3)
    )


def get_wedge_rings(apexes, far_x, far_y, far):
    """Ring k: its apex, then the far side at x = far_x - k steps, from step k to step far - k.

    Even rings are counter-clockwise (shells), odd rings clockwise (holes).
    """
    step_x = get_step(far_x)
    step_y = get_step(far_y)
    rings = []
    for level, apex in enumerate(apexes):
        x = to_float32(far_x - level * step_x)
        side = [(x, to_float32(far_y + step * step_y)) for step in range(level, far - level + 1)]
        corners = [apex, side[0], side[-1]]
        assert get_orientation(*corners) > 0
        if level > 0:
            previous = rings[level - 1]['corners']
            assert all(is_strictly_inside(point, previous) for point in [apex, *side])
        rings.append({'corners': corners, 'ring': [apex, *side]})
    return [
        entry['ring'] if level % 2 == 0 else [entry['ring'][0], *reversed(entry['ring'][1:])]
        for level, entry in enumerate(rings)
    ]


def get_random_wedge(generator, depth, far):
    """A nested wedge with random far corner and apexes, or None when an apex does not fit."""
    far_x = to_float32(4096 + generator.random() * 4095)
    far_y = to_float32(4096 + generator.random() * 4095)
    step_x = get_step(far_x)
    step_y = get_step(far_y)
    apexes = [(to_float32(1 + generator.random()), to_float32(1 + generator.random()))]
    for level in range(1, depth):
        # The ring before this one, as a triangle; the new apex goes between its two long sides.
        outer_apex = apexes[level - 1]
        outer_x = to_float32(far_x - (level - 1) * step_x)
        low = (outer_x, to_float32(far_y + (level - 1) * step_y))
        high = (outer_x, to_float32(far_y + (far - level + 1) * step_y))
        apex_x = to_float32(outer_apex[0] + 3 + generator.random())
        fraction = (apex_x - outer_apex[0]) / (outer_x - outer_apex[0])
        low_y = outer_apex[1] + (low[1] - outer_apex[1]) * fraction
        high_y = outer_apex[1] + (high[1] - outer_apex[1]) * fraction
        apex = (apex_x, to_float32((low_y + high_y) / 2))
        if not is_strictly_inside(apex, [outer_apex, low, high]):
            return None
        apexes.append(apex)
    try:
        return get_wedge_rings(apexes, far_x, far_y, far)
    except AssertionError:
        return None


def get_projected_bands(depth, step_count, multiplier):
    """Nested thin bands on the float32 grid of projected metres.

    In grid units (1/32 m east, 1/2 m north) ring k is the parallelogram with corners (2k, k),
    (2k, k) + (n - k) d, (2k, T - k) + (n - k) d, (2k, T - k), with d = (64, 8) * multiplier and
    T = 2 * depth + 1, and a vertex at every multiple of d on the long sides.
    """
    east = 500000.0
    north = 4500000.0
    total = 2 * depth + 1
    rings = []
    for level in range(depth):
        count = step_count - level
        lower = [
            (east + (2 * level + step * 64 * multiplier) / 32, north + (level + step * 8 * multiplier) / 2)
            for step in range(count + 1)
        ]
        upper = [
            (
                east + (2 * level + step * 64 * multiplier) / 32,
                north + (total - level + step * 8 * multiplier) / 2,
            )
            for step in range(count + 1)
        ]
        ring = lower + upper[::-1]
        rings.append(ring if level % 2 == 0 else ring[::-1])
    return rings


def get_ring_key(coordinates):
    return frozenset((float(x), float(y)) for x, y in coordinates)


def get_nesting_scene(name, rings):
    for ring in rings:
        for x, y in ring:
            assert to_float32(x) == x and to_float32(y) == y, (name, x, y)
    built = shapely.build_area(GeometryCollection([LinearRing(ring) for ring in rings]))
    keys = [get_ring_key(ring) for ring in rings]
    shell_of_ring = [None] * len(rings)
    polygon_areas = []
    for polygon in shapely.get_parts(built):
        assert polygon.is_valid, name
        shell = keys.index(get_ring_key(polygon.exterior.coords))
        shell_of_ring[shell] = shell
        for interior in polygon.interiors:
            shell_of_ring[keys.index(get_ring_key(interior.coords))] = shell
        polygon_areas.append(polygon.area)
    assert None not in shell_of_ring, name
    return {
        'name': name,
        # Rings without the closing vertex; shells counter-clockwise, holes clockwise.
        'rings': [[[float(x), float(y)] for x, y in ring] for ring in rings],
        'shellOfRing': shell_of_ring,
        'polygonAreas': polygon_areas,
    }


def get_wedge_scenes():
    scenes = []
    # Found by a search on the GPU: with the float32 area rule the hole of each was left without
    # a shell (float32 areas 6.5722 against 7.2638, and 6.5031 against 6.9722).
    curated = [
        (
            [(1.3911610841751099, 1.1006715297698975), (5.029665470123291, 4.701545238494873)],
            5928.61328125,
            5867.0185546875,
        ),
        (
            [(1.3915069103240967, 1.2200878858566284), (4.4022321701049805, 4.042481899261475)],
            6032.61328125,
            5655.1650390625,
        ),
    ]
    for index, (apexes, far_x, far_y) in enumerate(curated):
        scenes.append(get_nesting_scene(f'wedge-12-steps-{index}', get_wedge_rings(apexes, far_x, far_y, 12)))
    generator = random.Random(20261114)
    for depth, far, count in [(4, 60, 4), (2, 120, 3), (4, 120, 4)]:
        made = 0
        while made < count:
            rings = get_random_wedge(generator, depth, far)
            if rings is None:
                continue
            scenes.append(get_nesting_scene(f'wedge-{far}-steps-depth-{depth}-{made}', rings))
            made += 1
    return scenes


def get_band_scenes():
    return [
        get_nesting_scene('bands-60-m', get_projected_bands(4, 30, 1)),
        get_nesting_scene('bands-18-km', get_projected_bands(4, 30, 300)),
        get_nesting_scene('bands-18-km-200-steps', get_projected_bands(4, 200, 45)),
    ]


def get_turn_scenes():
    half = 2**21
    tip = (1.0, 1.0)
    first_direction = (half - 1, half - 2)
    second_direction = (half, half - 1)
    scenes = []
    for name, shift in [('left', (-0.25, 0.0)), ('up', (0.0, 0.25))]:
        moved = (tip[0] + shift[0], tip[1] + shift[1])
        # A sliver between the two directions, and a triangle from the second direction to straight up.
        sliver = [
            tip,
            (tip[0] + first_direction[0], tip[1] + first_direction[1]),
            (
                tip[0] + first_direction[0] + second_direction[0],
                tip[1] + first_direction[1] + second_direction[1],
            ),
        ]
        triangle = [
            moved,
            (moved[0] + second_direction[0], moved[1] + second_direction[1]),
            (moved[0], moved[1] + 2 * half - 2),
        ]
        for ring in (sliver, triangle):
            for x, y in ring:
                assert to_float32(x) == x and to_float32(y) == y, (name, x, y)
        polygons = [Polygon(sliver), Polygon(triangle)]
        assert all(polygon.is_valid and polygon.exterior.is_ccw for polygon in polygons)
        assert polygons[0].disjoint(polygons[1]), name
        union = shapely.union_all(polygons)
        parts = shapely.get_parts(union)
        keys = [get_ring_key(part.exterior.coords) for part in parts]
        assert sorted(map(sorted, keys)) == sorted(sorted(get_ring_key(ring)) for ring in (sliver, triangle))
        scenes.append(
            {
                'name': f'tips-a-quarter-apart-{name}',
                'rings': [[[float(x), float(y)] for x, y in ring] for ring in (sliver, triangle)],
                'separation': polygons[0].distance(polygons[1]),
                'partCount': len(parts),
                'partAreas': sorted(part.area for part in parts),
            }
        )
    return scenes


def get_cross_sign(first, second):
    """Exact sign of the cross product of two vectors of fractions or floats."""
    cross = Fraction(first[0]) * Fraction(second[1]) - Fraction(first[1]) * Fraction(second[0])
    return (cross > 0) - (cross < 0)


def get_turn_class(incoming, outgoing):
    """0 right turn, 1 straight on, 2 left turn or u-turn: the classes of the kernel."""
    side = get_cross_sign(incoming, outgoing)
    if side < 0:
        return 0
    if side > 0:
        return 2
    if incoming[0] != 0:
        behind = (incoming[0] > 0) != (outgoing[0] > 0)
    else:
        behind = (incoming[1] > 0) != (outgoing[1] > 0)
    return 2 if behind else 1


def get_successor(incoming, candidates):
    """The candidate that turns furthest left of `incoming` (interior on the left); ties to the lowest row."""
    best = None
    for row, outgoing in candidates:
        turn_class = get_turn_class(incoming, outgoing)
        if best is None:
            best = (row, turn_class, outgoing)
            continue
        order = turn_class - best[1]
        if order == 0 and turn_class != 1:
            order = -get_cross_sign(outgoing, best[2])
        if order > 0 or (order == 0 and row < best[0]):
            best = (row, turn_class, outgoing)
    return best[0]


def get_exact_direction(segment):
    start, end = segment
    return (Fraction(end[0]) - Fraction(start[0]), Fraction(end[1]) - Fraction(start[1]))


def get_float32_direction(segment):
    start, end = segment
    return (
        float(np.float32(end[0]) - np.float32(start[0])),
        float(np.float32(end[1]) - np.float32(start[1])),
    )


def get_tip_successors(segments, get_direction):
    """Successors of the two segments that end at the tips (rows 2 and 5); candidates rows 0 and 3."""
    candidates = [(row, get_direction(segments[row])) for row in (0, 3)]
    return [get_successor(get_direction(segments[row]), candidates) for row in (2, 5)]


def get_rounded_turn_scenes():
    """Two triangles that meet within the tolerance, with directions that float32 differences round.

    A thin sliver and a triangle beside it, both with a tip near (1, 1) and far vertices one to five
    million away, so the float32 difference `end - start` of a long edge is rounded. The edges out
    of the tips differ in direction by about 1e-7 radians. Two forms:

    - `apart`: the tips are two points up to 0.3 apart (the triangles are disjoint);
    - `closing`: both triangles start at one point and the segment that closes the sliver ends up
      to 0.3 from it (the triangles touch at that point).

    With the exact directions each closing segment continues into its own triangle. `float32Choice`
    is the pair of successors that the float32 direction vectors give; in the scenes kept for
    `roundingChangesChoice` it differs, so a rule on rounded vectors loses a ring. Oracle of the
    result: shapely.union_all of the two triangles has two parts with the vertices of the two rings.
    """
    generator = random.Random(20261111)
    half = 2.0**21
    scenes = []
    kept = {('apart', True): 0, ('apart', False): 0, ('closing', True): 0, ('closing', False): 0}
    wanted = {('apart', True): 24, ('apart', False): 8, ('closing', True): 24, ('closing', False): 8}
    attempt = 0
    while any(kept[key] < wanted[key] for key in wanted):
        attempt += 1
        form = 'apart' if attempt % 2 else 'closing'
        scale = generator.uniform(0.3, 1.0)
        tip = (to_float32(generator.uniform(0.6, 1.4)), to_float32(generator.uniform(0.6, 1.4)))

        def jitter():
            return generator.uniform(-0.6, 0.6)

        first_direction = ((half - 1) * scale + jitter(), (half - 2) * scale + jitter())
        second_direction = (1.5 * first_direction[0] + jitter(), 1.5 * first_direction[1] + jitter())
        moved = (
            to_float32(tip[0] + generator.uniform(-0.3, 0.3)),
            to_float32(tip[1] + generator.uniform(-0.3, 0.3)),
        )
        if moved == tip:
            continue
        # apart: the triangle starts at the moved point. closing: the sliver closes onto it.
        triangle_tip = moved if form == 'apart' else tip
        sliver_close = tip if form == 'apart' else moved
        sliver = [
            tip,
            (to_float32(tip[0] + first_direction[0]), to_float32(tip[1] + first_direction[1])),
            (
                to_float32(tip[0] + first_direction[0] + second_direction[0]),
                to_float32(tip[1] + first_direction[1] + second_direction[1]),
            ),
        ]
        triangle = [
            triangle_tip,
            (to_float32(triangle_tip[0] + second_direction[0]), to_float32(triangle_tip[1] + second_direction[1])),
            (triangle_tip[0], to_float32(triangle_tip[1] + 2 * half * scale)),
        ]
        segments = [
            (sliver[0], sliver[1]),
            (sliver[1], sliver[2]),
            (sliver[2], sliver_close),
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ]
        # Only the tips are within the matching distance of each other.
        points = [tip, moved, sliver[1], sliver[2], triangle[1], triangle[2]]
        if any(
            max(abs(points[first][0] - points[second][0]), abs(points[first][1] - points[second][1])) < 4
            for first in range(len(points))
            for second in range(first + 1, len(points))
            if second >= 2
        ):
            continue
        if get_tip_successors(segments, get_exact_direction) != [0, 3]:
            continue
        polygons = [Polygon(sliver), Polygon(triangle)]
        if not all(polygon.is_valid and polygon.exterior.is_ccw for polygon in polygons):
            continue
        intersection = polygons[0].intersection(polygons[1])
        if intersection.area != 0:
            continue
        if form == 'apart' and not polygons[0].disjoint(polygons[1]):
            continue
        if form == 'closing' and not (polygons[0].touches(polygons[1]) and intersection.geom_type == 'Point'):
            continue
        float32_choice = get_tip_successors(segments, get_float32_direction)
        key = (form, float32_choice != [0, 3])
        if kept[key] >= wanted[key]:
            continue
        kept[key] += 1
        parts = shapely.get_parts(shapely.union_all(polygons))
        part_keys = sorted(sorted(get_ring_key(part.exterior.coords)) for part in parts)
        assert part_keys == sorted(sorted(get_ring_key(ring)) for ring in (sliver, triangle)), form
        scenes.append(
            {
                'name': f'{form}-{len(scenes)}',
                'form': form,
                # Rows 0 to 2 are the sliver and rows 3 to 5 the triangle: x0, y0, x1, y1.
                'segments': [[float(value) for value in (*start, *end)] for start, end in segments],
                'rings': [[[float(x), float(y)] for x, y in ring] for ring in (sliver, triangle)],
                'partCount': len(parts),
                'float32Choice': float32_choice,
                'roundingChangesChoice': float32_choice != [0, 3],
            }
        )
    for scene in scenes:
        for value in (value for segment in scene['segments'] for value in segment):
            assert to_float32(value) == value, scene['name']
    return scenes


def is_exactly_opposite(first, second):
    """Whether two segments have exactly opposite directions (cross product 0, dot product negative)."""
    first_direction = get_exact_direction(first)
    second_direction = get_exact_direction(second)
    dot = first_direction[0] * second_direction[0] + first_direction[1] * second_direction[1]
    return get_cross_sign(first_direction, second_direction) == 0 and dot < 0


def get_polar_turn_scenes():
    """The scenes of `roundedTurns` in degrees beside the antimeridian and the pole.

    Tips near longitude 179.99 and latitude 89.4 on the float32 grid of 2^-12 degrees, far vertices
    up to 40 degrees away toward the south-west. The reviewer's generator of the re-review
    (finding frc-geometry-07), seed 5, which gives the reviewer's 240 scenes value for value:
    `apart` scenes have two tips up to 0.3 grid steps apart,
    `closing` scenes one tip and a sliver that closes up to 0.3 steps from it. In
    `oppositeCandidate` scenes the segment that closes the sliver is exactly opposite to the
    sliver's own first edge (cross product 0 from the four endpoints), so that edge is a u-turn
    candidate; it is no twin of the closing segment, whose end is not the tip.
    """
    generator = random.Random(5)
    half = 2.0**17
    unit = 2.0**-12
    center = (179.99, 89.4)
    scenes = []
    controls = {'apart': 0, 'closing': 0}
    attempt = 0
    while len(scenes) < 240 and attempt < 400000:
        attempt += 1
        form = 'apart' if attempt % 2 else 'closing'
        scale = generator.uniform(0.3, 1.0)
        tip = (
            to_float32(center[0] + unit * generator.uniform(-0.4, 0.4)),
            to_float32(center[1] + unit * generator.uniform(-0.4, 0.4)),
        )

        def jitter():
            return generator.uniform(-0.6, 0.6)

        first_direction = (-unit * ((half - 1) * scale + jitter()), -unit * ((half - 2) * scale + jitter()))
        second_direction = (1.5 * first_direction[0] - unit * jitter(), 1.5 * first_direction[1] - unit * jitter())
        moved = (
            to_float32(tip[0] + unit * generator.uniform(-0.3, 0.3)),
            to_float32(tip[1] + unit * generator.uniform(-0.3, 0.3)),
        )
        if moved == tip:
            continue
        triangle_tip = moved if form == 'apart' else tip
        sliver_close = tip if form == 'apart' else moved
        sliver = [
            tip,
            (to_float32(tip[0] + first_direction[0]), to_float32(tip[1] + first_direction[1])),
            (
                to_float32(tip[0] + first_direction[0] + second_direction[0]),
                to_float32(tip[1] + first_direction[1] + second_direction[1]),
            ),
        ]
        triangle = [
            triangle_tip,
            (to_float32(triangle_tip[0] + second_direction[0]), to_float32(triangle_tip[1] + second_direction[1])),
            (triangle_tip[0], to_float32(triangle_tip[1] - unit * 2 * half * scale)),
        ]
        segments = [
            (sliver[0], sliver[1]),
            (sliver[1], sliver[2]),
            (sliver[2], sliver_close),
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ]
        points = [tip, sliver_close, triangle_tip, sliver[1], sliver[2], triangle[1], triangle[2]]
        if any(
            max(abs(points[first][0] - points[second][0]), abs(points[first][1] - points[second][1])) < 4 * unit
            for first in range(len(points))
            for second in range(first + 1, len(points))
            if second >= 3
        ):
            continue
        if get_tip_successors(segments, get_exact_direction) != [0, 3]:
            continue
        polygons = [Polygon(sliver), Polygon(triangle)]
        if not all(polygon.is_valid and polygon.exterior.is_ccw for polygon in polygons):
            continue
        intersection = polygons[0].intersection(polygons[1])
        if intersection.area != 0:
            continue
        if form == 'apart' and not polygons[0].disjoint(polygons[1]):
            continue
        if form == 'closing' and not (polygons[0].touches(polygons[1]) and intersection.geom_type == 'Point'):
            continue
        if get_tip_successors(segments, get_float32_direction) == [0, 3]:
            if controls[form] >= 60:
                continue
            controls[form] += 1
        scenes.append(
            {
                'name': f'{form}-{len(scenes)}',
                'form': form,
                # Rows 0 to 2 are the sliver and rows 3 to 5 the triangle: x0, y0, x1, y1.
                'segments': [[float(value) for value in (*start, *end)] for start, end in segments],
                'partCount': len(shapely.get_parts(shapely.union_all(polygons))),
                'oppositeCandidate': any(
                    is_exactly_opposite(segments[row], segments[candidate]) for row in (2, 5) for candidate in (0, 3)
                ),
            }
        )
    for scene in scenes:
        for value in (value for segment in scene['segments'] for value in segment):
            assert to_float32(value) == value, scene['name']
    return scenes


def main():
    fixture = {
        'generator': 'ring-assembly-float32-choices.py',
        'versions': {
            'python': platform.python_version(),
            'shapely': shapely.__version__,
            'geos': shapely.geos_version_string,
            'numpy': np.__version__,
        },
        'description': (
            'wedges and bands: shapely.build_area of nested rings, shellOfRing[i] is the ring index '
            'of the shell that owns ring i; turns: shapely.union_all of two disjoint triangles'
        ),
        'wedges': get_wedge_scenes(),
        'bands': get_band_scenes(),
        'turns': get_turn_scenes(),
        'roundedTurns': get_rounded_turn_scenes(),
        'polarTurns': get_polar_turn_scenes(),
    }
    with open(DESTINATION, 'w') as file:
        json.dump(fixture, file, separators=(',', ':'))
        file.write('\n')
    for family in ('wedges', 'bands'):
        for scene in fixture[family]:
            print(family, scene['name'], [len(ring) for ring in scene['rings']], scene['shellOfRing'])
    for scene in fixture['turns']:
        print('turns', scene['name'], scene['partCount'], scene['separation'])
    rounded = fixture['roundedTurns']
    print(
        'roundedTurns',
        len(rounded),
        'rounding changes the choice in',
        sum(scene['roundingChangesChoice'] for scene in rounded),
    )
    polar = fixture['polarTurns']
    print(
        'polarTurns',
        len(polar),
        'with an exactly opposite candidate:',
        [scene['name'] for scene in polar if scene['oppositeCandidate']],
    )
    print(f'-> {DESTINATION}')


if __name__ == '__main__':
    main()
