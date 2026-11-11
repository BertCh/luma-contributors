# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""RVT sky-view factor, anisotropic sky-view factor and openness references.

Calls unmodified rvt.vis.sky_view_factor (presets) and
rvt.vis.sky_view_factor_compute (free polynomial level and minimum weight) of
rvt-py on small elevation models where the direction convention is unambiguous.
Nothing of the GPU kernel is ported here.

Array convention. rvt has no georeference: it works on a bare 2D array. The
arrays here are north-up rasters, row 0 is the northern row and column 0 the
western column, which is how rvt.default hands a GeoTIFF to rvt.vis.

What rvt-py 2.2.1 does, read from rvt/vis.py and confirmed by `conventionProbe`
below (the generator asserts it):
- horizon_shift_vector gives direction i the angle a_i = i * 360 / n and the
  roll (round(cos(a_i) * r), round(sin(a_i) * r)) over (rows, columns);
  numpy.roll moves the neighbour at (row - shift_rows, column - shift_columns)
  onto the pixel. Angle 0 therefore looks north (toward row 0), 90 looks west,
  180 south and 270 east: the angle runs counterclockwise from north.
- The anisotropic weight of direction i is
  (1 - a_min_weight) * cos((a_i - a_main_direction) / 2) ** a_poly_level +
  a_min_weight, with a_i in [0, 360) and no wrap of the difference, and
  asvf = sum(w_i * (1 - sin(max(h_i, 0)))) / sum(w_i). The weight is largest for
  the direction whose angle equals asvf_dir, so asvf_dir = 315 (the default)
  weights the north-east sky (compass azimuth 45).
- svf = mean(1 - sin(max(h_i, 0))); opns = 90 - mean(h_i) in degrees.
  Negative openness is opns of the negated elevation.
- Samples are whole pixels: radii from 1 to svf_r_max in steps of one third,
  rounded to pixel offsets, each at its own centre-to-centre distance. On the
  four axes these are the pixels 1..svf_r_max; on the diagonals the pixels
  (k, k) for k up to round(svf_r_max / sqrt(2)). Other directions follow a
  rounded staircase, which is not the ray a bilinear march samples.
- The array is reflect-padded by svf_r_max, so pixels nearer than svf_r_max to
  an edge see a mirrored terrain. `interior` names the pixels that do not.
- rvt works in float32.

Elevation models (19 x 19, 2 m cells, radius 6 pixels, flat ground at 100 m):
- wall-<side>: a 12 m plateau on the six outer rows or columns of one side, or
  on the north-east corner triangle (column - row >= 8). It lies wholly outside
  the interior, on one side of every interior pixel.
- ramp-<side>: a plane of gradient 0.5 rising toward that side.
- rough: a fixed sum of sines with no symmetry.
- wall-north-north-east: the same plateau beyond 3.5 pixels from the centre
  along compass azimuth 22.5, for sixteen directions and main directions every
  22.5 degrees. Only the ranking of the centre pixel is meant to be compared:
  off the axes and diagonals rvt's staircase is not a ray.
No random numbers are used.
"""
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import numpy
import rvt.vis

OUTPUT = Path(__file__).parent / 'fixtures'
VERSIONS = {name: importlib.metadata.version(name) for name in ['numpy', 'rvt-py']}
VERSIONS.update(python=platform.python_version())

SIZE = 19
RESOLUTION = 2.0
RADIUS = 6
CENTER = SIZE // 2
MAIN_DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]
# rvt.vis.sky_view_factor presets: asvf_level 1 is (4, 0.4), 2 is (8, 0.1).
PRESETS = {'low': 1, 'high': 2}
# An odd level keeps the sign of the cosine: weights go below the minimum weight.
ODD = {'level': 3, 'minimumWeight': 0.2}
# Unit vectors (north, east) of the side a wall stands on or a ramp rises toward.
SIDES = {'north': (1, 0), 'east': (0, 1), 'south': (-1, 0), 'west': (0, -1),
         'north-east': (math.sqrt(0.5), math.sqrt(0.5))}
# The rvt main direction that looks at each side, if the angle is counterclockwise from north.
EXPECTED_HEAVIEST = {'north': 0, 'west': 90, 'south': 180, 'east': 270, 'north-east': 315}


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, numpy.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, (numpy.integer, numpy.bool_)):
        return int(value)
    if isinstance(value, (float, numpy.floating)):
        # rvt results are float32; nine significant digits round-trip them.
        return float(f'{float(value):.9g}') if math.isfinite(value) else None
    return value


def buildWall(side):
    rows, columns = numpy.indices((SIZE, SIZE))
    raised = {'north': rows < RADIUS, 'south': rows >= SIZE - RADIUS,
              'west': columns < RADIUS, 'east': columns >= SIZE - RADIUS,
              'north-east': columns - rows >= 8}[side]
    return numpy.where(raised, 112.0, 100.0)


def buildWallNorthNorthEast():
    rows, columns = numpy.indices((SIZE, SIZE))
    azimuth = math.radians(22.5)
    distance = (CENTER - rows) * math.cos(azimuth) + (columns - CENTER) * math.sin(azimuth)
    return numpy.where(distance >= 3.5, 112.0, 100.0)


def buildRamp(side):
    rows, columns = numpy.indices((SIZE, SIZE))
    north, east = SIDES[side]
    # Row 0 is north: northing falls as the row index grows.
    distance = ((CENTER - rows) * north + (columns - CENTER) * east) * RESOLUTION
    return 100.0 + 0.5 * distance


def buildRough():
    rows, columns = numpy.indices((SIZE, SIZE))
    return (100.0 + 5.0 * numpy.sin(0.9 * columns + 0.35 * rows) + 4.0 * numpy.cos(0.55 * rows - 0.7 * columns)
            + 3.0 * numpy.sin(1.7 * rows + 0.2) + 2.0 * numpy.cos(2.3 * columns + 1.1 * rows))


def computePreset(elevation, directionCount, mainDirection, preset):
    return rvt.vis.sky_view_factor(
        dem=elevation.copy(), resolution=RESOLUTION, compute_svf=True, compute_opns=True, compute_asvf=True,
        svf_n_dir=directionCount, svf_r_max=RADIUS, svf_noise=0, asvf_dir=mainDirection,
        asvf_level=PRESETS[preset], ve_factor=1, no_data=None)


def computeOdd(elevation, directionCount, mainDirection):
    # sky_view_factor only offers the two presets; this is the routine it calls, with the same
    # preparation (float32, elevation divided by the resolution, minimum radius 1).
    return rvt.vis.sky_view_factor_compute(
        height_arr=elevation / numpy.float32(RESOLUTION), radius_max=RADIUS, radius_min=1,
        num_directions=directionCount, compute_svf=False, compute_opns=False, compute_asvf=True,
        a_main_direction=mainDirection, a_poly_level=ODD['level'], a_min_weight=ODD['minimumWeight'])['asvf']


def cropInterior(values):
    return values[RADIUS:SIZE - RADIUS, RADIUS:SIZE - RADIUS].ravel()


def buildCase(name, elevation, directionCount, mainDirections):
    elevation = elevation.astype(numpy.float32)
    anisotropic = []
    isotropic = None
    for preset in PRESETS:
        for mainDirection in mainDirections:
            result = computePreset(elevation, directionCount, mainDirection, preset)
            if isotropic is None:
                isotropic = result
            else:
                # The isotropic outputs do not depend on the anisotropy settings.
                numpy.testing.assert_array_equal(result['svf'], isotropic['svf'])
                numpy.testing.assert_array_equal(result['opns'], isotropic['opns'])
            level, minimumWeight = {'low': (4, 0.4), 'high': (8, 0.1)}[preset]
            anisotropic.append({'mainDirection': mainDirection, 'level': level, 'minimumWeight': minimumWeight,
                                'interior': cropInterior(result['asvf'])})
    for mainDirection in mainDirections:
        anisotropic.append({'mainDirection': mainDirection, **ODD,
                            'interior': cropInterior(computeOdd(elevation, directionCount, mainDirection))})
    negative = rvt.vis.sky_view_factor(
        dem=-elevation, resolution=RESOLUTION, compute_svf=False, compute_opns=True, compute_asvf=False,
        svf_n_dir=directionCount, svf_r_max=RADIUS, svf_noise=0)['opns']
    return {'name': f'{name}-{directionCount}', 'elevationModel': name, 'directionCount': directionCount,
            'elevation': elevation.ravel(),
            # Whole grid, including the edge band where rvt sees its reflect padding.
            'skyViewFactor': isotropic['svf'].ravel(), 'positiveOpenness': isotropic['opns'].ravel(),
            'negativeOpenness': negative.ravel(), 'anisotropic': anisotropic}


def probeConvention(cases):
    """For each wall, the main direction with the lowest centre asvf is the one that looks at the wall."""
    probe = []
    for side, expected in EXPECTED_HEAVIEST.items():
        case = next(entry for entry in cases if entry['name'] == f'wall-{side}-8')
        centre = (CENTER - RADIUS) * (SIZE - 2 * RADIUS) + (CENTER - RADIUS)
        values = {entry['mainDirection']: float(entry['interior'][centre])
                  for entry in case['anisotropic'] if entry['level'] == 4}
        heaviest = min(values, key=values.get)
        assert heaviest == expected, (side, heaviest, values)
        # The wall is symmetric about its own direction, and so are rvt's weights.
        for offset in (45, 90, 135):
            assert abs(values[(expected + offset) % 360] - values[(expected - offset) % 360]) < 1e-6, (side, values)
        assert values[(expected + 180) % 360] == max(values.values()), (side, values)
        probe.append({'wallSide': side, 'compassAzimuthOfWall': (360 - expected) % 360,
                      'heaviestMainDirection': heaviest, 'lightestMainDirection': (expected + 180) % 360,
                      'centreValueByMainDirection': values})
    # Sixteen directions, a wall at compass azimuth 22.5: rvt main direction 337.5 looks at it.
    case = next(entry for entry in cases if entry['name'] == 'wall-north-north-east-16')
    centre = (CENTER - RADIUS) * (SIZE - 2 * RADIUS) + (CENTER - RADIUS)
    values = {entry['mainDirection']: float(entry['interior'][centre])
              for entry in case['anisotropic'] if entry['level'] == 4}
    heaviest = min(values, key=values.get)
    lightest = max(values, key=values.get)
    assert (heaviest, lightest) == (337.5, 157.5), (heaviest, lightest, values)
    probe.append({'wallSide': 'north-north-east', 'compassAzimuthOfWall': 22.5,
                  'heaviestMainDirection': heaviest, 'lightestMainDirection': lightest,
                  'centreValueByMainDirection': values})
    return probe


def main():
    allDirections = MAIN_DIRECTIONS
    few = [315, 90]
    cases = []
    for side in SIDES:
        cases.append(buildCase(f'wall-{side}', buildWall(side), 8, allDirections))
        cases.append(buildCase(f'wall-{side}', buildWall(side), 4, [0, 90, 180, 270]))
        cases.append(buildCase(f'ramp-{side}', buildRamp(side), 8, few))
    cases.append(buildCase('rough', buildRough(), 8, allDirections))
    cases.append(buildCase('rough', buildRough(), 4, few))
    # Sixteen directions: rvt's rounded staircase differs from any ray; kept to size the difference.
    cases.append(buildCase('ramp-north-east', buildRamp('north-east'), 16, few))
    cases.append(buildCase('rough', buildRough(), 16, few))
    cases.append(buildCase('wall-north-north-east', buildWallNorthNorthEast(), 16,
                           [index * 22.5 for index in range(16)]))
    payload = {
        'versions': VERSIONS, 'conventions': __doc__,
        'generatorSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference': 'rvt.vis.sky_view_factor and rvt.vis.sky_view_factor_compute, unmodified',
        'citations': ['https://doi.org/10.3390/rs3020398', 'https://doi.org/10.3390/rs11070747',
                      'https://github.com/EarthObservation/RVT_py'],
        'width': SIZE, 'height': SIZE, 'resolution': RESOLUTION, 'maximumRadius': RADIUS,
        'interior': {'first': RADIUS, 'end': SIZE - RADIUS},
        'conventionProbe': probeConvention(cases), 'cases': cases}
    path = OUTPUT / 'terrain-sky-view-rvt.json'
    path.write_text(json.dumps(encode(payload), separators=(',', ':'), allow_nan=False) + '\n')
    assert path.stat().st_size <= 1_500_000, path.stat().st_size
    print(path.name, path.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
