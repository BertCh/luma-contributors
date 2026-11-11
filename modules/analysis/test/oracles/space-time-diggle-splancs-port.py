# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Space-time K functions and the Diggle et al. (1995) test as splancs computes them.

THIS IS A PORT OF THE SPLANCS SOURCE, NOT A RUN OF R. R is not installed where the fixtures are
generated. The numbers below come from a NumPy / pure Python implementation written from a reading
of splancs 2.01-45 (CRAN `splancs_2.01-45.tar.gz`, SHA-256
8bccf1d61d7feaab52da07a9c95aa66bcd3986a6b214f13b232c1e2bea4b76d3): `R/stkhat.S`, `R/stmctest.S`,
`R/stsecal.S`, `R/stvmat.S`, `R/stdiagn.S`, `src/stkhat.f`, `src/stsecal.f`, `src/cncvwt.f`,
`src/iplace.f`, `src/plarea.f`. splancs is GPL (>= 2): no splancs code is copied here or executed;
this file states, in its own code, the quantities that source evaluates. A fixture from R itself
(`splancs::stkhat`, `stsecal`, `stmctest`) would be stronger evidence and should replace this one
when R is available.

Writes fixtures/space-time-diggle-splancs-port.json next to this script (consumed by
test/gpu-dataframe/pair-statistics/space-time-diggle-splancs.node.spec.ts and
space-time-diggle-splancs.spec.ts). Run from anywhere:
    <venv>/bin/python -I modules/analysis/test/oracles/space-time-diggle-splancs-port.py
in a venv with numpy and shapely (versions are recorded in the fixture; shapely is used only to
check the edge-correction weight of this file, not to produce a number in the fixture).

What splancs computes, for events `(x_i, y_i, t_i)`, `i = 1..n`, in a polygon of area `A`, time
limits `[tlow, thigh]` with `T = thigh - tlow`, sorted distances `s_1 < ... < s_S` and time lags
`t_1 < ... < t_V`. `d_ij` is the distance and `u_ij = |t_i - t_j|`.

- `stkhat` (`ksthat` in `stkhat.f`):
  - spatial weight `w_ij = 2 pi / (angle of the circle about event i through event j that lies
    inside the polygon)` (`cncvwt.f`, Ripley's isotropic correction);
  - temporal weight `v_ij = 2` when `t_i - tlow <= u_ij` or `thigh - t_i <= u_ij`, else 1. Note
    the `<=`: a lag that exactly reaches a time limit already counts double;
  - a pair is in distance class `k` when `s_k` is the first distance with `s_k >= d_ij`
    (`iplace.f`), and it is dropped when `d_ij >= s_S`; the same for lags with `u_ij >= t_V`. So
    a class is closed above (`d_ij <= s_k`) except the last, which is open (`d_ij < s_S`);
  - `ks(s_k) = A / (n (n - 1)) sum_{i != j} w_ij [d_ij in class <= k]`,
    `kt(t_l) = T / (n (n - 1)) sum_{i != j} v_ij [u_ij in class <= l]`,
    `kst(s_k, t_l) = A T / (n (n - 1)) sum_{i != j} w_ij v_ij [both]`.
  - Not reproduced: when `d_ij == s_S` exactly, `stkhat.f` skips the spatial branch but its test
    `sj .le. smax` then adds to `hst` with the class and weights left over from an earlier pair.
    The scenes here have no pair at that distance (asserted), so the quirk is never reached.
- `stmctest`: the statistic is `sum(kst - outer(ks, kt))`, the plain sum of `D(s_k, t_l) =
  kst - ks kt` over the whole grid of classes. It is NOT standardised: `stsecal` is not called.
  The same sum is recomputed after `times <- sample(times)` for each simulation. splancs returns
  `t0` and `t` and prints the rank of `t0`; surveillance 1.x `stKtest` wraps it and reports
  `p = (1 + #{t >= t0}) / (nsim + 1)`.
- `stsecal` (`stsecal.f`, the diagonal of `stvmat`): the standard error of `kst(s_k, t_l)` under
  random relabelling of the times, from the pair matrices of symmetrised weights
  `(w_ij + w_ji) / 2` over `d_ij <= s_k` and `(v_ij + v_ji) / 2` over `u_ij <= t_l` (both classes
  closed above here, the last included). With `a_i` the row sums, `w1 = sum a_i`, `w2 = sum a_i^2`
  and `w3` the sum of squared entries over ordered pairs (and `v1`, `v2`, `v3` likewise),
  `Var = (A T / (n (n - 1)))^2 (c2 P + c3 Q + c4 w3 v3 - c1 (w1 v1)^2)`,
  `P = (w1^2 - 4 w2 + 2 w3)(v1^2 - 4 v2 + 2 v3)`, `Q = (w2 - w3)(v2 - v3)`,
  `c1 = 1 / (n (n - 1))^2`, `c2 = 1 / (n (n - 1)(n - 2)(n - 3))`, `c3 = 4 / (n (n - 1)(n - 2))`,
  `c4 = 2 / (n (n - 1))`. This is the exact permutation variance of the symmetrised statistic;
  the script checks it against all 720 relabellings of six events.
- `stdiagn`: `R = D / stsecal`, the standardised residual it plots against `ks kt`.

Checks made when the script runs (it fails otherwise), recorded under `checks`:
- the circle angle used for `w_ij` against shapely (a 16,384-vertex circle clipped by the polygon)
  for every ordered pair of the rectangle scene and of an L-shaped polygon scene;
- the `stsecal` variance against brute-force enumeration of the permutations (n = 6);
- no pair of the continuous-time scene lies within 1e-4 of a class boundary or of the temporal
  edge rule, so float32 kernels and this binary64 port bin every pair alike.

The `integerTimes` section is the opposite case, added after the review of the closing pass
(frc-fields-01): 60 events whose times are whole days 0 to 30 in the limits `[0, 30]`, with three
lag grids (8.5 in 5 classes, 7 in 7 classes, 13 in 13 classes). Whole-day times are ordinary
data, and for them the two tie rules of `stkhat.f` act on many pairs: a lag that exactly reaches
a time limit (weight 2), and a lag exactly equal to the largest lag (dropped). With whole-day
breaks every lag also sits on a class break (closed above). The counts of such pairs are stored
with each case. Distances stay off the distance breaks (asserted).

Differences this port does not hide: `cncvwt.f` uses `pi = 3.141592654d0` (a relative 1.3e-10)
and a `1e-7` tolerance for crossings at polygon vertices; this file uses `math.pi` and has no
vertex crossings in its scenes.
"""

import itertools
import json
import math
import os
import platform

import numpy as np
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'space-time-diggle-splancs-port.json')

BOUNDS = [0.0, 0.0, 10.0, 8.0]
TIME_LIMITS = [0.0, 20.0]
MAXIMUM_DISTANCE = 3.0
DISTANCE_CLASS_COUNT = 6
MAXIMUM_TIME_LAG = 6.0
TIME_CLASS_COUNT = 4
EVENT_COUNT = 72
RELABELLING_COUNT = 6
MONTE_CARLO_COUNT = 199
BOUNDARY_MARGIN = 1e-4

INTEGER_TIME_LIMITS = [0.0, 30.0]
INTEGER_EVENT_COUNT = 60
INTEGER_MAXIMUM_DISTANCE = 3.0
INTEGER_DISTANCE_CLASS_COUNT = 4
# [largest lag, number of classes]: no lag on a break; every lag on a break; and a grid whose
# scaled float32 lag `7 / 13 * 13` rounds above 7.
INTEGER_LAG_GRIDS = [[8.5, 5], [7.0, 7], [13.0, 13]]
INTEGER_RELABELLING_COUNT = 3


def get_polygon_area(polygon):
    """Area of a simple polygon given as a list of (x, y)."""
    total = 0.0
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def is_inside_polygon(x, y, polygon):
    """Even-odd rule."""
    inside = False
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def get_circle_angle_inside(x, y, radius, polygon):
    """Angle, in radians, of the circle of `radius` about (x, y) that lies inside the polygon."""
    angles = []
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        # Points of the edge at distance `radius` from the centre: a quadratic in the edge parameter.
        ax, ay = x1 - x, y1 - y
        ex, ey = x2 - x1, y2 - y1
        a = ex * ex + ey * ey
        b = 2 * (ax * ex + ay * ey)
        c = ax * ax + ay * ay - radius * radius
        discriminant = b * b - 4 * a * c
        if discriminant <= 0:
            continue
        for sign in (-1.0, 1.0):
            parameter = (-b + sign * math.sqrt(discriminant)) / (2 * a)
            if 0 <= parameter < 1:
                # The scenes keep crossings away from the vertices, where splancs has a tolerance.
                assert 1e-9 < parameter < 1 - 1e-9, (x, y, radius, parameter)
                angles.append(math.atan2(ay + parameter * ey, ax + parameter * ex) % (2 * math.pi))
    if not angles:
        # splancs assumes the whole circle is inside when nothing crosses; make sure it is.
        assert is_inside_polygon(x + radius, y, polygon), (x, y, radius)
        return 2 * math.pi
    angles.sort()
    inside = 0.0
    for index, start in enumerate(angles):
        end = angles[index + 1] if index + 1 < len(angles) else angles[0] + 2 * math.pi
        middle = (start + end) / 2
        if is_inside_polygon(x + radius * math.cos(middle), y + radius * math.sin(middle), polygon):
            inside += end - start
    return inside


def get_pair_tables(points, times, polygon, time_limits):
    """Distances, lags and the two weights of every ordered pair, as n x n arrays."""
    count = len(times)
    time_low, time_high = min(time_limits), max(time_limits)
    distances = np.zeros((count, count))
    lags = np.zeros((count, count))
    spatial_weights = np.zeros((count, count))
    temporal_weights = np.zeros((count, count))
    for first in range(count):
        for second in range(count):
            if first == second:
                continue
            distance = math.hypot(points[first][0] - points[second][0],
                                  points[first][1] - points[second][1])
            lag = abs(times[first] - times[second])
            distances[first, second] = distance
            lags[first, second] = lag
            spatial_weights[first, second] = 2 * math.pi / get_circle_angle_inside(
                points[first][0], points[first][1], distance, polygon)
            touches = times[first] - time_low <= lag or time_high - times[first] <= lag
            temporal_weights[first, second] = 2.0 if touches else 1.0
    return distances, lags, spatial_weights, temporal_weights


def get_class_indices(values, breaks):
    """First class whose upper break is >= the value; `len(breaks)` when dropped (>= the last)."""
    indices = np.searchsorted(breaks, values, side='left')
    return np.where(values >= breaks[-1], len(breaks), indices)


def get_space_time_k(tables, area, time_limits, distance_breaks, time_breaks):
    """`stkhat`: ks, kt and kst (distance-major)."""
    distances, lags, spatial_weights, temporal_weights = tables
    count = distances.shape[0]
    duration = max(time_limits) - min(time_limits)
    off_diagonal = ~np.eye(count, dtype=bool)
    distance_classes = get_class_indices(distances, distance_breaks)
    time_classes = get_class_indices(lags, time_breaks)
    spatial = np.zeros(len(distance_breaks))
    temporal = np.zeros(len(time_breaks))
    joint = np.zeros((len(distance_breaks), len(time_breaks)))
    for first, second in zip(*np.nonzero(off_diagonal)):
        distance_class = distance_classes[first, second]
        time_class = time_classes[first, second]
        if distance_class < len(distance_breaks):
            spatial[distance_class] += spatial_weights[first, second]
        if time_class < len(time_breaks):
            temporal[time_class] += temporal_weights[first, second]
            if distance_class < len(distance_breaks):
                joint[distance_class, time_class] += (spatial_weights[first, second]
                                                      * temporal_weights[first, second])
    pairs = count * (count - 1)
    spatial_k = np.cumsum(spatial) * area / pairs
    temporal_k = np.cumsum(temporal) * duration / pairs
    k = np.cumsum(np.cumsum(joint, axis=0), axis=1) * area * duration / pairs
    return spatial_k, temporal_k, k


def get_standard_error(tables, area, time_limits, distance_breaks, time_breaks):
    """`stsecal`: the standard error of kst under random relabelling, per class."""
    distances, lags, spatial_weights, temporal_weights = tables
    count = distances.shape[0]
    duration = max(time_limits) - min(time_limits)
    off_diagonal = ~np.eye(count, dtype=bool)
    spatial_pairs = (spatial_weights + spatial_weights.T) / 2
    temporal_pairs = (temporal_weights + temporal_weights.T) / 2

    def get_moments(pair_weights, values, upper):
        selected = np.where(off_diagonal & (values <= upper), pair_weights, 0.0)
        row_sums = selected.sum(axis=1)
        return row_sums.sum(), (row_sums ** 2).sum(), (selected ** 2).sum()

    n = float(count)
    c1 = 1 / (n * (n - 1)) ** 2
    c2 = 1 / (n * (n - 1) * (n - 2) * (n - 3))
    c3 = 4 / (n * (n - 1) * (n - 2))
    c4 = 2 / (n * (n - 1))
    errors = np.zeros((len(distance_breaks), len(time_breaks)))
    for distance_index, distance_break in enumerate(distance_breaks):
        w1, w2, w3 = get_moments(spatial_pairs, distances, distance_break)
        for time_index, time_break in enumerate(time_breaks):
            v1, v2, v3 = get_moments(temporal_pairs, lags, time_break)
            variance = (area * duration) ** 2 * c1 * (
                c2 * (w1 * w1 - 4 * w2 + 2 * w3) * (v1 * v1 - 4 * v2 + 2 * v3)
                + c3 * (w2 - w3) * (v2 - v3) + c4 * w3 * v3 - c1 * (w1 * v1) ** 2)
            errors[distance_index, time_index] = math.sqrt(variance)
    return errors


def get_analysis(points, times, polygon, time_limits, distance_breaks, time_breaks):
    """stkhat, the stmctest statistic, stsecal and the stdiagn residual of one labelling.

    `stsecal` divides by `(n - 2)(n - 3)`: with fewer than four events the last two are omitted.
    """
    area = get_polygon_area(polygon)
    tables = get_pair_tables(points, times, polygon, time_limits)
    spatial_k, temporal_k, k = get_space_time_k(tables, area, time_limits, distance_breaks, time_breaks)
    d = k - np.outer(spatial_k, temporal_k)
    analysis = {
        'tables': tables,
        'spatialK': spatial_k,
        'temporalK': temporal_k,
        'k': k,
        'd': d,
        'statistic': float(d.sum()),
    }
    if len(times) >= 4:
        standard_error = get_standard_error(tables, area, time_limits, distance_breaks, time_breaks)
        analysis['standardError'] = standard_error
        analysis['standardisedResidual'] = d / standard_error
    return analysis


def get_boundary_margin(tables, time_limits, times, distance_breaks, time_breaks):
    """Smallest gap of a pair to a class boundary or to the temporal edge rule."""
    distances, lags, _, _ = tables
    count = distances.shape[0]
    off_diagonal = ~np.eye(count, dtype=bool)
    margin = min(np.abs(distances[off_diagonal][:, None] - distance_breaks[None, :]).min(),
                 np.abs(lags[off_diagonal][:, None] - time_breaks[None, :]).min())
    times = np.asarray(times)
    radius = np.minimum(times - min(time_limits), max(time_limits) - times)
    return float(min(margin, np.abs(lags - radius[:, None])[off_diagonal].min()))


def check_weights_against_shapely(points, tables, polygon, maximum_distance):
    """Largest difference of the circle fraction from a clipped 16,384-vertex shapely circle."""
    distances, _, spatial_weights, _ = tables
    region = shapely.Polygon(polygon)
    theta = np.linspace(0, 2 * math.pi, 16385)
    worst = 0.0
    for first, second in zip(*np.nonzero(distances > 0)):
        radius = distances[first, second]
        if radius >= maximum_distance:
            continue
        circle = shapely.LineString(np.column_stack([points[first][0] + radius * np.cos(theta),
                                                     points[first][1] + radius * np.sin(theta)]))
        fraction = circle.intersection(region).length / circle.length
        worst = max(worst, abs(fraction - 1 / spatial_weights[first, second]))
    return worst


def check_variance_by_enumeration(generator):
    """`stsecal` squared against the variance over all relabellings of six events."""
    polygon = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    points = [(float(x), float(y)) for x, y in generator.uniform([0.3, 0.3], [3.7, 2.7], size=(6, 2))]
    times = [float(value) for value in generator.uniform(0.5, 9.5, size=6)]
    time_limits = [0.0, 10.0]
    distance_breaks = np.array([1.0, 2.5])
    time_breaks = np.array([2.0, 5.0])
    area = get_polygon_area(polygon)
    tables = get_pair_tables(points, times, polygon, time_limits)
    distances, lags, spatial_weights, temporal_weights = tables
    spatial_pairs = (spatial_weights + spatial_weights.T) / 2
    temporal_pairs = (temporal_weights + temporal_weights.T) / 2
    errors = get_standard_error(tables, area, time_limits, distance_breaks, time_breaks)
    off_diagonal = ~np.eye(6, dtype=bool)
    worst = 0.0
    for distance_index, distance_break in enumerate(distance_breaks):
        spatial = np.where(off_diagonal & (distances <= distance_break), spatial_pairs, 0.0)
        for time_index, time_break in enumerate(time_breaks):
            temporal = np.where(off_diagonal & (lags <= time_break), temporal_pairs, 0.0)
            values = []
            for order in itertools.permutations(range(6)):
                index = np.array(order)
                values.append((spatial * temporal[np.ix_(index, index)]).sum() * area * 10.0 / 30)
            variance = float(np.var(values))
            assert variance > 0
            worst = max(worst, abs(errors[distance_index, time_index] ** 2 - variance) / variance)
    return worst


def create_scene(seed):
    """Clustered events in the rectangle, rounded to float32 as the GPU receives them."""
    generator = np.random.default_rng(seed)
    centers = generator.uniform([1.5, 1.5, 3.0], [8.5, 6.5, 17.0], size=(7, 3))
    rows = []
    while len(rows) < EVENT_COUNT:
        if generator.random() < 0.65:
            center = centers[generator.integers(len(centers))]
            candidate = center + generator.normal(0, [0.55, 0.55, 1.4])
        else:
            candidate = generator.uniform([0, 0, 0], [10, 8, 20])
        if 0.05 < candidate[0] < 9.95 and 0.05 < candidate[1] < 7.95 and 0.05 < candidate[2] < 19.95:
            rows.append(candidate)
    rows = np.asarray(rows, dtype=np.float32).astype(np.float64)
    return [(float(x), float(y)) for x, y in rows[:, :2]], [float(value) for value in rows[:, 2]]


def create_integer_time_scene(seed):
    """Events anywhere in the rectangle with whole-day times, the two time limits included."""
    generator = np.random.default_rng(seed)
    positions = generator.uniform([0.05, 0.05], [9.95, 7.95], size=(INTEGER_EVENT_COUNT, 2))
    positions = positions.astype(np.float32).astype(np.float64)
    times = generator.integers(0, 31, size=INTEGER_EVENT_COUNT)
    return [(float(x), float(y)) for x, y in positions], [float(value) for value in times]


def get_integer_time_section(rectangle, describe):
    """stkhat and the stmctest statistic on whole-day times, for the lag grids above."""
    distance_breaks = (INTEGER_MAXIMUM_DISTANCE * np.arange(1, INTEGER_DISTANCE_CLASS_COUNT + 1)
                       / INTEGER_DISTANCE_CLASS_COUNT)
    seed = 20261150
    while True:
        points, times = create_integer_time_scene(seed)
        tables = get_pair_tables(points, times, rectangle, INTEGER_TIME_LIMITS)
        off_diagonal = ~np.eye(INTEGER_EVENT_COUNT, dtype=bool)
        margin = np.abs(tables[0][off_diagonal][:, None] - distance_breaks[None, :]).min()
        # Both time limits must hold an event (edge radius 0) and no distance may sit on a break.
        if margin > BOUNDARY_MARGIN and 0.0 in times and 30.0 in times:
            break
        seed += 1
    generator = np.random.default_rng(seed + 1)
    orders = [generator.permutation(INTEGER_EVENT_COUNT) for _ in range(INTEGER_RELABELLING_COUNT)]
    labellings = [times] + [[times[index] for index in order] for order in orders]
    cases = []
    for maximum_time_lag, time_class_count in INTEGER_LAG_GRIDS:
        time_breaks = maximum_time_lag * np.arange(1, time_class_count + 1) / time_class_count
        analyses = [get_analysis(points, labelling, rectangle, INTEGER_TIME_LIMITS, distance_breaks,
                                 time_breaks) for labelling in labellings]
        lags = analyses[0]['tables'][1]
        observed_times = np.asarray(times)
        radius = np.minimum(observed_times - INTEGER_TIME_LIMITS[0],
                            INTEGER_TIME_LIMITS[1] - observed_times)
        in_range = off_diagonal & (lags <= maximum_time_lag)
        cases.append({
            'maximumTimeLag': maximum_time_lag,
            'timeClassCount': time_class_count,
            'timeBreaks': to_list(time_breaks),
            # Ordered pairs of the observed labelling on which a tie rule acts.
            'edgeTieCount': int((in_range & (lags == radius[:, None])).sum()),
            'maximumLagTieCount': int((off_diagonal & (lags == maximum_time_lag)).sum()),
            'interiorBreakTieCount': int((off_diagonal[:, :, None]
                                          & (lags[:, :, None] == time_breaks[None, None, :-1])).sum()),
            'labellings': [describe(analysis) for analysis in analyses],
        })
        assert cases[-1]['edgeTieCount'] > 0
    assert cases[1]['maximumLagTieCount'] > 0 and cases[2]['interiorBreakTieCount'] > 0
    return {
        'seed': seed,
        'timeLimits': INTEGER_TIME_LIMITS,
        'maximumDistance': INTEGER_MAXIMUM_DISTANCE,
        'distanceClassCount': INTEGER_DISTANCE_CLASS_COUNT,
        'distanceBreaks': to_list(distance_breaks),
        'positions': [coordinate for point in points for coordinate in point],
        'labellingTimes': labellings,
        'cases': cases,
        'distanceBoundaryMargin': float(margin),
    }


def to_list(values):
    return np.asarray(values, dtype=np.float64).tolist()


def main():
    rectangle = [(BOUNDS[0], BOUNDS[1]), (BOUNDS[2], BOUNDS[1]), (BOUNDS[2], BOUNDS[3]),
                 (BOUNDS[0], BOUNDS[3])]
    distance_breaks = MAXIMUM_DISTANCE * np.arange(1, DISTANCE_CLASS_COUNT + 1) / DISTANCE_CLASS_COUNT
    time_breaks = MAXIMUM_TIME_LAG * np.arange(1, TIME_CLASS_COUNT + 1) / TIME_CLASS_COUNT

    # The first seed whose observed and relabelled patterns keep every pair off the boundaries.
    seed = 20261111
    while True:
        points, times = create_scene(seed)
        generator = np.random.default_rng(seed + 1)
        orders = [generator.permutation(EVENT_COUNT) for _ in range(RELABELLING_COUNT)]
        labellings = [times] + [[times[index] for index in order] for order in orders]
        analyses = [get_analysis(points, labelling, rectangle, TIME_LIMITS, distance_breaks, time_breaks)
                    for labelling in labellings]
        margin = min(get_boundary_margin(analysis['tables'], TIME_LIMITS, labelling,
                                         distance_breaks, time_breaks)
                     for analysis, labelling in zip(analyses, labellings))
        if margin > BOUNDARY_MARGIN:
            break
        seed += 2
    observed = analyses[0]
    assert not np.any(observed['tables'][0] == distance_breaks[-1])

    shapely_rectangle = check_weights_against_shapely(points, observed['tables'], rectangle,
                                                      MAXIMUM_DISTANCE)
    # An L-shaped (concave) window, to check the polygon case of the weight as well.
    l_shape = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (5.0, 4.0), (5.0, 8.0), (0.0, 8.0)]
    l_points = [point for point in points if is_inside_polygon(point[0], point[1], l_shape)
                and not (4.9 < point[0] < 5.1 or 3.9 < point[1] < 4.1)]
    l_tables = get_pair_tables(l_points, times[:len(l_points)], l_shape, TIME_LIMITS)
    shapely_l_shape = check_weights_against_shapely(l_points, l_tables, l_shape, MAXIMUM_DISTANCE)
    variance_error = check_variance_by_enumeration(np.random.default_rng(7))
    assert shapely_rectangle < 2e-5, shapely_rectangle
    assert shapely_l_shape < 2e-5, shapely_l_shape
    assert variance_error < 1e-10, variance_error

    # stmctest: the same statistic after `sample(times)`, here numpy permutations.
    generator = np.random.default_rng(seed + 2)
    area = get_polygon_area(rectangle)
    monte_carlo = []
    for _ in range(MONTE_CARLO_COUNT):
        order = generator.permutation(EVENT_COUNT)
        permuted = [times[index] for index in order]
        tables = get_pair_tables(points, permuted, rectangle, TIME_LIMITS)
        # Locations are fixed: reuse the spatial weights, which do not depend on the times.
        tables = (observed['tables'][0], tables[1], observed['tables'][2], tables[3])
        spatial_k, temporal_k, k = get_space_time_k(tables, area, TIME_LIMITS, distance_breaks, time_breaks)
        monte_carlo.append(float((k - np.outer(spatial_k, temporal_k)).sum()))
    greater = sum(1 for value in monte_carlo if value >= observed['statistic'])

    # Two events whose lag exactly reaches the lower time limit: splancs counts the first double.
    tie_points = [(2.0, 2.0), (3.0, 2.0)]
    tie = get_analysis(tie_points, [3.0, 6.0], rectangle, [0.0, 10.0], np.array([2.0]), np.array([4.0]))

    def describe(analysis):
        description = {
            'spatialK': to_list(analysis['spatialK']),
            'temporalK': to_list(analysis['temporalK']),
            'k': to_list(analysis['k'].ravel()),
            'd': to_list(analysis['d'].ravel()),
            'statistic': analysis['statistic'],
        }
        if 'standardError' in analysis:
            description['standardError'] = to_list(analysis['standardError'].ravel())
            description['standardisedResidual'] = to_list(analysis['standardisedResidual'].ravel())
        return description

    fixture = {
        'generator': 'test/oracles/space-time-diggle-splancs-port.py',
        'provenance': ('A port of the splancs 2.01-45 source (stkhat, stsecal, stmctest, stdiagn) '
                       'to NumPy, not a run of R.'),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'shapely': shapely.__version__, 'splancsSource': '2.01-45'},
        'seed': seed,
        'bounds': BOUNDS,
        'timeLimits': TIME_LIMITS,
        'maximumDistance': MAXIMUM_DISTANCE,
        'distanceClassCount': DISTANCE_CLASS_COUNT,
        'distanceBreaks': to_list(distance_breaks),
        'maximumTimeLag': MAXIMUM_TIME_LAG,
        'timeClassCount': TIME_CLASS_COUNT,
        'timeBreaks': to_list(time_breaks),
        'positions': [coordinate for point in points for coordinate in point],
        'times': times,
        'layout': 'k, d, standardError and standardisedResidual are distance-major: [distance][time]',
        'observed': describe(observed),
        'relabellings': [{'times': labelling, **describe(analysis)}
                         for labelling, analysis in zip(labellings[1:], analyses[1:])],
        'monteCarlo': {
            'permutationCount': MONTE_CARLO_COUNT,
            'statistics': monte_carlo,
            'mean': float(np.mean(monte_carlo)),
            'standardDeviation': float(np.std(monte_carlo, ddof=1)),
            'greaterCount': greater,
            'pseudoPGreater': (greater + 1) / (MONTE_CARLO_COUNT + 1),
        },
        'integerTimes': get_integer_time_section(rectangle, describe),
        'temporalTie': {
            'positions': [coordinate for point in tie_points for coordinate in point],
            'times': [3.0, 6.0],
            'timeLimits': [0.0, 10.0],
            'distanceBreaks': [2.0],
            'timeBreaks': [4.0],
            **describe(tie),
        },
        'checks': {
            'boundaryMargin': margin,
            'shapelyCircleFractionMaximumDifferenceRectangle': shapely_rectangle,
            'shapelyCircleFractionMaximumDifferenceLShape': shapely_l_shape,
            'lShapeEventCount': len(l_points),
            'stsecalVarianceMaximumRelativeErrorAgainstEnumeration': variance_error,
        },
    }
    with open(OUTPUT, 'w', encoding='utf-8') as file:
        json.dump(fixture, file, separators=(',', ':'))
        file.write('\n')
    print(f'wrote {OUTPUT}: seed {seed}, statistic {observed["statistic"]:.6f}, '
          f'p {fixture["monteCarlo"]["pseudoPGreater"]}, checks {fixture["checks"]}')


if __name__ == '__main__':
    main()
