# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""RTS-05 reference fixtures: binary64 NumPy/SciPy implementations of the published definitions of
compositing and phenology metrics.

hdmedians (Apache-2.0) is the named reference for the medoid and geometric median. It is not
installed here, so these are NumPy implementations of its definitions written from the
documentation: `nanmedoid` is the observation with the smallest sum of Euclidean distances to the
other observations that have no NaN band; the geometric median is Weiszfeld's recursion started at
the mean, returning a sole retained observation exactly and updating general stacks
by weighted mean displacement. Scalar medoid ties use the central order-statistic
interval and earliest acquisition. These identity/tie rules are independently derived,
not outputs attributed to hdmedians. The geometric median is additionally checked against
the true minimiser found by scipy.optimize.minimize. The best-available-pixel score is the
Griffiths et al. 2013 weighted sum
of a Gaussian day-of-year score, a linear year score and a linear cloud-distance score. Phenology
metrics are threshold crossings of the amplitude on a given smoothed curve with linear
interpolation (Copernicus MR-VPP definition); no TIMESAT or phenofit code was run.
Output: fixtures/rts05-compositing-numpy.json.
"""
import json
import pathlib
import platform
import sys

import numpy as np
import scipy
from scipy.optimize import minimize

OUTPUT = pathlib.Path(__file__).resolve().parent / 'fixtures' / 'rts05-compositing-numpy.json'
SECONDS_PER_YEAR = 31556952.0
rng = np.random.default_rng(20261109)


def nullable(array):
    return [None if not np.isfinite(value) else float(value) for value in np.ravel(array)]


def nanmedoid(observations, valid):
    """Index of the valid observation minimising the sum of distances to the other valid ones."""
    keep = [i for i in range(len(observations)) if valid[i] and np.all(np.isfinite(observations[i]))]
    if not keep:
        return -1
    varying = np.flatnonzero(np.any(observations[keep] != observations[keep[0]], axis=0))
    if len(varying) == 0:
        return keep[0]
    if len(varying) == 1:
        # In one dimension all points in the closed central order-statistic
        # interval minimize the absolute-distance sum. Decide this exact tie
        # without comparing differently rounded sums; preserve acquisition order.
        band = varying[0]
        ordered = sorted(observations[index, band] for index in keep)
        lower, upper = ordered[(len(ordered) - 1) // 2], ordered[len(ordered) // 2]
        return next(index for index in keep if lower <= observations[index, band] <= upper)
    sums = []
    for i in keep:
        sums.append(sum(np.linalg.norm(observations[i] - observations[j]) for j in keep if j != i))
    return keep[int(np.argmin(sums))]  # argmin returns the first minimum: ties go to the earliest slice


def weiszfeld(points, iterations, epsilon):
    if len(points) == 1:
        return points[0].copy(), 0.0
    estimate = points[0] + (points - points[0]).mean(axis=0)
    step = 0.0
    for _ in range(iterations):
        distance = np.maximum(np.linalg.norm(points - estimate, axis=1), epsilon)
        weight = 1.0 / distance
        displacement = points - estimate
        updated = estimate + (weight[:, None] * displacement).sum(axis=0) / weight.sum()
        step = float(np.linalg.norm(updated - estimate))
        estimate = updated
    return estimate, step


def objective(points, location):
    return float(np.linalg.norm(points - location, axis=1).sum())


def medoid_fixture():
    slices, bands = 7, 3
    cells = []
    for case in range(6):
        observations = rng.uniform(0.0, 1.0, (slices, bands)).round(4)
        valid = np.ones(slices, dtype=int)
        if case == 1:
            observations[2, 1] = np.nan
        if case == 2:
            valid[[0, 3]] = 0
        if case == 3:
            valid[:] = 0
        if case == 4:  # two valid observations are equidistant: the earliest wins
            valid[:] = 0
            valid[[2, 5]] = 1
        if case == 5:  # one valid observation
            valid[:] = 0
            valid[4] = 1
        cells.append({'observations': observations, 'valid': valid})
    out = []
    for cell in cells:
        out.append(
            {
                'values': nullable(cell['observations']),
                'validity': cell['valid'].tolist(),
                'expectedSlice': nanmedoid(cell['observations'], cell['valid']),
            }
        )
    return {'sliceCount': slices, 'bandCount': bands, 'cells': out}


def scalar_medoid_fixture():
    # Independent scalar theorem: minimizers of sum |x-a| form the closed
    # central order-statistic interval. These cases are not hdmedians output.
    cells = []
    for values, valid in [([.1, .2, .3, .4], [1, 1, 1, 1]),
                          ([.1, .3, .2, .4], [1, 1, 1, 1]),
                          ([.4, .2, .2, .1], [1, 1, 1, 1]),
                          ([.2, .1, .3, .2], [1, 1, 1, 1]),
                          ([.2, .1, .3, .4], [0, 1, 1, 1])]:
        observations = np.array(values, dtype=float)[:, None]
        selected = nanmedoid(observations, valid)
        cells.append({'values': values, 'validity': valid,
                      'expectedSlice': selected, 'expectedValue': values[selected]})
    return {'source': 'independent central-order-statistic interval derivation; decimal inputs',
            'sliceCount': 4, 'bandCount': 1, 'cells': cells}


def geometric_median_fixture():
    slices, bands, iterations, epsilon = 12, 4, 32, 1e-5
    out = []
    for case in range(5):
        points = rng.normal(0.3, 0.15, (slices, bands)).round(4)
        if case == 1:  # heavy outliers
            points[:3] += 2.0
        valid = np.ones(slices, dtype=int)
        if case == 2:
            valid[[1, 7]] = 0
        if case == 3:
            points[4, 2] = np.nan
        if case == 4:
            valid[:] = 0
        usable = points[(valid == 1) & np.all(np.isfinite(points), axis=1)]
        if len(usable):
            estimate, step = weiszfeld(usable, iterations, epsilon)
            start = usable.mean(axis=0)
            optimum = minimize(lambda y: objective(usable, y), start, method='Nelder-Mead',
                               options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 20000, 'maxfev': 40000})
            expected, last_step, optimum_objective = estimate.tolist(), step, float(optimum.fun)
            early, early_step = weiszfeld(usable, 3, epsilon)  # few iterations: not converged
            early, early_step = early.tolist(), early_step
        else:
            expected, last_step, optimum_objective, early, early_step = None, None, None, None, None
        out.append(
            {
                'values': nullable(points),
                'validity': valid.tolist(),
                'expected': expected,
                'expectedLastStep': last_step,
                'optimumObjective': optimum_objective,
                'expectedAfterThreeIterations': early,
                'expectedLastStepAfterThreeIterations': early_step,
            }
        )
    return {'sliceCount': slices, 'bandCount': bands, 'iterationCount': iterations,
            'distanceEpsilon': epsilon, 'cells': out}


def geometric_median_identity_fixture():
    cells = []
    epsilon = float(np.float32(1e-5))
    for value in [7.0, 456787.5, .1]:
        for valid in [[0, 1, 0], [1, 1, 1]]:
            points = np.array([[value], [value], [value]])
            if valid == [0, 1, 0]:
                points[0, 0], points[2, 0] = value + 13, value - 23
            usable = points[np.array(valid) != 0]
            estimate, step = weiszfeld(usable, 1, epsilon)
            cells.append({'values': nullable(points), 'validity': valid,
                          'expected': estimate.tolist(), 'expectedLastStep': step})
    return {'source': 'independent identity: one retained point or identical points minimizes sum of distances exactly',
            'sliceCount': 3, 'bandCount': 1, 'iterationCount': 1,
            'distanceEpsilon': epsilon, 'cells': cells}


def best_available_pixel_fixture():
    slices, bands = 10, 2
    days = np.array([20, 75, 130, 170, 200, 215, 260, 300, 340, 362], dtype=float)
    years = np.array([2021, 2021, 2022, 2022, 2022, 2023, 2023, 2023, 2024, 2024], dtype=float)
    params = {
        'targetDayOfYear': 190.0, 'dayOfYearSigma': 40.0, 'dayOfYearWeight': 0.5,
        'targetYear': 2023.0, 'maximumYearDistance': 3.0, 'yearWeight': 0.2,
        'minimumCloudDistance': 10.0, 'maximumCloudDistance': 100.0, 'cloudWeight': 0.3,
    }
    out = []
    for case in range(5):
        values = rng.uniform(0, 1, (slices, bands)).round(3)
        cloud = rng.uniform(0, 140, slices).round(1)
        valid = np.ones(slices, dtype=int)
        if case == 1:
            valid[[4, 5]] = 0
        if case == 2:
            cloud[:] = np.inf
        if case == 3:
            valid[:] = 0
        if case == 4:
            cloud[[1, 2]] = np.nan
            values[6, 0] = np.nan
        scores = np.full(slices, -np.inf)
        for s in range(slices):
            if not valid[s] or not np.all(np.isfinite(values[s])):
                continue
            delta = abs(days[s] - params['targetDayOfYear'])
            delta = min(delta, 365.2425 - delta)
            day_score = np.exp(-0.5 * (delta / params['dayOfYearSigma']) ** 2)
            year_score = max(0.0, 1 - abs(years[s] - params['targetYear']) / params['maximumYearDistance'])
            c = cloud[s]
            lo, hi = params['minimumCloudDistance'], params['maximumCloudDistance']
            cloud_score = 0.0 if np.isnan(c) else float(np.clip((c - lo) / (hi - lo), 0, 1))
            scores[s] = (params['dayOfYearWeight'] * day_score + params['yearWeight'] * year_score
                         + params['cloudWeight'] * cloud_score)
        best = int(np.argmax(scores)) if np.isfinite(scores).any() else -1
        out.append({
            'values': nullable(values),
            'validity': valid.tolist(),
            # JSON has no infinity: the spec restores 'inf' strings.
            'cloudDistance': ['inf' if np.isinf(c) else (None if np.isnan(c) else float(c)) for c in cloud],
            'expectedSlice': best,
            'expectedScore': None if best < 0 else float(scores[best]),
        })
    return {'sliceCount': slices, 'bandCount': bands, 'dayOfYear': days.tolist(),
            'year': years.tolist(), 'parameters': params, 'cells': out}


def phenology(times, curve, start_fraction, end_fraction, minimum_amplitude, second_fraction, maximum_seasons):
    """Threshold-crossing seasons of one curve; returns (status, seasons sorted by peak time)."""
    if not np.all(np.isfinite(curve)):
        return 1, []
    base, top = curve.min(), curve.max()
    amplitude = top - base
    if not amplitude > 0 or amplitude < minimum_amplitude:
        return 2, []
    start_level = base + start_fraction * amplitude
    end_level = base + end_fraction * amplitude
    seasons, excluded = [], None
    for number in range(maximum_seasons):
        allowed = np.ones(len(curve), dtype=bool)
        if excluded:
            allowed[excluded[0]:excluded[1] + 1] = False
        if not allowed.any():
            break
        peak = int(np.argmax(np.where(allowed, curve, -np.inf)))
        if number > 0 and curve[peak] < base + second_fraction * amplitude:
            break
        lower, upper = 0, len(curve) - 1
        if excluded:
            if peak < excluded[0]:
                upper = excluded[0] - 1
            else:
                lower = excluded[1] + 1
        flags = 0
        below = np.flatnonzero(curve[lower:peak] < start_level)
        if len(below):
            j = lower + below[-1]
            start_time = times[j] + (start_level - curve[j]) / (curve[j + 1] - curve[j]) * (times[j + 1] - times[j])
            start_value, left = start_level, j + 1
        else:
            start_time, start_value, left = times[lower], curve[lower], lower
            flags |= 1 if lower == 0 else 4
        below = np.flatnonzero(curve[peak + 1:upper + 1] < end_level)
        if len(below):
            k = peak + 1 + below[0]
            end_time = times[k - 1] + (curve[k - 1] - end_level) / (curve[k - 1] - curve[k]) * (times[k] - times[k - 1])
            end_value, right = end_level, k - 1
        else:
            end_time, end_value, right = times[upper], curve[upper], upper
            flags |= 2 if upper == len(curve) - 1 else 8
        knots_t = np.concatenate([[start_time], times[left:right + 1], [end_time]])
        knots_z = np.concatenate([[start_value], curve[left:right + 1], [end_value]]) - base
        integral = float(np.sum(0.5 * (knots_z[1:] + knots_z[:-1]) * np.diff(knots_t)))
        peak_time = times[peak]
        seasons.append({
            'startTime': start_time, 'endTime': end_time, 'peakTime': peak_time,
            'length': end_time - start_time, 'peakValue': curve[peak], 'baseValue': base,
            'amplitude': curve[peak] - base,
            'leftSlope': (curve[peak] - start_value) / (peak_time - start_time) if peak_time > start_time else None,
            'rightSlope': (end_value - curve[peak]) / (end_time - peak_time) if end_time > peak_time else None,
            'integral': integral, 'flags': flags,
        })
        excluded = (left, right)
    seasons.sort(key=lambda season: season['peakTime'])
    return 0, seasons


def phenology_fixture():
    count = 48
    # Irregular acquisition seconds (roughly 8-day cadence with jitter), centred like the date axis.
    spacing = np.cumsum(rng.integers(6 * 86400, 11 * 86400, count))
    seconds = (1_600_000_000 + spacing).astype(np.int64)
    midpoint = seconds[0] + (seconds[-1] - seconds[0]) // 2
    years = (seconds - midpoint) / SECONDS_PER_YEAR
    unit = (years - years[0]) / (years[-1] - years[0])
    bump = lambda centre, width: np.exp(-0.5 * ((unit - centre) / width) ** 2)
    curves = {
        'one-season': 0.1 + 0.7 * bump(0.5, 0.12),
        'two-seasons': 0.1 + 0.6 * bump(0.28, 0.07) + 0.45 * bump(0.72, 0.08),
        'two-seasons-reversed-heights': 0.1 + 0.4 * bump(0.25, 0.07) + 0.7 * bump(0.7, 0.07),
        'high-at-series-start': 0.1 + 0.8 * bump(0.02, 0.15),
        'high-at-series-end': 0.1 + 0.8 * bump(0.98, 0.15),
        'flat': np.full(count, 0.3),
        'nan-sample': np.where(np.arange(count) == 9, np.nan, 0.1 + 0.7 * bump(0.5, 0.12)),
        'weak-second-bump': 0.1 + 0.7 * bump(0.3, 0.08) + 0.12 * bump(0.8, 0.06),
    }
    settings = {'startFraction': 0.25, 'endFraction': 0.15, 'minimumAmplitude': 0.05,
                'secondSeasonMinimumFraction': 0.4}
    out = []
    for name, curve in curves.items():
        curve = curve.astype(np.float32).astype(np.float64)  # the GPU reads float32 input
        status, seasons = phenology(years, curve, settings['startFraction'], settings['endFraction'],
                                    settings['minimumAmplitude'], settings['secondSeasonMinimumFraction'], 2)
        out.append({'name': name, 'curve': nullable(curve), 'expectedStatus': status,
                    'expectedSeasons': seasons})
    return {'sliceCount': count, 'timesSeconds': seconds.tolist(), 'centredYears': years.tolist(),
            'settings': settings, 'cells': out}


def verify_reference_boundaries():
    """Read-only exact controls; also run before every pinned fixture regeneration.

    Run this file with --check-reference in the pinned Python environment to
    exercise the actual calculations without writing any fixture.
    """
    for values, selected in [([.1,.2,.3,.4], 1), ([.1,.3,.2,.4], 1),
                             ([.4,.2,.2,.1], 1), ([.2,.1,.3,.2], 0)]:
        for dtype in [float, np.float32]:
            scalar = np.array(values, dtype=dtype).astype(float)
            for observations in [scalar[:, None], np.column_stack((np.ones(4), scalar)),
                                 np.column_stack((scalar, np.ones(4)))]:
                actual = nanmedoid(observations, [1,1,1,1])
                assert actual == selected, ('TWN-02 earliest interval medoid', values, actual)
                np.testing.assert_array_equal(observations[actual], observations[selected])
    assert nanmedoid(np.ones((4, 2)), [0,1,1,1]) == 1
    for value in [7.0, 456787.5, .1]:
        for count in [1, 3]:
            estimate, step = weiszfeld(np.full((count, 1), value), 1, float(np.float32(1e-5)))
            np.testing.assert_array_equal(estimate, [value])
            assert step == 0, ('TWN-06 exact identity', value, count, step)
    identity = geometric_median_identity_fixture()
    assert len(identity['cells']) == 6
    for cell, value in zip(identity['cells'], [7.,7.,456787.5,456787.5,.1,.1]):
        assert cell['expected'] == [value] and cell['expectedLastStep'] == 0
        if cell['validity'] == [0,1,0]:
            assert cell['values'][0] != value and cell['values'][2] != value


verify_reference_boundaries()
if '--check-reference' in sys.argv:
    print('compositing reference boundaries: passed (read-only)')
    raise SystemExit(0)

fixture = {
    'source': 'binary64 NumPy/SciPy implementation of the published definitions; hdmedians, '
              'phenofit and TIMESAT were not run',
    'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
    'medoid': medoid_fixture(),
    'geometricMedian': geometric_median_fixture(),
    'bestAvailablePixel': best_available_pixel_fixture(),
    'phenology': phenology_fixture(),
    'scalarMedoid': scalar_medoid_fixture(),
    'geometricMedianIdentity': geometric_median_identity_fixture(),
}
OUTPUT.write_text(json.dumps(fixture, indent=1, default=lambda value: float(value)) + '\n')
print('wrote', OUTPUT)
