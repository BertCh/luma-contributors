# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""RTS-04 reference fixture: pyxccd `cold_detect` on synthetic Landsat-like series with known breaks.

pyxccd 1.1.0 (Apache-2.0) is the reference implementation of COLD (Zhu et al. 2020). Each case is a
seven-band integer series (blue, green, red, NIR, SWIR1, SWIR2, thermal, in pyxccd's input order and
0-10000 scaling), ordinal dates and a cfmask QA band. `cold_detect` is run twice per case: with its
default LASSO penalty (`lam=20`) and with `lam=0`, which pyxccd documents as ordinary least squares.

Nothing here is computed from the GPU kernel or from `rts04-change-monitoring-numpy.py`. Besides
pyxccd's records the fixture holds only what a caller of `GPURasterCOLDMonitor` must supply to mirror
pyxccd's settings, each by an independent definition:

- `changeThreshold`, `outlierThreshold`: chi-squared quantiles at 0.99 and `1 - 1e-5` with 5 degrees of
  freedom. pyxccd scores the five bands green to SWIR2 (`lasso_blist`, `NUM_LASSO_BANDS` 5) and its
  `T_MAX_CG` is 30.8562, the `1 - 1e-5` quantile.
- `minimumRMSE`: the median absolute difference of successive clear observations per band, which is
  pyxccd's variogram floor (`adjust_median_variogram`, option 1).
- `period`: 365.25 / 365.2425, so the harmonics of the monitor (years of 365.2425 days) have pyxccd's
  angular frequency `2 pi / 365.25` per day.

pyxccd reports coefficients as `[intercept at ordinal day 0, slope per day times 10000, cos, sin, ...]`
on absolute ordinal days. `coefficients` in the fixture is the same model in the monitor's convention
(intercept at the midpoint of the date axis, slope per year of 365.2425 days, harmonics of days after
the midpoint); `raw` keeps pyxccd's own numbers.

A second fixture sizes the agreement on break dates. Eight series cannot: whether pyxccd and the
monitor confirm the same break depends on how far the break stands above the change threshold. The
break-size study draws 160 series (one break of a quarter, a half and the full disturbance, on two
revisit patterns: 80, 40 and 40 series) and keeps pyxccd's segment starts and confirmed breaks for each.

Run with the interpreter that has pyxccd (CPython 3.12, numpy 2.5.4, scipy 1.18.1, pyxccd 1.1.0):
`python modules/analysis/test/oracles/rts04-cold-pyxccd.py`.
Output: fixtures/rts04-cold-pyxccd.json and fixtures/rts04-cold-pyxccd-break-sizes.json.
"""
import json
import math
import pathlib
import platform

import numpy as np
import pyxccd
import scipy
from pyxccd import cold_detect
from scipy.stats import chi2

OUTPUT = pathlib.Path(__file__).resolve().parent / 'fixtures' / 'rts04-cold-pyxccd.json'
PYXCCD_DAYS_PER_YEAR = 365.25
MONITOR_DAYS_PER_YEAR = 365.2425
ANGULAR_FREQUENCY = 2.0 * math.pi / PYXCCD_DAYS_PER_YEAR
SLOPE_SCALE = 10000.0
# pyxccd's detection bands (`lasso_blist` in cold.c): green, red, NIR, SWIR1, SWIR2.
DETECTION_BANDS = [1, 2, 3, 4, 5]
CHANGE_PROBABILITY = 0.99
OUTLIER_PROBABILITY = 1.0 - 1e-5

# Per band: level, slope per year, annual amplitude, phase. Blue, green, red, NIR, SWIR1, SWIR2, thermal.
BANDS = [
    (450, 2, 60, 0.3),
    (700, 3, 90, 0.5),
    (600, -2, 110, 0.9),
    (3200, 10, 700, 0.4),
    (1900, -4, 350, 1.1),
    (1100, 3, 220, 1.3),
    (2900, 1, 120, 0.2)
]
DISTURBANCE = [150, 250, 400, -900, 500, 400, 30]
CLOUD = [2500, 2500, 2600, 1200, 900, 700, -500]
NOISE = 40.0
FIRST_ORDINAL = 730500


def make_series(seed, count, breaks, revisit='uniform'):
    """A trend, an annual harmonic, Gaussian noise and step changes on irregular dates: an 8 to 24
    day revisit (`uniform`), or runs of nine dense (3 to 8 days) then nine sparse (24 to 55 days)
    revisits (`blocks`), as cloud seasons give."""
    generator = np.random.default_rng(seed)
    if revisit == 'uniform':
        gaps = generator.integers(8, 25, count - 1)
    else:
        dense = generator.integers(3, 9, count - 1)
        sparse = generator.integers(24, 56, count - 1)
        gaps = np.where((np.arange(count - 1) // 9) % 2 == 0, dense, sparse)
    dates = FIRST_ORDINAL + np.concatenate([[0], np.cumsum(gaps)])
    values = np.zeros((count, len(BANDS)))
    for band, (level, slope, amplitude, phase) in enumerate(BANDS):
        values[:, band] = (
            level
            + slope * (dates - dates[0]) / PYXCCD_DAYS_PER_YEAR
            + amplitude * np.cos(ANGULAR_FREQUENCY * dates - phase)
            + NOISE * generator.standard_normal(count)
        )
    for index, shift in breaks:
        values[index:] += np.array(shift)
    return dates.astype(np.int64), np.rint(values).astype(np.int64)


def to_monitor_convention(coefficients, midpoint_day):
    """pyxccd `[a, 10000 b, c1, s1, ...]` on ordinal days as the monitor's centred-years model."""
    converted = np.zeros_like(coefficients, dtype=np.float64)
    slope_per_day = coefficients[:, 1].astype(np.float64) / SLOPE_SCALE
    converted[:, 0] = coefficients[:, 0] + slope_per_day * midpoint_day
    converted[:, 1] = slope_per_day * MONITOR_DAYS_PER_YEAR
    for harmonic in (1, 2, 3):
        angle = harmonic * ANGULAR_FREQUENCY * midpoint_day
        cosine = coefficients[:, 2 * harmonic].astype(np.float64)
        sine = coefficients[:, 2 * harmonic + 1].astype(np.float64)
        converted[:, 2 * harmonic] = cosine * math.cos(angle) + sine * math.sin(angle)
        converted[:, 2 * harmonic + 1] = -cosine * math.sin(angle) + sine * math.cos(angle)
    return converted


def run_pyxccd(dates, values, qa, consecutive, penalty):
    records = cold_detect(
        dates, *[values[:, band] for band in range(values.shape[1])], qa, conse=consecutive, lam=penalty
    )
    midpoint_day = (int(dates[0]) + int(dates[-1])) / 2.0

    def to_slice(ordinal):
        if ordinal == 0:
            return None
        index = int(np.searchsorted(dates, ordinal))
        if dates[index] != ordinal:
            raise SystemExit(f'ordinal {ordinal} is not an observation date')
        return index

    segments = []
    for record in records:
        coefficients = to_monitor_convention(record['coefs'], midpoint_day)
        segments.append(
            {
                'startSlice': to_slice(int(record['t_start'])),
                'endSlice': to_slice(int(record['t_end'])),
                'breakSlice': to_slice(int(record['t_break'])),
                'observationCount': int(record['num_obs']),
                'category': int(record['category']),
                'changeProbability': int(record['change_prob']),
                'coefficients': coefficients[DETECTION_BANDS].tolist(),
                'rmse': record['rmse'][DETECTION_BANDS].astype(np.float64).tolist(),
                'magnitude': record['magnitude'][DETECTION_BANDS].astype(np.float64).tolist(),
                'raw': {
                    't_start': int(record['t_start']),
                    't_end': int(record['t_end']),
                    't_break': int(record['t_break']),
                    'coefs': record['coefs'].astype(np.float64).tolist(),
                    'rmse': record['rmse'].astype(np.float64).tolist(),
                    'magnitude': record['magnitude'].astype(np.float64).tolist()
                }
            }
        )
    return segments


def make_case(name, note, dates, values, known_break_slices, qa=None, consecutive=6):
    qa = np.zeros(len(dates), dtype=np.int64) if qa is None else qa
    # pyxccd keeps cfmask clear (0) and water (1).
    clear = qa < 2
    detection = values[clear][:, DETECTION_BANDS].astype(np.float64)
    return {
        'name': name,
        'note': note,
        'knownBreakSlices': known_break_slices,
        'dates': [int(value) for value in dates],
        'values': values.tolist(),
        'qa': [int(value) for value in qa],
        'consecutive': consecutive,
        'minimumRMSE': np.median(np.abs(np.diff(detection, axis=0)), axis=0).tolist(),
        'pyxccd': {
            'lasso20': run_pyxccd(dates, values, qa, consecutive, 20),
            'leastSquares': run_pyxccd(dates, values, qa, consecutive, 0)
        }
    }


STUDY_OUTPUT = OUTPUT.with_name('rts04-cold-pyxccd-break-sizes.json')
STUDY_SLICE_COUNT = 150
STUDY_FIRST_SEED = 5000
STUDY_SEEDS_PER_GROUP = 40
# Series per revisit pattern for each size: the size near the change threshold gets the larger sample.
STUDY_SIZES = {0.25: 40, 0.5: 20, 1.0: 20}
STUDY_REVISITS = ['uniform', 'blocks']


def make_break_size_study():
    """One break of a quarter, a half and the full `DISTURBANCE` at a random slice in [60, 100), each
    series with its own dates and noise. Only pyxccd's segment starts and confirmed breaks are kept,
    and the five scored bands of the input."""
    series = []
    group = 0
    for revisit in STUDY_REVISITS:
        for size, series_count in STUDY_SIZES.items():
            for index in range(series_count):
                seed = STUDY_FIRST_SEED + group * STUDY_SEEDS_PER_GROUP + index
                break_slice = int(np.random.default_rng(seed + 100000).integers(60, 100))
                shift = [size * value for value in DISTURBANCE]
                dates, values = make_series(seed, STUDY_SLICE_COUNT, [(break_slice, shift)], revisit)
                qa = np.zeros(STUDY_SLICE_COUNT, dtype=np.int64)
                detection = values[:, DETECTION_BANDS].astype(np.float64)

                def bounds(penalty):
                    segments = run_pyxccd(dates, values, qa, 6, penalty)
                    return [
                        [segment['startSlice'], segment['breakSlice'] if segment['changeProbability'] == 100 else None]
                        for segment in segments
                    ]

                series.append(
                    {
                        'seed': seed,
                        'revisit': revisit,
                        'size': size,
                        'breakSlice': break_slice,
                        'dates': [int(value) for value in dates],
                        'values': values[:, DETECTION_BANDS].tolist(),
                        'minimumRMSE': np.median(np.abs(np.diff(detection, axis=0)), axis=0).tolist(),
                        'pyxccd': {'leastSquares': bounds(0), 'lasso20': bounds(20)}
                    }
                )
            group += 1
    return series


def break_slices(segments):
    return [segment['breakSlice'] for segment in segments if segment['changeProbability'] == 100]


def main():
    cases = []

    dates, values = make_series(1, 150, [(80, DISTURBANCE)])
    cases.append(make_case('one-break', 'One abrupt disturbance at slice 80.', dates, values, [80]))

    dates, values = make_series(
        2, 160, [(55, [100, 200, 300, -800, 400, 300, 0]), (110, [-80, -150, -250, 700, -350, -250, 0])]
    )
    cases.append(make_case('two-breaks', 'A disturbance at slice 55 and a recovery at slice 110.', dates, values, [55, 110]))

    dates, values = make_series(3, 140, [])
    cases.append(make_case('no-break', 'A stable series.', dates, values, []))

    dates, values = make_series(4, 150, [(90, [40, 60, 100, -220, 130, 100, 0])])
    cases.append(
        make_case(
            'small-break',
            'A disturbance a quarter the size of the others at slice 90. The first window and the window '
            'after the break pass the stability test only when the larger of the two end residuals is used.',
            dates,
            values,
            [90]
        )
    )

    dates, values = make_series(5, 150, [(70, DISTURBANCE)])
    qa = np.zeros(150, dtype=np.int64)
    clouds = np.random.default_rng(55).choice(150, 18, replace=False)
    qa[clouds] = 4
    values[clouds] += np.array([3000, 3000, 3200, 1500, 1000, 800, -600])
    shadows, waters = [3, 40], [11, 100]
    if set(clouds.tolist()) & set(shadows + waters):
        raise SystemExit('flagged-clouds: a shadow or water slice is also a cloud')
    qa[shadows] = 2
    qa[waters] = 1
    cases.append(
        make_case(
            'flagged-clouds',
            'A disturbance at slice 70 with 18 cloud (4) and 2 shadow (2) observations flagged in QA and '
            '2 water (1) observations, which pyxccd keeps.',
            dates,
            values,
            [70],
            qa
        )
    )

    dates, values = make_series(6, 150, [(85, DISTURBANCE)])
    values[6] += np.array(CLOUD)
    cases.append(
        make_case(
            'unflagged-cloud-first-window',
            'A disturbance at slice 85 and an unflagged cloud at slice 6, inside the first window. pyxccd '
            'removes it with Tmask.',
            dates,
            values,
            [85]
        )
    )

    dates, values = make_series(8, 150, [(100, DISTURBANCE)])
    cases.append(make_case('four-consecutive', 'A disturbance at slice 100, confirmed by 4 observations.', dates, values, [100], consecutive=4))

    dates, values = make_series(9, 160, [])
    ramp = np.clip((np.arange(160) - 70) / 30, 0, 1)[:, None] * np.array(DISTURBANCE)
    values = values + np.rint(ramp).astype(np.int64)
    cases.append(
        make_case(
            'gradual-change',
            'A linear transition over slices 70 to 100. The confirmed date depends on the refit schedule, '
            'the temporally adjusted RMSE and LASSO.',
            dates,
            values,
            []
        )
    )

    for case in cases:
        for variant, segments in case['pyxccd'].items():
            if case['knownBreakSlices'] and break_slices(segments) != case['knownBreakSlices']:
                raise SystemExit(f"{case['name']} {variant}: pyxccd does not find the known breaks")
            print(
                case['name'],
                variant,
                [(segment['startSlice'], segment['endSlice'], segment['breakSlice'], segment['changeProbability']) for segment in segments]
            )

    OUTPUT.write_text(
        json.dumps(
            {
                'description': (
                    'pyxccd cold_detect (lam 20 and lam 0) on synthetic seven-band series with known breaks. '
                    'Coefficients are also given in the convention of GPURasterCOLDMonitor; see the generator.'
                ),
                'versions': {
                    'python': platform.python_version(),
                    'numpy': np.__version__,
                    'scipy': scipy.__version__,
                    'pyxccd': pyxccd.__version__
                },
                'bandOrder': ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'],
                'detectionBands': DETECTION_BANDS,
                'period': PYXCCD_DAYS_PER_YEAR / MONITOR_DAYS_PER_YEAR,
                'changeThreshold': float(chi2.ppf(CHANGE_PROBABILITY, len(DETECTION_BANDS))),
                'outlierThreshold': float(chi2.ppf(OUTLIER_PROBABILITY, len(DETECTION_BANDS))),
                'cases': cases
            }
        )
    )

    study = make_break_size_study()
    for revisit in STUDY_REVISITS:
        for size in STUDY_SIZES:
            group = [entry for entry in study if entry['revisit'] == revisit and entry['size'] == size]
            found = sum(
                [bounds[1] for bounds in entry['pyxccd']['leastSquares'] if bounds[1] is not None] == [entry['breakSlice']]
                for entry in group
            )
            print('break-size study', revisit, size, 'pyxccd lam 0 confirms exactly the true break in', found, 'of', len(group))
    STUDY_OUTPUT.write_text(
        json.dumps(
            {
                'description': (
                    'pyxccd cold_detect (lam 0 and lam 20) segment starts and confirmed breaks on 160 synthetic '
                    'series with one break of a quarter, a half and the full disturbance; see the generator.'
                ),
                'versions': {
                    'python': platform.python_version(),
                    'numpy': np.__version__,
                    'scipy': scipy.__version__,
                    'pyxccd': pyxccd.__version__
                },
                'detectionBands': DETECTION_BANDS,
                'period': PYXCCD_DAYS_PER_YEAR / MONITOR_DAYS_PER_YEAR,
                'changeThreshold': float(chi2.ppf(CHANGE_PROBABILITY, len(DETECTION_BANDS))),
                'outlierThreshold': float(chi2.ppf(OUTLIER_PROBABILITY, len(DETECTION_BANDS))),
                'consecutive': 6,
                'noise': NOISE,
                'disturbance': [DISTURBANCE[band] for band in DETECTION_BANDS],
                'series': study
            },
            separators=(',', ':')
        )
    )


if __name__ == '__main__':
    main()
