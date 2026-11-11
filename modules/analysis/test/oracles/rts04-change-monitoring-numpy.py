# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""RTS-04 reference fixtures: binary64 NumPy implementations of irregular-date harmonic regression,
the simplified COLD monitor, and BFAST Monitor.

pyxccd (Apache-2.0) is not run by this generator: it has no wheel for the pinned CPython 3.14. It is
run by `rts04-cold-pyxccd.py` (pyxccd 1.1.0 `cold_detect` on CPython 3.12.13, NumPy 2.5.4, SciPy 1.18.1),
whose fixture is `fixtures/rts04-cold-pyxccd.json`. The stability test of `cold_monitor` below takes the
larger of the two end residuals, which is pyxccd's rule (read in its `cold.c` and confirmed by that fixture).

bfast is GPL and R-only: the BFAST Monitor below is written from the published definition
(Verbesselt, Zeileis and Herold 2012; Leisch, Hornik and Kuan 2000 for the OLS-MOSUM process and
Chu, Hornik and Kuan 1995 / Zeileis 2005 for the boundary), and no R or GPL code was read or copied.

The toggles of `cold_monitor` implement the published COLD elements the simplified monitor omits
(Zhu et al. 2020, RSE 238:111116; Zhu and Woodcock 2014). They are written from those papers, are
not pyxccd, and exist so that each documented omission is shown to change the result on a fixture.

Output: fixtures/rts04-change-monitoring-numpy.json. Run with the pinned set of `requirements.txt`
(CPython 3.14.8, numpy 2.5.4, scipy 1.16.3; SciPy supplies the chi-squared quantiles only, and
scipy 1.18.1 gives the same fixture).
"""
import json
import math
import pathlib
import platform

import numpy as np
import scipy
from scipy.stats import chi2

OUTPUT = pathlib.Path(__file__).resolve().parent / 'fixtures' / 'rts04-change-monitoring-numpy.json'
SECONDS_PER_YEAR = 31556952.0
TWO_PI = 2.0 * math.pi


def centred_years(seconds):
    """Centred fractional years of the irregular axis, the midpoint rounded down as on Int64 words."""
    first, last = int(seconds[0]), int(seconds[-1])
    midpoint = first + (last - first) // 2
    return np.array([(int(second) - midpoint) / SECONDS_PER_YEAR for second in seconds])


def design_matrix(years, column_count, period):
    columns = [np.ones_like(years), years]
    for harmonic in range(1, (column_count - 2) // 2 + 1):
        angle = TWO_PI * harmonic * years / period
        columns += [np.cos(angle), np.sin(angle)]
    return np.stack(columns, axis=1)


def harmonic_regression(y, years, harmonic_count, period):
    usable = np.isfinite(y)
    design = design_matrix(years[usable], 2 + 2 * harmonic_count, period)
    coefficients, *_ = np.linalg.lstsq(design, y[usable], rcond=None)
    residual = y[usable] - design @ coefficients
    return {
        'coefficients': coefficients.tolist(),
        'rmse': float(np.sqrt(np.sum(residual**2) / (usable.sum() - design.shape[1]))),
        'observationCount': int(usable.sum()),
        'fitted': (design_matrix(years, 2 + 2 * harmonic_count, period) @ coefficients).tolist()
    }


def coefficient_count_for(count, maximum):
    """COLD's rule: 4 coefficients below 18 observations, 6 below 24, 8 above; capped by `maximum`."""
    return min(maximum, 8 if count >= 24 else 6 if count >= 18 else 4)


def lasso_fit(design, y, penalty, sweeps=2000):
    """Coordinate descent on the glmnet objective RSS / (2n) + penalty * sum(|beta|), intercept
    unpenalized, predictors not standardized (a stated difference from glmnet's default)."""
    beta = np.zeros(design.shape[1])
    squared_norms = np.sum(design**2, axis=0)
    penalty_times_count = penalty * design.shape[0]
    residual = y.copy()
    for _ in range(sweeps):
        largest_change = 0.0
        for column in range(design.shape[1]):
            old = beta[column]
            correlation = design[:, column] @ residual + squared_norms[column] * old
            if column == 0:
                updated = correlation / squared_norms[column]
            else:
                updated = np.sign(correlation) * max(abs(correlation) - penalty_times_count, 0.0) / squared_norms[column]
            if updated != old:
                residual += design[:, column] * (old - updated)
                beta[column] = updated
                largest_change = max(largest_change, abs(updated - old))
        if largest_change < 1e-13:
            break
    return beta


def fit_bands(values, years, column_count, period, toggles, penalty):
    design = design_matrix(years, column_count, period)
    coefficients = np.zeros((values.shape[1], 8))
    root_mean_squares = np.zeros(values.shape[1])
    for band in range(values.shape[1]):
        if 'lasso' in toggles:
            beta = lasso_fit(design, values[:, band], penalty)
        else:
            beta, *_ = np.linalg.lstsq(design, values[:, band], rcond=None)
        coefficients[band, :column_count] = beta
        residual = values[:, band] - design @ beta
        root_mean_squares[band] = math.sqrt(np.sum(residual**2) / (len(years) - column_count))
    return coefficients, root_mean_squares


def predict(coefficients, year, period):
    return (design_matrix(np.array([year]), 8, period) @ coefficients[:, :].T).ravel()


def temporal_rmse(residuals, included_years, year, count=24):
    """COLD's temporally adjusted RMSE: residuals of the `count` observations closest in day of year."""
    distance = np.abs(((included_years - year) + 0.5) % 1.0 - 0.5)
    order = np.argsort(distance, kind='stable')[: min(count, len(included_years))]
    return np.sqrt(np.mean(residuals[order] ** 2, axis=0))


def tmask_outliers(values, years, period, threshold=4.42, passes=3):
    """Robust (bisquare) four-coefficient fit on two bands; flags observations beyond 4.42 RMSE."""
    design = design_matrix(years, 4, period)
    flagged = np.zeros(len(years), dtype=bool)
    for band in (0, values.shape[1] - 1):
        y = values[:, band]
        weights = np.ones(len(years))
        for _ in range(passes):
            root = np.sqrt(weights)
            beta, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
            residual = y - design @ beta
            scale = max(1.4826 * np.median(np.abs(residual)), 1e-9)
            u = np.minimum(np.abs(residual) / (6.0 * scale), 1.0)
            weights = np.maximum((1 - u**2) ** 2, 1e-4)
        rmse = math.sqrt(np.sum(weights * residual**2) / max(np.sum(weights) - 4, 1))
        flagged |= np.abs(residual) > threshold * rmse
    return flagged


def cold_monitor(values, seconds, config, toggles=frozenset(), mask=None):
    """The simplified monitor, with optional COLD elements. `values` is slices by bands."""
    years = centred_years(seconds)
    period = config['period']
    minimum_rmse = np.array(config['minimumRMSE'])
    usable = np.all(np.isfinite(values), axis=1)
    if mask is not None:
        usable &= mask
    observations = np.nonzero(usable)[0]
    count = len(observations)
    fit_budget = config['maximumFitCount']
    state = {'fits': 0, 'status': 0}

    def fit(positions, column_count):
        if state['fits'] >= fit_budget:
            state['status'] |= 8
            return None
        state['fits'] += 1
        chosen = observations[positions]
        return fit_bands(
            values[chosen], years[chosen], column_count, period, toggles, config.get('lassoPenalty', 0.0)
        )

    segments = []
    cursor = 0
    while cursor < count:
        if len(segments) == config['maximumSegmentCount']:
            state['status'] |= 16
            break
        start = cursor
        window = None
        while start + config['minimumObservations'] <= count and state['status'] & 8 == 0:
            end = start + config['minimumObservations']
            while end <= count and years[observations[end - 1]] - years[observations[start]] < config['minimumSpanYears']:
                end += 1
            if end > count:
                break
            span = observations[start:end]
            if np.max(np.diff(years[span])) >= config['maximumInitializationGapYears']:
                start += 1
                continue
            positions = np.arange(start, end)
            if 'tmask' in toggles:
                outliers = tmask_outliers(values[span], years[span], period)
                positions = positions[~outliers]
                if len(positions) < config['minimumObservations']:
                    start += 1
                    continue
            result = fit(positions, 4)
            if result is None:
                break
            coefficients, rmse = result
            floor = np.maximum(rmse, minimum_rmse)
            kept = observations[positions]
            first_residual = values[kept[0]] - predict(coefficients, years[kept[0]], period)
            last_residual = values[kept[-1]] - predict(coefficients, years[kept[-1]], period)
            span_years = years[kept[-1]] - years[kept[0]]
            # The larger of the two end residuals, as pyxccd (`fabs(v_slope) + max(fabs(v_start), fabs(v_end))`).
            end_residual = np.maximum(np.abs(first_residual), np.abs(last_residual))
            score = np.sum(((np.abs(coefficients[:, 1]) * span_years + end_residual) / floor) ** 2)
            if score <= config['changeThreshold']:
                window = (end, positions)
                break
            start += 1
        if window is None:
            break
        end, included_positions = window
        included = list(included_positions)
        column_count = coefficient_count_for(len(included), config['maximumCoefficientCount'])
        result = fit(np.array(included), column_count)
        if result is None:
            break
        coefficients, rmse = result
        fitted_count = len(included)
        position = end
        break_position = None
        magnitudes = None
        while position + config['consecutive'] <= count:
            peek = observations[position : position + config['consecutive']]
            floors = np.zeros((len(peek), values.shape[1]))
            residuals = values[peek] - np.stack([predict(coefficients, years[slice_index], period) for slice_index in peek])
            if 'temporal-rmse' in toggles:
                chosen = observations[np.array(included)]
                fit_residual = values[chosen] - np.stack([predict(coefficients, years[slice_index], period) for slice_index in chosen])
                for row, slice_index in enumerate(peek):
                    floors[row] = np.maximum(temporal_rmse(fit_residual, years[chosen], years[slice_index]), minimum_rmse)
            else:
                floors[:] = np.maximum(rmse, minimum_rmse)
            normalized = residuals / floors
            scores = np.sum(normalized**2, axis=1)
            cosines = []
            for row in range(len(peek) - 1):
                left, right = normalized[row], normalized[row + 1]
                cosines.append((left @ right) / max(math.sqrt((left @ left) * (right @ right)), 1e-12))
            mean_cosine = np.mean(cosines)
            if np.all(scores > config['changeThreshold']) and mean_cosine >= config['minimumMeanCosine']:
                break_position = position
                magnitudes = np.median(residuals, axis=0)
                break
            if scores[0] > config['outlierThreshold']:
                position += 1
                continue
            included.append(position)
            position += 1
            refit_due = len(included) * 100 >= fitted_count * 133 or 'every-observation-refit' in toggles
            if refit_due and len(included) != fitted_count:
                refit_column_count = coefficient_count_for(len(included), config['maximumCoefficientCount'])
                result = fit(np.array(included), refit_column_count)
                if result is None:
                    break
                coefficients, rmse = result
                column_count = refit_column_count
                fitted_count = len(included)
        if break_position is None:
            included.extend(range(position, count))
        if len(included) != fitted_count:
            final_column_count = coefficient_count_for(len(included), config['maximumCoefficientCount'])
            result = fit(np.array(included), final_column_count)
            if result is not None:
                coefficients, rmse = result
                column_count = final_column_count
        segments.append(
            {
                'startSlice': int(observations[included[0]]),
                'endSlice': int(observations[included[-1]]),
                'breakSlice': None if break_position is None else int(observations[break_position]),
                'coefficientCount': column_count,
                'coefficients': coefficients.tolist(),
                'rmse': rmse.tolist(),
                'changeMagnitude': None if magnitudes is None else magnitudes.tolist()
            }
        )
        if state['status'] & 8 or break_position is None:
            break
        cursor = break_position
    return {
        'segments': segments,
        'status': state['status'] | (0 if segments else 1),
        'fitCount': state['fits']
    }


def bfast_monitor(y, seconds, history_count, harmonic_count, include_trend, period, bandwidth, critical_value):
    """OLS-MOSUM monitoring of a harmonic fit on the history period, as a binary64 definition."""
    years = centred_years(seconds)
    usable = np.nonzero(np.isfinite(y))[0]
    history = usable[usable < history_count]
    monitor = usable[usable >= history_count]
    full = design_matrix(years, 2 + 2 * harmonic_count, period)
    if not include_trend:
        full = np.delete(full, 1, axis=1)
    design = full[history]
    if len(history) <= design.shape[1]:
        return {'status': 1}
    beta, *_ = np.linalg.lstsq(design, y[history], rcond=None)
    residuals = y[usable] - full[usable] @ beta
    history_count_valid = len(history)
    sigma = math.sqrt(np.sum(residuals[: history_count_valid] ** 2) / (history_count_valid - design.shape[1]))
    bandwidth_count = max(1, math.floor(bandwidth * history_count_valid))
    process_scale = sigma * math.sqrt(history_count_valid)
    break_slice = None
    largest_ratio = 0.0
    process = []
    for offset, slice_index in enumerate(monitor):
        position = history_count_valid + offset
        window = residuals[max(0, position - bandwidth_count + 1) : position + 1]
        value = float(np.sum(window) / process_scale)
        ratio_time = (position + 1) / history_count_valid
        log_plus = math.log(ratio_time) if ratio_time > math.e else 1.0
        boundary = critical_value * math.sqrt(2.0 * log_plus)
        process.append(value)
        largest_ratio = max(largest_ratio, abs(value) / boundary)
        if break_slice is None and abs(value) > boundary:
            break_slice = int(slice_index)
    return {
        'status': 0,
        'coefficients': beta.tolist(),
        'breakSlice': break_slice,
        'magnitude': float(np.median(residuals[history_count_valid:])) if len(monitor) else None,
        'maximumBoundaryRatio': largest_ratio,
        'historyObservationCount': history_count_valid,
        'process': process
    }


def irregular_seconds(generator, count, base=1_000_000_000):
    gaps = np.round(generator.uniform(10, 32, count - 1) * 86400).astype(np.int64)
    return (base + np.concatenate([[0], np.cumsum(gaps)])).astype(np.int64)


def series(seconds, intercept, slope, amplitude, phase, noise, generator):
    years = centred_years(seconds)
    return (
        intercept
        + slope * years
        + amplitude * np.cos(TWO_PI * years - phase)
        + noise * generator.standard_normal(len(years))
    )


def stack(seconds, generator, band_parameters, noise):
    return np.stack(
        [series(seconds, *parameters, noise, generator) for parameters in band_parameters], axis=1
    )


BANDS = [(0.25, 0.002, 0.06, 0.4), (0.18, -0.001, 0.05, 0.9), (0.42, 0.003, 0.08, 1.4)]


def base_config():
    probability = 0.99
    return {
        'period': 1.0,
        'bandCount': len(BANDS),
        'changeProbability': probability,
        'changeThreshold': float(chi2.ppf(probability, len(BANDS))),
        'outlierThreshold': float(chi2.ppf(1 - 1e-6, len(BANDS))),
        'minimumMeanCosine': math.cos(math.radians(45.0)),
        'minimumObservations': 12,
        'minimumSpanYears': 1.0,
        'maximumInitializationGapYears': 1.0,
        'consecutive': 4,
        'maximumCoefficientCount': 8,
        'maximumSegmentCount': 4,
        'maximumFitCount': 96,
        'minimumRMSE': [0.002] * len(BANDS)
    }


def to_jsonable(values):
    return [[None if not np.isfinite(value) else float(value) for value in row] for row in values]


def cold_case(name, seconds, values, config, variant_toggles, note, mask=None):
    simplified = cold_monitor(values, seconds, config, mask=mask)
    variants = []
    for toggle in variant_toggles:
        variant = cold_monitor(values, seconds, config, frozenset([toggle]), mask=mask)
        variants.append({'toggle': toggle, 'result': variant})
    return {
        'name': name,
        'note': note,
        'timesSeconds': [int(value) for value in seconds],
        'values': to_jsonable(values),
        'mask': None if mask is None else [int(value) for value in mask],
        'config': config,
        'simplified': simplified,
        'variants': variants
    }


def break_slices(result):
    return [segment['breakSlice'] for segment in result['segments']]


def require_difference(case, toggle, compare):
    variant = next(entry for entry in case['variants'] if entry['toggle'] == toggle)
    if not compare(case['simplified'], variant['result']):
        raise SystemExit(f"{case['name']}: {toggle} does not change the result")


def main():
    generator = np.random.default_rng(404)
    cases = []
    harmonic_seconds = irregular_seconds(generator, 48)
    harmonic_y = series(harmonic_seconds, 0.3, 0.01, 0.1, 0.7, 0.01, generator)
    harmonic_y[[5, 6, 31]] = np.nan
    harmonic_cases = []
    for harmonics in (1, 3):
        harmonic_cases.append(
            {
                'harmonicCount': harmonics,
                **harmonic_regression(harmonic_y, centred_years(harmonic_seconds), harmonics, 1.0)
            }
        )

    config = base_config()
    seconds = irregular_seconds(generator, 72)
    values = stack(seconds, generator, BANDS, 0.004)
    values[40:] += np.array([0.08, 0.1, -0.12])
    cases.append(cold_case('one-break', seconds, values, config, [], 'One abrupt change; matches the simplified reference.'))

    seconds = irregular_seconds(generator, 96)
    values = stack(seconds, generator, BANDS, 0.004)
    values[34:] += np.array([0.1, 0.09, -0.08])
    values[70:] += np.array([-0.12, -0.08, 0.1])
    cases.append(cold_case('two-breaks', seconds, values, config, [], 'Two abrupt changes on one cell.'))

    seconds = irregular_seconds(generator, 72)
    values = stack(seconds, generator, BANDS, 0.004)
    values[50:] += np.array([0.08, 0.09, -0.07])
    mask = np.ones(72, dtype=bool)
    mask[[7, 22, 23, 61]] = False
    cases.append(cold_case('masked-one-break', seconds, values, config, [], 'Quality mask removes observations; the axis is unchanged.', mask))

    # Tmask: a cloud-like low value on the two Tmask bands inside the first initialization window.
    # With Tmask it is removed and the series initializes at once; without it the stability test
    # rejects the window and the segment starts later.
    tmask_case = None
    for seed in range(100):
        trial_generator = np.random.default_rng(1000 + seed)
        seconds = irregular_seconds(trial_generator, 64)
        values = stack(seconds, trial_generator, BANDS, 0.004)
        values[9] -= np.array([0.5, 0.0, 0.5])
        trial = cold_case('tmask-cloud-in-window', seconds, values, config, ['tmask'], '')
        simplified_segment = trial['simplified']['segments'][0] if trial['simplified']['segments'] else None
        tmask_segment = trial['variants'][0]['result']['segments'][0] if trial['variants'][0]['result']['segments'] else None
        if simplified_segment and tmask_segment and simplified_segment['startSlice'] != tmask_segment['startSlice']:
            tmask_case = trial
            tmask_case['seed'] = 1000 + seed
            break
    if tmask_case is None:
        raise SystemExit('no Tmask seed found')
    tmask_case['note'] = (
        'A cloud on the Tmask bands inside the first window. Simplified has no Tmask, so the stability '
        'test shifts the start of the first segment; with the Tmask toggle the window initializes at once.'
    )
    cases.append(tmask_case)
    require_difference(
        tmask_case,
        'tmask',
        lambda left, right: left['segments'][0]['startSlice'] != right['segments'][0]['startSlice']
    )

    # LASSO (lambda 20 on a 0-10000 scale, 0.002 here) shrinks harmonic terms of a short fit.
    lasso_config = dict(base_config(), lassoPenalty=0.002)
    seconds = irregular_seconds(generator, 40)
    values = stack(seconds, generator, BANDS, 0.01)
    lasso_case = cold_case('lasso-shrinkage', seconds, values, lasso_config, ['lasso'], (
        'Short noisy stack: OLS keeps every coefficient, COLD-style LASSO shrinks them. '
        'The simplified output must equal OLS, not the shrunken fit.'
    ))
    cases.append(lasso_case)

    def coefficient_gap(left, right):
        gaps = []
        for left_segment, right_segment in zip(left['segments'], right['segments'], strict=False):
            gaps.append(
                np.max(
                    np.abs(np.array(left_segment['coefficients'])[:, 1:] - np.array(right_segment['coefficients'])[:, 1:])
                )
            )
        return max(gaps) if gaps else 0.0

    require_difference(lasso_case, 'lasso', lambda left, right: coefficient_gap(left, right) > 1e-3)
    lasso_case['minimumCoefficientGap'] = 1e-3

    # Every-observation refit: a slow drift after the first segment initializes. The simplified
    # monitor refits only when the included count grows by 1.33x and flags a change the fully
    # updated model does not.
    refit_case = None
    for seed in range(200):
        trial_generator = np.random.default_rng(2000 + seed)
        seconds = irregular_seconds(trial_generator, 88)
        values = stack(seconds, trial_generator, BANDS, 0.003)
        years = centred_years(seconds)
        drift_start = years[20]
        values += np.outer(np.maximum(years - drift_start, 0.0), [0.12, 0.1, -0.1]) * 0.6
        trial = cold_case('refit-cadence-drift', seconds, values, config, ['every-observation-refit'], '')
        left = break_slices(trial['simplified'])
        right = break_slices(trial['variants'][0]['result'])
        if left != right and any(item is not None for item in left):
            refit_case = trial
            refit_case['seed'] = 2000 + seed
            break
    if refit_case is None:
        raise SystemExit('no refit cadence seed found')
    refit_case['note'] = 'Slow drift: refitting at 1.33x growth versus at every observation changes the detected breaks.'
    cases.append(refit_case)
    require_difference(refit_case, 'every-observation-refit', lambda left, right: break_slices(left) != break_slices(right))

    # Temporally adjusted RMSE: noise concentrated in one season, change in the quiet season.
    temporal_case = None
    for seed in range(300):
        trial_generator = np.random.default_rng(3000 + seed)
        seconds = irregular_seconds(trial_generator, 96)
        years = centred_years(seconds)
        phase = np.cos(TWO_PI * years)
        values = stack(seconds, trial_generator, BANDS, 0.0)
        noise = 0.001 + 0.012 * (phase > 0.2)
        values += trial_generator.standard_normal(values.shape) * noise[:, None]
        change = 40 + int(trial_generator.integers(0, 20))
        values[change:] += np.array([0.016, 0.014, -0.016])
        trial = cold_case('temporal-rmse-season-noise', seconds, values, config, ['temporal-rmse'], '')
        left = break_slices(trial['simplified'])
        right = break_slices(trial['variants'][0]['result'])
        if left != right and any(item is not None for item in right):
            temporal_case = trial
            temporal_case['seed'] = 3000 + seed
            break
    if temporal_case is None:
        raise SystemExit('no temporal RMSE seed found')
    temporal_case['note'] = (
        'Noise concentrated in one season and a small change: a seasonally adjusted RMSE and a single '
        'whole-segment RMSE detect different breaks.'
    )
    cases.append(temporal_case)
    require_difference(temporal_case, 'temporal-rmse', lambda left, right: break_slices(left) != break_slices(right))

    # COLD's coefficient rule needs windows of under 18 and under 24 observations, which only a
    # sparse axis gives: a window must still span a year. Own generators keep earlier cases stable.
    for observation_count, expected_columns in ((16, 4), (20, 6)):
        sparse_generator = np.random.default_rng(4000 + observation_count)
        gaps = np.round(sparse_generator.uniform(34, 42, observation_count - 1) * 86400).astype(np.int64)
        seconds = (1_000_000_000 + np.concatenate([[0], np.cumsum(gaps)])).astype(np.int64)
        values = stack(seconds, sparse_generator, BANDS, 0.003)
        sparse_case = cold_case(
            f'sparse-{observation_count}-observations',
            seconds,
            values,
            # A short window has few residuals, so a realistic RMSE floor keeps the stability test
            # from rejecting every window of this short series.
            dict(config, minimumRMSE=[0.01] * len(BANDS)),
            [],
            f'Sparse axis with {observation_count} observations: the coefficient rule selects {expected_columns}.'
        )
        segments = sparse_case['simplified']['segments']
        if len(segments) != 1 or segments[0]['coefficientCount'] != expected_columns:
            raise SystemExit(f'sparse case {observation_count} does not exercise the coefficient rule')
        cases.append(sparse_case)

    # BFAST Monitor: a stable history, then an abrupt change in the monitoring period.
    bfast_generator = np.random.default_rng(505)
    bfast_seconds = irregular_seconds(bfast_generator, 96)
    bfast_cases = []
    y = series(bfast_seconds, 0.4, 0.002, 0.1, 0.5, 0.008, bfast_generator)
    y[60:] -= 0.09
    y[[10, 47, 70]] = np.nan
    settings = {'historyCount': 48, 'harmonicCount': 3, 'includeTrend': True, 'period': 1.0, 'bandwidth': 0.25, 'criticalValue': 1.1}
    bfast_cases.append(
        {
            'name': 'abrupt-drop',
            'values': [None if not np.isfinite(value) else float(value) for value in y],
            'timesSeconds': [int(value) for value in bfast_seconds],
            **settings,
            'expected': bfast_monitor(y, bfast_seconds, 48, 3, True, 1.0, 0.25, 1.1)
        }
    )
    y = series(bfast_seconds, 0.4, 0.0, 0.1, 0.5, 0.008, bfast_generator)
    y[[3, 80]] = np.nan
    settings = {'historyCount': 48, 'harmonicCount': 2, 'includeTrend': False, 'period': 1.0, 'bandwidth': 0.5, 'criticalValue': 2.0}
    bfast_cases.append(
        {
            'name': 'stable-no-trend',
            'values': [None if not np.isfinite(value) else float(value) for value in y],
            'timesSeconds': [int(value) for value in bfast_seconds],
            **settings,
            'expected': bfast_monitor(y, bfast_seconds, 48, 2, False, 1.0, 0.5, 2.0)
        }
    )
    if bfast_cases[0]['expected']['breakSlice'] is None or bfast_cases[1]['expected']['breakSlice'] is not None:
        raise SystemExit('BFAST Monitor fixture is not discriminating')

    OUTPUT.write_text(
        json.dumps(
            {
                'description': 'Binary64 NumPy references for RTS-04. pyxccd is not run here; its fixture is rts04-cold-pyxccd.json (see rts04-cold-pyxccd.py).',
                'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
                'harmonicRegression': {
                    'timesSeconds': [int(value) for value in harmonic_seconds],
                    'values': [None if not np.isfinite(value) else float(value) for value in harmonic_y],
                    'cases': harmonic_cases
                },
                'cold': cases,
                'bfastMonitor': bfast_cases
            }
        )
    )
    for case in cases:
        print(case['name'], break_slices(case['simplified']), [break_slices(entry['result']) for entry in case['variants']], case['simplified']['status'])
    for case in bfast_cases:
        print(case['name'], case['expected']['breakSlice'], case['expected']['maximumBoundaryRatio'], case['expected']['magnitude'])


if __name__ == '__main__':
    main()
