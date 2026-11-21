# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/emerging-hot-spots/audit-emerging-hot-spots.oracle.py

Independent definition oracle. NumPy matrix products compute Getis-Ord Gi* with a
population variance (ddof=0), explicit space-time weight vectors and finite/masked
selection. SciPy normal tails and false_discovery_control supply p-values/BH/BY;
Holm and Bonferroni use their closed forms. Mann-Kendall uses sign comparisons and
the exact tie correction with continuity correction. Modified MK uses SciPy Sen
slopes/rankdata and NumPy correlation, not production running sums or heap selection.
Category predicates follow the ArcGIS category table and the documented finite-slice,
final-slice, minimum-trend and legacy-sporadic conventions. No TypeScript is imported.

JSON null denotes NaN. This script prints the literal fixture; tests never run Python.
"""
import json
import math
import numpy as np
import scipy
from scipy.stats import norm, rankdata, theilslopes, false_discovery_control


def pack(settings):
    confidence = settings.get('confidenceLevel', .95)
    method = settings.get('multipleTesting', 'benjamini-hochberg')
    critical = settings.get('criticalZ', norm.ppf((1 + confidence) / 2) if confidence in [.9, .95, .99] else 0)
    return np.array([settings.get('radius', 0), settings['temporalWindow'], critical,
                     settings.get('trendSignificanceLevel', 1-confidence),
                     settings.get('persistentFraction', .9), settings.get('minimumTrendSliceCount', 10),
                     1-confidence, int(not settings.get('sporadicRequiresFinalSlice', True))], dtype=np.float32)


def adjust(scores, method):
    result = np.full(scores.shape, np.nan)
    included = np.isfinite(scores)
    values = scores[included]
    count = len(values)
    if not count:
        return result
    if method in ['benjamini-hochberg', 'benjamini-yekutieli']:
        adjusted = false_discovery_control(values, method='bh' if method == 'benjamini-hochberg' else 'by')
    elif method == 'bonferroni':
        adjusted = np.minimum(values * count, 1)
    elif method == 'holm':
        order = np.argsort(values, kind='stable')
        ordered = np.minimum(np.maximum.accumulate(values[order] * np.arange(count, 0, -1)), 1)
        adjusted = np.empty_like(values)
        adjusted[order] = ordered
    else:
        adjusted = values
    result[included] = adjusted
    return result


def trend(series, correction='none', maximum_lag=0, critical=norm.ppf(.975)):
    values = np.array(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    count = len(values)
    statistic = int(np.sign(values[np.newaxis, :] - values[:, np.newaxis])[np.triu_indices(count, 1)].sum())
    groups = np.unique(values, return_counts=True)[1]
    tie_term = int(np.sum(groups * (groups-1) * (2*groups+5)))
    variance = (count*(count-1)*(2*count+5) - tie_term) / 18
    factor = 1.
    if correction != 'none' and count >= 3:
        slope = theilslopes(values, np.arange(count)).slope
        detrended = values - slope * np.arange(1, count+1)
        signal = rankdata(detrended, method='average') if correction == 'hamed-rao' else detrended
        centered = signal - signal.mean()
        squares = centered @ centered
        if squares > 0:
            correlations = np.correlate(centered, centered, mode='full')[count:] / squares
            last = min(maximum_lag or count-1, count-1)
            lags = np.arange(1, last+1)
            correlations = correlations[:last]
            if correction == 'hamed-rao':
                selected = np.abs(correlations) > critical / np.sqrt(count)
                gaps = count - lags[selected]
                candidate = 1 + 2*np.sum(gaps*(gaps-1)*(gaps-2)*correlations[selected])/(count*(count-1)*(count-2))
            else:
                candidate = 1 + 2*np.sum((1-lags/count)*correlations)
            if candidate > 0:
                factor = float(candidate)
    deviation = np.sqrt(variance * factor)
    score = (statistic - np.sign(statistic)) / deviation if variance > 0 else 0.
    probability = 2 * norm.sf(abs(score)) if variance > 0 else 1.
    return dict(count=count, statistic=statistic, tieTerm=tie_term, variance=variance,
                factor=factor, z=float(score), p=float(probability))


def cells(scores, packed, method='none', final='require', correction='none', maximum_lag=0, critical=norm.ppf(.975)):
    probabilities = 2 * norm.sf(np.abs(scores))
    adjusted = adjust(probabilities.ravel(), method).reshape(scores.shape)
    states = np.where(np.isfinite(scores), np.where(scores >= packed[2], 1, np.where(scores <= -packed[2], -1, 0)), 0)
    if method != 'none':
        states = np.where(np.isfinite(scores) & (adjusted <= packed[6]), np.sign(scores), 0)
    result = dict(trendS=[], trendZ=[], trendP=[], category=[], hotSliceCount=[], coldSliceCount=[], lastValidSlice=[])
    for series, state in zip(scores, states):
        included = np.isfinite(series)
        finite_states = state[included]
        count = int(included.sum())
        last = int(np.flatnonzero(included)[-1]) if count else 0xffffffff
        test = trend(series, correction, maximum_lag, critical)
        known = count >= packed[5]
        z, probability = (test['z'], test['p']) if known else (np.nan, np.nan)
        hot, cold = int((finite_states == 1).sum()), int((finite_states == -1).sum())
        category = 0
        if count:
            final_state = int(finite_states[-1])
            if final == 'require' and last != len(series)-1:
                category = 17
            elif final_state:
                count_same = hot if final_state == 1 else cold
                count_other = cold if final_state == 1 else hot
                trailing = 0
                for flag in finite_states[::-1]:
                    if flag != final_state:
                        break
                    trailing += 1
                persistent = count_same >= float(packed[4]) * count
                significant_direction = np.sign(z) * final_state if probability <= packed[3] else 0
                if count_same == 1:
                    base = 1
                elif trailing == count_same and trailing >= 2 and not persistent:
                    base = 2
                elif persistent:
                    base = 17 if not known else 3 if significant_direction > 0 else 5 if significant_direction < 0 else 4
                else:
                    base = 6 if count_other == 0 else 7
                category = base if base == 17 or final_state == 1 else base + 8
            elif hot > 0 and hot >= float(packed[4]) * count:
                category = 8
            elif cold > 0 and cold >= float(packed[4]) * count:
                category = 16
            elif packed[7] and (hot == 0) != (cold == 0):
                category = 6 if hot else 14
        for name, value in dict(trendS=test['statistic'], trendZ=z, trendP=probability, category=category,
                                hotSliceCount=hot, coldSliceCount=cold, lastValidSlice=last).items():
            result[name].append(value)
    return result, probabilities.ravel().tolist(), adjusted.ravel().tolist()


VALUES = np.array([[-4, -3, -3, -1], [4, 3, 3, 1], [-2, 0, 0, 2], [2, 0, 0, -2],
                   [0, 1, np.nan, 3], [0, -1, np.nan, -3], [9, 9, 9, 9], [np.nan]*4])
MASK = np.array([1, 1, 1, 1, 1, 1, 0, 1])
WEIGHTS = dict(offsets=[0, 2, 4, 6, 8, 9, 9, 10, 10],
               neighbors=[1, 2, 0, 3, 0, 4, 1, 5, 2, 0], weights=[.5, 2, 1, .5, 2, 1, .5, 2, 1, 1])
SETTINGS = dict(radius=1, temporalWindow=1, confidenceLevel=.8, criticalZ=.75,
                multipleTesting='none', trendSignificanceLevel=.2, persistentFraction=.75, minimumTrendSliceCount=3)


def cube(name, mode='lattice', method='none', self_weight=1, correction='none', maximum_lag=1,
         final='require', affine=False, unsigned=False, constant=False, empty=False, missing_final=False):
    values = VALUES.copy()
    if unsigned:
        values = np.nan_to_num(values, nan=0) + 4
    if constant:
        values[:] = 3
    if missing_final:
        values[4:6, -1] = np.nan
    if affine:
        values = 2*values + 1048576
    values = values.astype(np.float32).astype(np.float64)
    mask = np.zeros(8) if empty else MASK
    settings = {**SETTINGS, 'multipleTesting': method}
    if method != 'none':
        del settings['criticalZ']
    packed = pack(settings)
    valid = np.isfinite(values) & (mask[:, None] != 0)
    count = int(valid.sum())
    mean = float(values[valid].mean()) if count else 0
    variance = float(values[valid].var(ddof=0)) if count else 0
    spatial = np.zeros((8, 8))
    if mode == 'lattice':
        for first in range(8):
            for second in range(8):
                if (first//4-second//4)**2 + (first%4-second%4)**2 <= 1:
                    spatial[first, second] = 1
    else:
        np.fill_diagonal(spatial, self_weight)
        for row in range(8):
            for slot in range(WEIGHTS['offsets'][row], WEIGHTS['offsets'][row+1]):
                spatial[row, WEIGHTS['neighbors'][slot]] = WEIGHTS['weights'][slot]
    scores = np.full(values.shape, np.nan)
    for row in range(8):
        for period in range(4):
            if not valid[row, period] or count < 2 or variance == 0:
                continue
            time_mask = np.zeros(4)
            time_mask[max(0, period-1):period+1] = 1
            weighted = spatial[row, :, None] * time_mask[None, :] * valid
            weight_sum = weighted.sum()
            spread = (count*np.square(weighted).sum() - weight_sum**2)/(count-1)
            if spread > 0:
                scores[row, period] = ((weighted* np.nan_to_num(values)).sum() - mean*weight_sum)/np.sqrt(variance*spread)
    # Documented ties are exact f32 score equality, not fuzzy float64 equality.
    published_scores = scores.astype(np.float32).astype(np.float64)
    expected, probabilities, adjusted = cells(published_scores, packed, method, final, correction, maximum_lag, critical=0.1)
    expected.update(giZScores=scores.ravel().tolist(), pValues=probabilities, adjustedPValues=adjusted,
                    globalStatistics=[count, mean, variance, math.sqrt(variance)], includedCount=[count])
    return dict(name=name, mode=mode, method=method, correction=correction, maximumLag=maximum_lag,
                correctionCriticalZ=.1, finalSlice=final, selfWeight=self_weight, unsigned=unsigned,
                values=values.ravel().tolist(), mask=mask.tolist(), settings=settings,
                parameters=packed.astype(float).tolist(), expected=expected)


def classification_cases():
    positive = [[0]*20, [0]*19+[3], [0]*17+[3]*3, [2+i/8 for i in range(20)], [3]*20,
                [2+i/8 for i in reversed(range(20))], [3 if i % 2 == 0 or i == 19 else 0 for i in range(20)],
                [-3]+[3 if i % 2 == 1 else 0 for i in range(1, 20)], [3]*19+[0]]
    rows = positive + [[-value for value in row] for row in positive[1:]]
    rows += [[np.nan]*14+[3]*6, [0]*18+[3, np.nan], [np.nan]*20, [3, 0]*10, [-3, 0]*10]
    settings = dict(temporalWindow=0, multipleTesting='none', confidenceLevel=.95)
    scores = np.array(rows, dtype=np.float32).astype(float)
    result = []
    for final in ['require', 'last-valid']:
        for legacy in [False, True]:
            packed = pack({**settings, 'sporadicRequiresFinalSlice': not legacy})
            expected, _, _ = cells(scores, packed, final=final)
            result.append(dict(name=f'categories-{final}-legacy-{legacy}', scores=scores.ravel().tolist(), sliceCount=20,
                               parameters=packed.astype(float).tolist(), correction='none', maximumLag=0,
                               correctionCriticalZ=norm.ppf(.975), finalSlice=final, expected=expected))
    return result


def trend_cases():
    # Row 5: exact slope -1 leaves a constant detrended signal (10), or midrank 6.5.
    # All corrections must retain factor 1: S=-66, Var(S)=638/3, Z=-65/sqrt(638/3).
    # A rounded GPU mean must not turn identical nonzero residuals into autocorrelation.
    rows = [[0, 0, 0, 0, 2, 2, 2, 2, -1, -1, -1, -1],
            [0, 1, 2, 2, 3, 4, 3, 5, 5, 6, 7, 6],
            [3]*12, [np.nan]*12, [1, np.nan, 1, 2, 3, np.nan, 3, 4, 2, 1, 0, -1],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2], [0]*11+[3], [np.nan]*4+[3]*8]
    scores = np.array(rows, dtype=np.float32).astype(float)
    packed = pack(dict(temporalWindow=0, multipleTesting='none', minimumTrendSliceCount=3, criticalZ=1.5))
    result = []
    for correction, maximum_lag, critical in [('none', 0, norm.ppf(.975)), ('hamed-rao', 1, norm.ppf(.975)),
                                               ('hamed-rao', 3, .1), ('yue-wang', 1, norm.ppf(.975)), ('yue-wang', 0xffffffff, norm.ppf(.975))]:
        expected, _, _ = cells(scores, packed, final='last-valid', correction=correction, maximum_lag=maximum_lag, critical=critical)
        result.append(dict(name=f'trend-{correction}-{maximum_lag}-{critical}', scores=scores.ravel().tolist(), sliceCount=12,
                           parameters=packed.astype(float).tolist(), correction=correction, maximumLag=maximum_lag,
                           correctionCriticalZ=critical, finalSlice='last-valid', expected=expected))
    # Both cooperative dispatch boundaries, with tie groups spanning lane/word boundaries.
    for length in [64, 65]:
        scores = np.array([[index//16 for index in range(length)], [3]*length,
                           [index//16 for index in reversed(range(length))], [np.nan]*length,
                           [0 if index % 2 else 3 for index in range(length)],
                           [np.nan if index % 7 == 0 else index//8 for index in range(length)],
                           [0]*(length-1)+[3], [3]*(length-1)+[0]], dtype=np.float32).astype(float)
        expected, _, _ = cells(scores, packed, final='require')
        result.append(dict(name=f'cooperative-{length}', scores=scores.ravel().tolist(), sliceCount=length,
                           parameters=packed.astype(float).tolist(), correction='none', maximumLag=0,
                           correctionCriticalZ=norm.ppf(.975), finalSlice='require', expected=expected))
    # Largest accepted corrected heap: every pairwise slope of 0..62 is exactly one.
    # S=63*62/2=1953, Var(S)=28427, continuity Z=1952/sqrt(28427); correction factor 1.
    for correction in ['hamed-rao', 'yue-wang']:
        scores = np.tile(np.arange(63, dtype=np.float32), (8, 1)).astype(float)
        expected, _, _ = cells(scores, packed, final='require', correction=correction)
        result.append(dict(name=f'maximum-corrected-63-{correction}', scores=scores.ravel().tolist(), sliceCount=63,
                           parameters=packed.astype(float).tolist(), correction=correction, maximumLag=0,
                           correctionCriticalZ=norm.ppf(.975), finalSlice='require', expected=expected))
    return result


def clean(value):
    if isinstance(value, dict):
        return {key: clean(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(entry) for entry in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


if __name__ == '__main__':
    cubes = [cube('lattice'), cube('lattice-affine', affine=True), cube('weighted', mode='weights'),
             cube('weighted-affine', mode='weights', affine=True), cube('self-zero-island', mode='weights', self_weight=0),
             cube('lattice-uint32', unsigned=True), cube('constant', constant=True), cube('empty', empty=True),
             cube('missing-final-require', missing_final=True), cube('missing-final-last-valid', missing_final=True, final='last-valid'),
             cube('hamed-rao', correction='hamed-rao'), cube('yue-wang', correction='yue-wang')]
    cubes += [cube(method, method=method) for method in ['bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli']]
    payload = dict(provenance=dict(generator='audit-emerging-hot-spots.oracle.py', numpy=np.__version__, scipy=scipy.__version__),
                   weights=WEIGHTS, cubes=cubes, stages=classification_cases()+trend_cases(),
                   parameterCriticalValues=[norm.ppf(.95), norm.ppf(.975), norm.ppf(.995)])
    print(json.dumps(clean(payload), separators=(',', ':'), allow_nan=False))
