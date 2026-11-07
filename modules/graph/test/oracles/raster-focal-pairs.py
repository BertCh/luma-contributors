# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""RS05 composed NumPy 2.5.4 reference, not a direct terra execution.

Run: python -m pip install numpy==2.5.4; python raster-focal-pairs.py
Square focal windows match terra focalPairs/focalReg's unweighted, intercept-included
subset with paired NA removal and fillvalue=NA. Clamp/reflect are explicit extensions.
Window extraction and native nodata/calibration are adapters; cov, corrcoef and
linalg.lstsq are independent library calls, not copies of the GPU moment recurrence.
Inputs and calibration are rounded to float32; numerical reference calls use float64.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

assert np.__version__ == '2.5.4', 'Regenerate with the pinned NumPy version.'
METRICS = ['covariance', 'correlation', 'slope', 'intercept', 'rSquared', 'residualSumSquares']


def band(values, **options):
    return dict(values=values, **options)


def calibrated(source):
    values = np.array([np.nan if value is None else value for value in source['values']], dtype=np.float32)
    valid = np.isfinite(values)
    if 'noDataValue' in source:
        valid &= values != source['noDataValue']
    if 'validity' in source:
        valid &= np.asarray(source['validity'], dtype=bool)
    values = (values * np.float32(source.get('scale', 1)) + np.float32(source.get('offset', 0))).astype(np.float32)
    valid &= np.isfinite(values)
    return np.where(valid, values, np.nan).astype(np.float64)


def reflect(coordinate, length):
    if length == 1:
        return 0
    folded = coordinate % (2 * (length - 1))
    return min(folded, 2 * (length - 1) - folded)


def reference(case):
    width, height, radius = case['width'], case['height'], case['radius']
    predictor, response = calibrated(case['predictor']), calibrated(case['response'])
    result = {name: [] for name in ['pairedCount', 'status'] + METRICS}
    for row in range(height):
        for column in range(width):
            indices, outside = [], 0
            for neighborRow in range(row - radius, row + radius + 1):
                for neighborColumn in range(column - radius, column + radius + 1):
                    mappedRow, mappedColumn = neighborRow, neighborColumn
                    if not (0 <= neighborRow < height and 0 <= neighborColumn < width):
                        if case.get('borderMode', 'nodata') == 'nodata':
                            outside += 1
                            continue
                        if case['borderMode'] == 'clamp':
                            mappedRow, mappedColumn = np.clip(neighborRow, 0, height - 1), np.clip(neighborColumn, 0, width - 1)
                        else:
                            mappedRow, mappedColumn = reflect(neighborRow, height), reflect(neighborColumn, width)
                    indices.append(int(mappedRow * width + mappedColumn))
            selectedPredictor, selectedResponse = predictor[indices], response[indices]
            valid = np.isfinite(selectedPredictor) & np.isfinite(selectedResponse)
            selectedPredictor, selectedResponse = selectedPredictor[valid], selectedResponse[valid]
            count = len(selectedPredictor)
            center = row * width + column
            missing = (case.get('noDataPolicy', 'ignore') == 'propagate' and (outside > 0 or not valid.all())) or (case.get('propagateCenterNoData', False) and not np.isfinite([predictor[center], response[center]]).all())
            status = int(count < case.get('minimumCount', 2)) | (2 if missing else 0)
            values = {name: None for name in METRICS}
            if status == 0:
                predictorCentered = selectedPredictor - selectedPredictor.mean()
                responseCentered = selectedResponse - selectedResponse.mean()
                constantPredictor = np.ptp(selectedPredictor) == 0
                constantResponse = np.ptp(selectedResponse) == 0
                status |= 4 if constantPredictor else 0
                status |= 8 if constantResponse else 0
                values['covariance'] = float(np.cov(selectedPredictor, selectedResponse, ddof=0 if case.get('covarianceNormalization') == 'population' else 1)[0, 1])
                if not constantPredictor:
                    # Center the independent solve to avoid conditioning on arbitrary map offsets.
                    design = np.column_stack([predictorCentered, np.ones(count)])
                    coefficients, _, rank, _ = np.linalg.lstsq(design, responseCentered, rcond=None)
                    assert rank == 2
                    slope, centeredIntercept = coefficients
                    residuals = responseCentered - design @ coefficients
                    residualSumSquares = float(np.dot(residuals, residuals))
                    values.update(slope=float(slope), intercept=float(selectedResponse.mean() - slope * selectedPredictor.mean() + centeredIntercept), residualSumSquares=residualSumSquares)
                    if not constantResponse:
                        values['correlation'] = float(np.corrcoef(selectedPredictor, selectedResponse)[0, 1])
                        values['rSquared'] = float(1 - residualSumSquares / np.dot(responseCentered, responseCentered))
            result['pairedCount'].append(count)
            result['status'].append(status)
            for name in METRICS:
                result[name].append(values[name])
    return result


cases = []
base = [index % 7 - 3 + (index // 7) / 4 for index in range(35)]
response = [2 * value + ((index * 11) % 7 - 3) / 8 for index, value in enumerate(base)]
for mode in ['nodata', 'clamp', 'reflect']:
    for normalization in ['sample', 'population']:
        cases.append(dict(name=f'noisy-{mode}-{normalization}', width=7, height=5, radius=2, borderMode=mode, covarianceNormalization=normalization, predictor=band(base), response=band(response)))
for scenario, predictor, dependent in [
    ('positive', list(range(20)), [3 + 2 * index for index in range(20)]),
    ('negative', list(range(20)), [7 - index * 3 for index in range(20)]),
    ('constant-predictor', [5] * 20, list(range(20))),
    ('constant-response', list(range(20)), [8] * 20),
    ('constant-both', [5] * 20, [8] * 20),
    ('large-offset', [1e6 + index for index in range(20)], [2e6 + 3 + index * 2 for index in range(20)])
]:
    cases.append(dict(name=scenario, width=5, height=4, radius=1, predictor=band(predictor), response=band(dependent)))
# Unequal clipped window counts produce a 1/6 mean difference at the center.
# At offset 1e6 that mean is not representable, despite every original sample being exact.
# These distinguish centered differences from reassociated value - (anchor + meanDifference).
for name, predictor, dependent in [
    ('fractional-mean-predictor-offset', [1e6, 1e6 + .125, 1e6 + .375], [7, 8, 6]),
    ('fractional-mean-response-offset', [7, 8, 6], [1e6, 1e6 + .125, 1e6 + .375]),
    ('fractional-mean-both-offset', [1e6, 1e6 + .125, 1e6 + .375], [1e6 + .25, 1e6 + .125, 1e6 + .5])
]:
    cases.append(dict(name=name, width=3, height=1, radius=1, predictor=band(predictor), response=band(dependent)))
predictor = list(range(35))
predictor[4] = -999
predictor[9] = None
response = [2 * index - 7 for index in range(35)]
response[18] = -999
predictorMask = [int(index not in [0, 17, 26]) for index in range(35)]
responseMask = [int(index not in [6, 17, 31]) for index in range(35)]
for policy in ['ignore', 'propagate']:
    for centerPolicy in [False, True]:
        cases.append(dict(name=f'calibrated-masks-{policy}-center-{centerPolicy}', width=7, height=5, radius=1, noDataPolicy=policy, propagateCenterNoData=centerPolicy,
            predictor=band(predictor, validity=predictorMask, scale=0.5, offset=3, noDataValue=-999),
            response=band(response, validity=responseMask, scale=2, offset=-1, noDataValue=-999)))
for mode in ['nodata', 'clamp', 'reflect']:
    cases.append(dict(name=f'single-cell-{mode}', width=1, height=1, radius=1, borderMode=mode, predictor=band([3]), response=band([8])))
cases.extend([
    dict(name='empty-pairs', width=3, height=3, radius=1, predictor=band([None] * 9), response=band(list(range(9)))),
    dict(name='radius-zero', width=3, height=3, radius=0, predictor=band(list(range(9))), response=band(list(range(9)))),
    dict(name='minimum-count', width=3, height=3, radius=1, minimumCount=8, predictor=band(list(range(9))), response=band([2 * index + 1 for index in range(9)])),
    dict(name='two-pairs', width=2, height=1, radius=1, predictor=band([2, 4]), response=band([5, 9])),
    dict(name='maximum-radius', width=3, height=2, radius=8, borderMode='reflect', predictor=band([1, 4, 2, 5, 3, 7]), response=band([3, 8, 6, 10, 4, 14]))
])
for case in cases:
    case['expected'] = reference(case)
fixture = dict(provenance=dict(reference='NumPy cov, corrcoef and centered linalg.lstsq over explicitly extracted paired-valid windows; composed reference, no direct terra execution', versions=dict(numpy=np.__version__), generatorSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rounding='float32 calibrated input, float64 reference', tolerance=dict(absolute=2e-4, relative=2e-5), terraCompatibility='unweighted rectangular focalPairs Pearson and focalReg OLS with intercept; no geographic wrap; nodata outside grid'), cases=cases)
Path(__file__).with_name('fixtures').joinpath('raster-focal-pairs.json').write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n')
