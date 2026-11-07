# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent numerical references for the geospatial statistical expansion.

Run with an isolated environment containing numpy, scipy, esda, libpysal, mgwr,
spreg, and statsmodels. Inputs are synthetic (no external dataset), rounded to
float32 before the reference call. Positions use a native Euclidean plane.
The fixture records versions and identifies direct calls versus compositions.
LOSH/Geary: direct esda. Generalized GWR: independent statsmodels local GLMs with
Gaussian spatial frequency weights and Poisson log-exposure. MGWR: direct mgwr
Sel_BW multi-backfit constrained to fixed per-variable bandwidths. Conditional
inference is a NumPy linear-smoother composition, not mgwr's full covariance.
Panel: direct spreg.GM_Lag on scipy block-diagonal W. Regimes: direct OLS_Regimes.
Pair bins: scipy distance matrices plus NumPy aggregation (composition, not a
claim of direct spatstat/scikit-gstat parity). Bins use closed final endpoints.
"""
from importlib.metadata import version
from pathlib import Path
import json
import warnings
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import block_diag
import statsmodels.api as sm
from libpysal.weights import W, WSP
from esda import LOSH, Geary_Local_MV
from mgwr.sel_bw import Sel_BW
from spreg import GM_Lag, OLS_Regimes

warnings.filterwarnings('ignore')

def rounded(values):
    return np.asarray(values, dtype=np.float32).astype(float)

def serial(values):
    return np.asarray(values).tolist()

positions = rounded([[column, row] for row in range(4) for column in range(4)])
predictors = rounded([np.sin(row * 1.7) + row % 3 * 0.2 for row in range(16)])[:, None]
response = rounded(2.0 + 1.2 * predictors[:, 0] + 0.15 * positions[:, 0] + 0.2 * np.cos(np.arange(16)))[:, None]
neighbors = {row: [other for other in range(16) if np.sum(abs(positions[row] - positions[other])) == 1] for row in range(16)}
weights = W(neighbors, silence_warnings=True)
weights.transform = 'r'
second = rounded(np.cos(np.arange(16) * 0.8) + positions[:, 1] * 0.2)
local = {
    'positions': serial(positions.ravel()), 'values': serial(response.ravel()),
    'multivariateValues': serial(np.column_stack([response.ravel(), second]).ravel()),
    'offsets': [0] + list(np.cumsum([len(neighbors[row]) for row in range(16)]).astype(int)),
    'neighbors': [other for row in range(16) for other in neighbors[row]],
    'weights': [weight for row in range(16) for weight in weights.weights[row]],
    'losh': serial(LOSH(connectivity=weights).fit(response.ravel()).Hi),
    'geary': serial(Geary_Local_MV(connectivity=weights, permutations=0).fit([response.ravel(), second]).localG)
}

# Generalized local models with varying positions, masks and exposure offsets.
mask = np.ones(16, dtype=bool)
mask[[3, 12]] = False
exposure = rounded(1.0 + (np.arange(16) % 4) * 0.5)
poisson = rounded([0, 1, 4, 2, 1, 3, 5, 2, 0, 4, 2, 6, 1, 2, 3, 5])
logistic = rounded([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1])
bandwidth = 2.2
local_models = []
for family, observations in [('poisson', poisson), ('logistic', logistic)]:
    coefficients = []
    predictions = []
    for focus in range(16):
        if not mask[focus]:
            coefficients.extend([None, None]); predictions.append(None); continue
        spatial = np.exp(-np.sum((positions[mask] - positions[focus]) ** 2, axis=1) / (2 * bandwidth**2))
        offset = rounded(np.log(exposure))[mask] if family == 'poisson' else None
        model = sm.GLM(observations[mask], sm.add_constant(predictors[mask]), family=sm.families.Poisson() if family == 'poisson' else sm.families.Binomial(), freq_weights=spatial, offset=offset).fit(tol=1e-11, maxiter=100)
        coefficients.extend(serial(model.params))
        linear = float(model.params[0] + model.params[1] * predictors[focus, 0])
        predictions.append(float(np.exp(linear + rounded(np.log(exposure))[focus])) if family == 'poisson' else float(1 / (1 + np.exp(-linear))))
    local_models.append({'family': family, 'response': serial(observations), 'offset': serial(rounded(np.log(exposure))) if family == 'poisson' else None, 'coefficients': coefficients, 'fitted': predictions})

# Direct mgwr backfit, with an exact fixed bandwidth for each component.
bandwidths = [1.6, 2.8]
selector = Sel_BW(positions, response, predictors, multi=True, fixed=True, kernel='gaussian', n_jobs=1)
selector.search(init_multi=2.0, multi_bw_min=bandwidths, multi_bw_max=bandwidths, tol_multi=1e-10, max_iter_multi=100, verbose=False)
standard_errors = []
for focus in range(16):
    for column, bandwidth_value in enumerate(bandwidths):
        spatial = np.exp(-np.sum((positions - positions[focus])**2, axis=1) / (2 * bandwidth_value**2))
        values = np.ones(16) if column == 0 else predictors[:, 0]
        linear_weights = spatial * values / np.sum(spatial * values**2)
        standard_errors.append(float(np.linalg.norm(linear_weights) * np.sqrt(0.25)))
multiscale = {'bandwidths': bandwidths, 'coefficients': serial(selector.params.ravel()), 'conditionalStandardErrors': standard_errors}
masked_selector = Sel_BW(positions[mask], response[mask], predictors[mask], multi=True, fixed=True, kernel='gaussian', n_jobs=1)
masked_selector.search(init_multi=2.0, multi_bw_min=bandwidths, multi_bw_max=bandwidths, tol_multi=1e-10, max_iter_multi=100, verbose=False)
masked_coefficients = []
masked_row = 0
for row in range(16):
    if mask[row]:
        masked_coefficients.extend(serial(masked_selector.params[masked_row])); masked_row += 1
    else:
        masked_coefficients.extend([None, None])
multiscale['maskedCoefficients'] = masked_coefficients


# Candidate-grid backfit: independent statsmodels WLS solves for each candidate and fit.
# This is a composition oracle, not a direct mgwr automatic-bandwidth result.
grid_candidates = [[1.2, 1.6, 2.4], [1.6, 2.8, 4.0]]
design = sm.add_constant(predictors)
grid_coefficients = np.zeros_like(design)
grid_bandwidths = [0.0, 0.0]
for sweep in range(64):
    previous = grid_coefficients.copy()
    for column, candidates in enumerate(grid_candidates):
        partial = response.ravel() - np.sum(design * grid_coefficients, axis=1) + design[:, column] * grid_coefficients[:, column]
        scores = []
        for candidate in candidates:
            score = 0.0
            for focus in range(16):
                spatial = np.exp(-np.sum((positions - positions[focus])**2, axis=1)/(2*candidate**2))
                spatial[focus] = 0.0
                model = sm.WLS(partial, design[:, column], weights=spatial).fit()
                score += (partial[focus] - design[focus, column] * model.params[0])**2
            scores.append(score)
        selected = candidates[int(np.argmin(scores))]
        grid_bandwidths[column] = selected
        for focus in range(16):
            spatial = np.exp(-np.sum((positions - positions[focus])**2, axis=1)/(2*selected**2))
            grid_coefficients[focus, column] = sm.WLS(partial, design[:, column], weights=spatial).fit().params[0]
    change = np.max(abs(design*(grid_coefficients-previous))/np.maximum(1.0, abs(design*previous)))
    if change <= 1e-5:
        break
multiscale['gridCandidates'] = serial(np.array(grid_candidates).ravel())
multiscale['gridCoefficients'] = serial(grid_coefficients.ravel())
multiscale['gridBandwidths'] = grid_bandwidths

# Direct pooled spatial lag on the block diagonal entity matrix.
panel_predictors = rounded(np.concatenate([predictors.ravel(), predictors.ravel() * 0.7 + np.sin(np.arange(16)) * 0.3]))[:, None]
panel_response = rounded(np.concatenate([response.ravel(), response.ravel() * 0.8 + np.cos(np.arange(16)) * 0.4]))[:, None]
panel_weights = WSP(block_diag([weights.sparse, weights.sparse], format='csr')).to_W()
panel = GM_Lag(panel_response, panel_predictors, w=panel_weights, w_lags=1)
panel_fixture = {'predictors': serial(panel_predictors.ravel()), 'response': serial(panel_response.ravel()), 'coefficients': serial(panel.betas.ravel())}

regimes = [int(row >= 8) for row in range(16)]
regime_model = OLS_Regimes(response, predictors, regimes, constant_regi='many', cols2regi='all', regime_err_sep=False)
regime_fixture = {'regimes': regimes, 'coefficients': serial(regime_model.betas.ravel())}

# Pair composition references: SciPy distances and NumPy masked array reductions.
distances = cdist(positions, positions)
times = rounded(np.arange(16) * 0.25)
space_time = []
cross_values = []
for space_bin in range(3):
    lower = space_bin * 1.1
    upper = (space_bin + 1) * 1.1
    selection = np.triu((distances >= lower) & (distances < upper if space_bin < 2 else distances <= upper), 1)
    terms = (response - response.T) * (second[:, None] - second[None, :]) * 0.5
    cross_values.append(float(np.mean(terms[selection])) if np.any(selection) else None)
    for time_bin in range(3):
        temporal_distance = abs(times[:, None] - times[None, :])
        temporal = (temporal_distance >= time_bin * 1.5) & (temporal_distance < (time_bin + 1) * 1.5 if time_bin < 2 else temporal_distance <= (time_bin + 1) * 1.5)
        selected = selection & temporal
        space_time.append(float(np.mean(terms[selected])) if np.any(selected) else None)
pairs = {'secondValues': serial(second), 'times': serial(times), 'cross': cross_values, 'spaceTime': space_time, 'parameters': [3.3, 4.5]}
point_types = np.arange(16) % 2
marks = rounded(2.0 + np.cos(np.arange(16) * 0.4))
intensities = rounded(0.1 + np.arange(16) * 0.03)
focus_mask = point_types == 0
neighbor_mask = point_types == 1
category_pairs = focus_mask[:, None] & neighbor_mask[None, :]
delta = abs(positions[:, None, :] - positions[None, :, :])
overlap = (5.0 - delta[:, :, 0]) * (5.0 - delta[:, :, 1])
intensity_weights = 25.0 / overlap / intensities[:, None] / intensities[None, :]
mark_products = marks[:, None] * marks[None, :]
mark_mean = np.mean(marks[focus_mask]) * np.mean(marks[neighbor_mask])
marked_values = []
marked_counts = []
for radius_bin in range(3):
    lower = float(radius_bin)
    upper = lower + 1.0
    cumulative = category_pairs & (distances <= upper)
    shell = cumulative & (distances > lower)
    estimate = np.sum(intensity_weights[cumulative]) / 25.0
    pair_correlation = np.sum(intensity_weights[shell]) / (25.0 * np.pi * (upper**2 - lower**2))
    mark_correlation = np.sum((intensity_weights * mark_products)[shell]) / np.sum(intensity_weights[shell]) / mark_mean
    marked_values.extend([float(estimate), float(np.sqrt(estimate / np.pi)), float(pair_correlation), float(mark_correlation)])
    marked_counts.append(int(np.sum(cumulative)))
marked = {'types': serial(point_types), 'marks': serial(marks), 'intensities': serial(intensities), 'parameters': [-1.0, -1.0, 4.0, 4.0, 3.0, 0.0], 'statistics': marked_values, 'counts': marked_counts}


fixture = {
    'provenance': {'source': 'synthetic float32 inputs', 'versions': {package: version(package) for package in ['numpy', 'scipy', 'esda', 'libpysal', 'mgwr', 'spreg', 'statsmodels']}, 'tolerance': 0.002, 'referenceKinds': {'local': 'direct esda', 'generalized': 'statsmodels local weighted GLM composition', 'multiscale': 'direct mgwr constrained-bandwidth multi-backfit', 'conditionalInference': 'NumPy linear smoother composition', 'bandwidthGrid': 'statsmodels univariate WLS composition', 'panel': 'direct spreg.GM_Lag', 'regimes': 'direct spreg.OLS_Regimes'}},
    'positions': serial(positions.ravel()), 'predictors': serial(predictors.ravel()), 'response': serial(response.ravel()),
    'local': local, 'marked': marked, 'pairs': pairs, 'generalized': {'bandwidth': bandwidth, 'mask': serial(mask.astype(int)), 'models': local_models},
    'multiscale': multiscale, 'panel': panel_fixture, 'regimes': regime_fixture
}
# numpy integer values in offsets need conversion for JSON.
fixture['local']['offsets'] = [int(value) for value in fixture['local']['offsets']]
output = Path(__file__).parent / 'fixtures' / 'geospatial-expansion-statistics.json'
output.write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n')
print(output)
