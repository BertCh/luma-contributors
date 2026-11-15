# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Regenerate small SIM-01 fixtures: /usr/bin/python3 <this-file>.

Independent binary64 NumPy FFT / dense conditional covariance oracle. Random123 Philox
words follow Salmon et al. (2011); NumPy's Philox uses a different counter width.
Covariance conventions: this repository's effective ranges (correlation exp(-3)).
Randomization formula and radial CDFs verified against the GSTools primary sources:
https://geostat-framework.readthedocs.io/projects/gstools/en/stable/api/gstools.field.generator.RandMeth.html
https://geostat-framework.readthedocs.io/projects/gstools/en/stable/_modules/gstools/covmodel/models.html
FFT normalization: inverse divides by grid size; complex normals have variance one per part.
Conditioning: Zc = Z + Cqo Coo^-1 (data - Zo), with no measurement error.
No GSTools dependency, datasets, or model weights.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import chi2, norm

PURPOSE = 0x47524631
MASK = (1 << 32) - 1
FACTORS = {3: 4.749031386012701, 4: 5.924462618951797}


def philox(row, realization, tag, seed):
    counter = [row, realization, tag, 0]
    key = [seed, PURPOSE]
    for unused in range(10):
        first = counter[0] * 0xD2511F53
        second = counter[2] * 0xCD9E8D57
        counter = [(second >> 32) ^ counter[1] ^ key[0], second & MASK,
                   (first >> 32) ^ counter[3] ^ key[1], first & MASK]
        key = [(key[0] + 0x9E3779B9) & MASK, (key[1] + 0xBB67AE85) & MASK]
    return counter


def open_uniform(word):
    return ((word >> 9) + 0.5) / 8388608


def normal_pair(first, second):
    angle = 2 * np.pi * (second >> 8) / 16777216
    return np.sqrt(-2 * np.log(open_uniform(first))) * np.array([np.cos(angle), np.sin(angle)])


def correlation(distance, model, extent):
    scaled = distance / extent
    if model == 0:
        return np.where(scaled >= 1, 0, 1 - 1.5 * scaled + 0.5 * scaled**3)
    if model == 1:
        return np.exp(-3 * scaled)
    if model == 2:
        return np.exp(-3 * scaled**2)
    argument = FACTORS[model] * scaled
    return (1 + argument + (argument**2 / 3 if model == 4 else 0)) * np.exp(-argument)


def simulate(scenario):
    model, nugget, sill, extent = scenario['covariance']
    seed, realization = scenario['seed'], scenario['realization']
    if scenario['method'] == 'fft':
        width, height = scenario['dimensions']
        padded_width, padded_height = [2**int(np.ceil(np.log2(side * 2))) for side in (width, height)]
        rows, columns = np.indices((padded_height, padded_width))
        step_x, step_y = scenario['steps']
        distance = np.hypot(np.minimum(rows, padded_height - rows) * step_y,
                            np.minimum(columns, padded_width - columns) * step_x)
        covariance = sill * correlation(distance, model, extent)
        covariance[0, 0] += nugget
        eigenvalues = np.fft.fft2(covariance).real
        count = padded_width * padded_height
        draws = np.array([normal_pair(*philox(row, realization, 0, seed)[:2]) for row in range(count)])
        spectrum = np.sqrt(count * np.maximum(eigenvalues.ravel(), 0)) * (draws[:, 0] + 1j * draws[:, 1])
        field = np.fft.ifft2(spectrum.reshape(covariance.shape)).real[:height, :width].ravel()
        mass = np.maximum(-eigenvalues, 0).sum() / count
        positions = np.array([(column * step_x, row * step_y) for row in range(height) for column in range(width)])
    else:
        positions = np.array(scenario['positions'], dtype=float)
        field = np.zeros(len(positions))
        for mode in range(scenario.get('modeCount', 256)):
            words = philox(mode, realization, 1, seed)
            uniform = open_uniform(words[0])
            angle = 2 * np.pi * (words[1] >> 8) / 16777216
            if model == 2:
                radius = np.sqrt(-12 * np.log(uniform)) / extent
            else:
                smoothness = {1: 0.5, 3: 1.5, 4: 2.5}[model]
                radius = {1: 3, **FACTORS}[model] * np.sqrt(uniform**(-1 / smoothness) - 1) / extent
            wave = radius * np.array([np.cos(angle), np.sin(angle)])
            amplitudes = normal_pair(words[2], words[3])
            phases = positions @ wave
            field += amplitudes[0] * np.cos(phases) + amplitudes[1] * np.sin(phases)
        field *= np.sqrt(sill / scenario.get('modeCount', 256))
        field += np.sqrt(nugget) * np.array([normal_pair(*philox(row, realization, 2, seed)[:2])[0] for row in range(len(positions))])
        mass = 0
    field += scenario['mean']
    if 'conditioning' in scenario:
        conditional = scenario['conditioning']
        indices = np.array(conditional['indices'])
        values = np.array(conditional['values'])
        masks = np.array(conditional.get('mask', [1] * len(indices)))
        residual = values - field[indices]
        for row, candidates in enumerate(conditional['neighborhoods']):
            candidates = np.array([value for value in candidates if value != MASK and masks[value]])
            if not len(candidates):
                continue
            observed_positions = positions[indices[candidates]]
            distances = np.linalg.norm(observed_positions[:, None] - observed_positions, axis=2)
            matrix = sill * correlation(distances, model, extent) + nugget * (distances == 0)
            query_distance = np.linalg.norm(observed_positions - positions[row], axis=1)
            right = sill * correlation(query_distance, model, extent) + nugget * (query_distance == 0)
            field[row] += np.linalg.solve(matrix, right) @ residual[candidates]
    return field.tolist(), float(mass)


def verify_statistics():
    """Independent NumPy draws, not Philox or the kernel, verify covariance normalization."""
    generator = np.random.default_rng(20261114)
    sample_count = 8192
    mode_count = 256
    for model in (1, 2, 3, 4):
        expected = 1 - correlation(np.array(1.0), model, 2.5)
        # Independent complex white noise, then the FFT square root of the covariance.
        rows, columns = np.indices((16, 16))
        distances = np.hypot(np.minimum(rows, 16 - rows), np.minimum(columns, 16 - columns))
        eigenvalues = np.fft.fft2(correlation(distances, model, 2.5)).real
        noise = generator.normal(size=(sample_count, 16, 16, 2))
        fields = np.fft.ifft2(np.sqrt(256 * np.maximum(eigenvalues, 0))[None] * (noise[..., 0] + 1j * noise[..., 1])).real
        fft_pair = fields[:, 0, :2]
        # Radial inversion and uniform angle are checked with independent NumPy normal amplitudes.
        uniforms = generator.uniform(size=(sample_count, mode_count))
        if model == 2:
            radii = np.sqrt(-12 * np.log(uniforms)) / 2.5
        else:
            smoothness = {1: 0.5, 3: 1.5, 4: 2.5}[model]
            radii = {1: 3, **FACTORS}[model] * np.sqrt(uniforms**(-1 / smoothness) - 1) / 2.5
        phases = radii * np.cos(generator.uniform(0, 2 * np.pi, size=uniforms.shape))
        amplitudes = generator.normal(size=(sample_count, mode_count, 2))
        spectral_pair = np.column_stack((amplitudes[..., 0].sum(axis=1),
            (amplitudes[..., 0] * np.cos(phases) + amplitudes[..., 1] * np.sin(phases)).sum(axis=1))) / np.sqrt(mode_count)
        for method_name, pair in [('fft', fft_pair), ('spectral', spectral_pair)]:
            mean = pair[:, 0].mean()
            variogram = np.mean((pair[:, 0] - pair[:, 1])**2) / 2
            if abs(mean) > 6 / np.sqrt(sample_count) or abs(variogram / expected - 1) > 0.12:
                raise RuntimeError(f'{method_name} model {model}: mean={mean}, variogram={variogram}, expected={expected}')
    # Dense residual correction: exact data and the independent Schur-complement covariance.
    positions = np.array([[0, 0], [1, 0], [2, 1], [0, 2]])
    distances = np.linalg.norm(positions[:, None] - positions, axis=2)
    covariance = np.exp(-3 * distances / 2.5)
    prior = generator.multivariate_normal(np.zeros(4), covariance, size=sample_count)
    weights = np.linalg.solve(covariance[:2, :2], covariance[:2]).T
    data = np.array([2.0, -1.0])
    conditioned = prior + (data - prior[:, :2]) @ weights.T
    expected_covariance = covariance - weights @ covariance[:2]
    if np.max(np.abs(conditioned[:, :2] - data)) > 1e-13:
        raise RuntimeError('Dense residual conditioning did not reproduce data')
    if np.max(np.abs(np.cov(conditioned.T) - expected_covariance)) > 0.04:
        raise RuntimeError('Dense residual conditioning covariance differs from Schur complement')
    print('NumPy: 8192 realizations per model/method pass mean, variogram and conditioning checks')


def main():
    verify_statistics()
    scenarios = []
    for model in range(5):
        scenario = dict(name=f'fft-model-{model}', method='fft', dimensions=[4, 3], steps=[1, -0.75],
                        covariance=[model, 0.1, 1.2, 2.5], seed=4294967283, realization=19, mean=0.75)
        scenarios.append(scenario)
    scenarios.append(dict(name='fft-clipped-long-range', method='fft', dimensions=[4, 3], steps=[1, -0.75],
                          covariance=[2, 0, 1, 20], seed=81, realization=0, mean=0))
    for model in (1, 2, 3, 4):
        scenarios.append(dict(name=f'spectral-model-{model}', method='spectral',
                              positions=[[0, 0], [1, 0], [0, 1], [0.5, 0.5], [2, -1]], modeCount=64,
                              covariance=[model, 0.2, 1.2, 2.5], seed=121, realization=3, mean=0.75))
    for method in ('fft', 'spectral'):
        scenario = dict(scenarios[1 if method == 'fft' else 7])
        count = 12 if method == 'fft' else 5
        scenario.update(name=f'{method}-conditioned', conditioning=dict(indices=[0, 2, 4], values=[2, -1, 999],
                        mask=[1, 1, 0], neighborhoods=[[0, 1, 2, MASK] if row != count - 1 else [MASK] * 4 for row in range(count)]))
        scenarios.append(scenario)
    scenarios.append(dict(name='spectral-pure-nugget-tiny-range', method='spectral',
                          positions=[[0, 0]], covariance=[1, 1, 0, 5e-38],
                          seed=121, realization=0, mean=0))
    for scenario in scenarios:
        scenario['expected'], scenario['clippedEigenvalueMass'] = simulate(scenario)
    # Independent Monte Carlo sanity check of both numerical constructions, using NumPy's RNG.
    generator = np.random.default_rng(20261114)
    covariance = np.exp(-3 * np.array([[0, 1], [1, 0]]) / 2.5)
    samples = generator.multivariate_normal([0, 0], covariance, size=100000)
    empirical_variogram = np.mean((samples[:, 0] - samples[:, 1])**2) / 2
    if abs(empirical_variogram - (1 - np.exp(-3 / 2.5))) >= 0.015:
        raise RuntimeError('NumPy covariance Monte Carlo variogram check failed')
    realization_count = 256
    confidence = 1e-6
    bounds = dict(realizationCount=realization_count,
                  meanTolerance=float(norm.ppf(1 - confidence / 2) / np.sqrt(realization_count)),
                  variogramRatio=[float(chi2.ppf(confidence / 2, realization_count) / realization_count),
                                  float(chi2.ppf(1 - confidence / 2, realization_count) / realization_count)])
    destination = Path(__file__).parent / 'fixtures/gaussian-random-field.json'
    destination.write_text(json.dumps(dict(scenarios=scenarios, statistics=bounds), indent=2) + '\n')
    print(f'Wrote {len(scenarios)} cases; NumPy variogram {empirical_variogram:.6f}')


if __name__ == '__main__':
    main()
