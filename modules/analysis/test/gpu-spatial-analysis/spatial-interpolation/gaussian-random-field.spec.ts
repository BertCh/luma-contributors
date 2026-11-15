// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import fixture from './fixtures/gaussian-random-field.json';
import {createFieldRun, type FieldScenario} from './gaussian-random-field-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scenario of fixture.scenarios) {
  it(`Gaussian random field matches NumPy: ${scenario.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const run = createFieldRun(device, scenario);
    try {
      const values = await run.execute();
      // Binary64 NumPy vs f32 FFT, trig, local solve and bounded accumulation.
      scenario.expected.forEach((expected, index) =>
        expect(values[index]).toBeCloseTo(expected, 3)
      );
      const status = await run.status();
      expect(status.validity).toEqual([...scenario.expected.map(() => 1), 0, 0]);
      expect(values.slice(-2).every(Number.isNaN)).toBe(true);
      expect(status.mass).toBeCloseTo(scenario.clippedEigenvalueMass, 5);
      expect(status.approximation).toBe(
        scenario.method === 'spectral' || 'conditioning' in scenario || status.mass > 0 ? 1 : 0
      );
      if (scenario.conditioning) {
        scenario.conditioning.indices.forEach((row, index) => {
          if (scenario.conditioning.mask[index])
            expect(values[row]).toBe(Math.fround(scenario.conditioning.values[index]!));
        });
      }
      expect(await run.execute()).toEqual(values);
      expect(await run.execute(scenario.realization + 1)).not.toEqual(values);
    } finally {
      run.destroy();
    }
  });
}

for (const method of ['fft', 'spectral'] as const) {
  for (const model of [1, 2]) {
    it(`Gaussian random field ${method} model ${model}: zero mean and variogram within confidence bounds`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const scenario: FieldScenario = {
        name: `statistics-${method}-${model}`,
        method,
        dimensions: [8, 8],
        steps: [1, 1],
        covariance: [model, 0, 1, 2.5],
        modeCount: 256,
        seed: 20261114,
        realization: 0,
        mean: 0
      };
      const run = createFieldRun(device, scenario);
      try {
        let mean = 0;
        let variogram = 0;
        const {realizationCount, meanTolerance, variogramRatio} = fixture.statistics;
        for (let realization = 0; realization < realizationCount; realization++) {
          const values = await run.execute(realization);
          mean += values[0]! / realizationCount;
          variogram += (values[0]! - values[1]!) ** 2 / (2 * realizationCount);
        }
        const expected = 1 - Math.exp(-3 * (model === 1 ? 1 / 2.5 : 1 / 2.5 ** 2));
        expect(Math.abs(mean)).toBeLessThan(meanTolerance);
        expect(variogram / expected).toBeGreaterThan(variogramRatio[0]!);
        expect(variogram / expected).toBeLessThan(variogramRatio[1]!);
      } finally {
        run.destroy();
      }
    }, 120_000);
  }
}

it('Gaussian random field clears empty point outputs and unwritten parameter results', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const positions of [
    [],
    [
      [0, 0],
      [1, 0]
    ]
  ]) {
    const run = createFieldRun(device, {
      name: 'empty-or-unwritten',
      method: 'spectral',
      positions,
      covariance: [2, 0, 1, 1],
      seed: 0,
      realization: 0,
      mean: 0
    });
    try {
      run.parameterBuffer.write(new Float32Array(16));
      expect((await run.execute()).every(Number.isNaN)).toBe(true);
      expect((await run.status()).validity.every(value => value === 0)).toBe(true);
    } finally {
      run.destroy();
    }
  }
});

it('Gaussian random field conditions more than 4096 observed rows with bounded indexed solves', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const observationCount = 5000;
  const rowCount = 128 * 64;
  const scenario: FieldScenario = {
    name: 'large-conditioned',
    method: 'fft',
    dimensions: [128, 64],
    steps: [1, -1],
    covariance: [1, 0, 1, 2.5],
    seed: 41,
    realization: 2,
    mean: 0
  };
  const prior = createFieldRun(device, {...scenario, name: 'large-prior'});
  const conditioned = createFieldRun(device, {
    ...scenario,
    conditioning: {
      indices: Array.from({length: observationCount}, (_, index) => index),
      values: Array.from({length: observationCount}, (_, index) => Math.fround(Math.sin(index))),
      neighborhoods: Array.from({length: rowCount}, (_, index) => [
        Math.min(index, observationCount - 1)
      ])
    }
  });
  try {
    const unconditional = await prior.execute();
    const values = await conditioned.execute();
    for (let index = 0; index < observationCount; index++)
      expect(values[index]).toBe(Math.fround(Math.sin(index)));
    const source = observationCount - 1;
    for (const index of [5000, 6000, 8191]) {
      const distance = Math.hypot(
        (index % 128) - (source % 128),
        Math.floor(index / 128) - Math.floor(source / 128)
      );
      const expected =
        unconditional[index]! +
        Math.exp((-3 * distance) / 2.5) * (Math.fround(Math.sin(source)) - unconditional[source]!);
      expect(values[index]).toBeCloseTo(expected, 4);
    }
  } finally {
    prior.destroy();
    conditioned.destroy();
  }
});

it('Gaussian random field updates covariance parameters and invalidates an unsupported spectral model', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const run = createFieldRun(device, {
    name: 'per-frame-model',
    method: 'spectral',
    positions: [
      [0, 0],
      [1, 0]
    ],
    covariance: [2, 0, 1, 2.5],
    seed: 9,
    realization: 0,
    mean: 0
  });
  try {
    const original = await run.execute();
    const {getGPUGaussianRandomFieldParameterValues} = await import(
      '../../../src/gpu-spatial-analysis/spatial-interpolation/index'
    );
    run.parameterBuffer.write(
      getGPUGaussianRandomFieldParameterValues({
        ...run.settings,
        variogram: {model: 'exponential', sill: 2, range: 4},
        mean: 5
      })
    );
    expect(await run.execute()).not.toEqual(original);
    run.parameterBuffer.write(
      getGPUGaussianRandomFieldParameterValues({
        ...run.settings,
        variogram: {model: 'spherical', sill: 1, range: 4}
      })
    );
    expect((await run.execute()).every(Number.isNaN)).toBe(true);
    expect((await run.status()).validity).toEqual([0, 0, 0, 0]);
  } finally {
    run.destroy();
  }
});

for (const method of ['spectral', 'fft'] as const) {
  for (const conditioned of [false, true]) {
    it(`Gaussian random field ${method} preserves subcell spacing at large grid origins${conditioned ? ' with conditioning' : ''}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const scenario: FieldScenario = {
        name: `translation-${method}-${conditioned}`,
        method,
        dimensions: [3, 1],
        steps: [0.1, -0.1],
        covariance: [1, 0, 1, 1],
        modeCount: 256,
        seed: 121,
        realization: 3,
        mean: 0,
        conditioning: conditioned
          ? {
              indices: [0, 2],
              values: [2, -1],
              neighborhoods: [
                [0, 1],
                [0, 1],
                [0, 1]
              ]
            }
          : undefined
      };
      const local = createFieldRun(device, scenario);
      const translated = createFieldRun(device, {...scenario, gridOrigin: [5000000, 6000000]});
      try {
        const expected = await local.execute();
        const actual = await translated.execute();
        expect(expected.slice(0, 3).every(Number.isFinite)).toBe(true);
        expect(actual).toEqual(expected);
        expect(await translated.status()).toEqual(await local.status());
        expect(new Set(actual.slice(0, 3)).size).toBeGreaterThan(1);
        if (conditioned) {
          expect(actual[0]).toBe(2);
          expect(actual[2]).toBe(-1);
          expect((await translated.status()).validity).toEqual([1, 1, 1, 0, 0]);
        }
      } finally {
        local.destroy();
        translated.destroy();
      }
    });
  }
}

it('Gaussian random field pure nugget stays finite and range-independent when spectral frequencies would overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const run = createFieldRun(device, {
    name: 'pure-nugget-tiny-range',
    method: 'spectral',
    positions: [[0, 0]],
    covariance: [1, 1, 0, 5e-38],
    seed: 121,
    realization: 0,
    mean: 0
  });
  try {
    // modeCount is deliberately omitted: this is the reviewer's default-mode regression.
    const values = await run.execute();
    expect(Number.isFinite(values[0])).toBe(true);
    expect((await run.status()).validity).toEqual([1, 0, 0]);
    const {getGPUGaussianRandomFieldParameterValues} = await import(
      '../../../src/gpu-spatial-analysis/spatial-interpolation/index'
    );
    run.parameterBuffer.write(
      getGPUGaussianRandomFieldParameterValues({
        ...run.settings,
        variogram: {...run.settings.variogram, range: 1}
      })
    );
    expect(await run.execute()).toEqual(values);
  } finally {
    run.destroy();
  }
});

it('Gaussian random field reports unrepresentable positive-sill frequencies as invalid and recovers after a parameter update', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const run = createFieldRun(device, {
    name: 'overflowing-frequencies',
    method: 'spectral',
    positions: [[0, 0]],
    covariance: [1, 0, 1, 5e-38],
    seed: 121,
    realization: 0,
    mean: 0
  });
  try {
    // The first frequency overflows for these Philox words, even though its point is the origin.
    expect((await run.execute()).every(Number.isNaN)).toBe(true);
    expect((await run.status()).validity).toEqual([0, 0, 0]);
    const {getGPUGaussianRandomFieldParameterValues} = await import(
      '../../../src/gpu-spatial-analysis/spatial-interpolation/index'
    );
    run.parameterBuffer.write(
      getGPUGaussianRandomFieldParameterValues({
        ...run.settings,
        variogram: {...run.settings.variogram, range: 1}
      })
    );
    expect(Number.isFinite((await run.execute())[0])).toBe(true);
    expect((await run.status()).validity).toEqual([1, 0, 0]);
  } finally {
    run.destroy();
  }
});

it('Gaussian random field rejects a nonfinite spectral phase without evaluating trigonometry on it', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const run = createFieldRun(device, {
    name: 'overflowing-phase',
    method: 'spectral',
    positions: [[3e38, 3e38]],
    covariance: [1, 0, 1, 1],
    seed: 121,
    realization: 0,
    mean: 0
  });
  try {
    expect((await run.execute()).every(Number.isNaN)).toBe(true);
    expect((await run.status()).validity).toEqual([0, 0, 0]);
  } finally {
    run.destroy();
  }
});
