// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUVariogramModelFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {
  fitVariogramModel,
  getVariogramModelShape
} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';
import {InterpolationBenchmarkScope} from '../../gpu-spatial-analysis/spatial-interpolation/interpolation-performance-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';

for (const weighting of ['pairs', 'none', 'cressie'] as const) {
  it(`variogram ${weighting} replay clears failures and preserves model outputs after changed bins`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scope = new InterpolationBenchmarkScope(device, `variogram-replay-${weighting}`);
    const distances = Float32Array.from({length: 20}, (_, index) => (index + 1) * 0.2);
    const pairCounts = Uint32Array.from({length: 20}, (_, index) => 10 + index);
    const models = ['gaussian', 'exponential', 'spherical'] as const;
    const observations = Float32Array.from(
      distances,
      distance => 0.2 + 1.5 * (1 - Math.exp((-3 * distance) / 2.2))
    );
    scope.graph.add(
      new GPUVariogramModelFit({
        distances: scope.input('distances', distances, 'float32'),
        semivariances: scope.input('semivariances', observations, 'float32'),
        pairCounts: scope.input('pairs', pairCounts, 'uint32'),
        models,
        weighting,
        output: {
          parameters: scope.output('parameters', 3, 'float32x4'),
          diagnostics: scope.output('diagnostics', 3, 'float32x4'),
          status: scope.output('status', 3, 'uint32')
        }
      })
    );
    const compiled = scope.graph.compile();
    try {
      scope.initialWrites[0].buffer.write(distances);
      for (const scenario of ['initial', 'overflow', 'missing', 'changed', 'initial'] as const) {
        const values = Float32Array.from(observations, (value, index) =>
          scenario === 'overflow'
            ? (1 + (index % 2)) * 1e30
            : scenario === 'changed'
              ? value * 1.5
              : value
        );
        const counts = scenario === 'missing' ? new Uint32Array(20) : pairCounts;
        scope.initialWrites[1].buffer.write(values);
        scope.initialWrites[2].buffer.write(counts);
        submitGraph(device, compiled, undefined);
        const result = await scope.read();
        if (scenario === 'missing' || scenario === 'overflow') {
          expect(result['status']).toEqual(Array(3).fill(scenario === 'missing' ? 1 : 2));
          expect(result['parameters'].every(Number.isNaN)).toBe(true);
          expect(result['diagnostics'].every(Number.isNaN)).toBe(true);
        } else {
          expect(result['status'].every(value => value === 0 || value === 3)).toBe(true);
          for (let modelIndex = 0; modelIndex < models.length; modelIndex++) {
            const expected = fitVariogramModel(
              {
                distances: Array.from(distances),
                semivariances: Array.from(values),
                pairCounts: Array.from(counts)
              },
              {model: models[modelIndex], weighting}
            );
            expect(result['diagnostics'][modelIndex * 4]).toBe(20);
            expect(
              result['parameters'].slice(modelIndex * 4, modelIndex * 4 + 4).every(Number.isFinite)
            ).toBe(true);
            expect(
              Math.abs(result['parameters'][modelIndex * 4 + 3] - expected.residual)
            ).toBeLessThan(0.002 * Math.max(1, Math.abs(expected.residual)));
          }
        }
      }
    } finally {
      compiled.destroy();
      scope.destroy();
    }
  });
}

it('variogram cressie fit is unchanged when the semivariances are scaled to 1e-8', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Noise-free bins of an exponential model: nugget 0.2, partial sill 1.5, practical range 2.2.
  // Cressie weights are N / fitted^2, so scaling every semivariance by a constant scales every
  // weight by one factor and the weighted least-squares solution scales with the data: the
  // nugget and sill of the scaled bins are the scale times those of the unscaled bins and the
  // range is the same. At a scale of 1e-8 with about 1e4 pairs per bin the weights are near
  // 1e20 and their normal-equation products leave float32 unless they are normalised.
  const truth = {nugget: 0.2, sill: 1.5, range: 2.2};
  const distances = Float32Array.from({length: 20}, (_, index) => (index + 1) * 0.2);
  const pairCounts = Uint32Array.from({length: 20}, (_, index) => 10000 + 100 * index);
  const unscaled = Float32Array.from(
    distances,
    distance => truth.nugget + truth.sill * (1 - Math.exp((-3 * distance) / truth.range))
  );
  const fits: number[][] = [];
  // 1e12 checks the other side: large semivariances whose squared float32 resolution is still
  // finite are fitted, not refused by the representability guard of the search.
  for (const scale of [1, 1e-8, 1e12]) {
    const scope = new InterpolationBenchmarkScope(device, `variogram-scale-${scale}`);
    scope.graph.add(
      new GPUVariogramModelFit({
        distances: scope.input('distances', distances, 'float32'),
        semivariances: scope.input(
          'semivariances',
          Float32Array.from(unscaled, value => value * scale),
          'float32'
        ),
        pairCounts: scope.input('pairs', pairCounts, 'uint32'),
        models: ['exponential'],
        weighting: 'cressie',
        output: {
          parameters: scope.output('parameters', 1, 'float32x4'),
          diagnostics: scope.output('diagnostics', 1, 'float32x4'),
          status: scope.output('status', 1, 'uint32')
        }
      })
    );
    const compiled = scope.graph.compile();
    try {
      for (const write of scope.initialWrites) {
        write.buffer.write(write.values);
      }
      submitGraph(device, compiled, undefined);
      const result = await scope.read();
      expect(result['status'], `scale ${scale}`).toEqual([0]);
      const [nugget, sill, range] = result['parameters'];
      fits.push([nugget / scale, sill / scale, range]);
    } finally {
      compiled.destroy();
      scope.destroy();
    }
  }
  for (const [label, fit] of [
    ['unscaled', fits[0]],
    ['scaled by 1e-8', fits[1]],
    ['scaled by 1e12', fits[2]]
  ] as const) {
    // The bins are exact model values, so the fit recovers the model.
    expect(Math.abs(fit[0] - truth.nugget), `${label} nugget ${fit[0]}`).toBeLessThan(0.02);
    expect(Math.abs(fit[1] - truth.sill), `${label} sill ${fit[1]}`).toBeLessThan(0.02);
    expect(Math.abs(fit[2] - truth.range), `${label} range ${fit[2]}`).toBeLessThan(0.03);
  }
});

it('runtime variogram round slots retain the closed-form exponential fit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const distances = Float32Array.from({length: 32}, (_, index) => (index + 1) / 8);
  const truth = {nugget: 0.2, sill: 1.5, range: 2.2};
  const values = Float32Array.from(
    distances,
    distance => truth.nugget + truth.sill * (1 - Math.exp((-3 * distance) / truth.range))
  );
  for (const weighting of ['pairs', 'cressie'] as const) {
    const scope = new InterpolationBenchmarkScope(device, `runtime-rounds-${weighting}`);
    scope.graph.add(
      new GPUVariogramModelFit({
        distances: scope.input('distances', distances, 'float32'),
        semivariances: scope.input('semivariances', values, 'float32'),
        pairCounts: scope.input('counts', new Uint32Array(32).fill(10), 'uint32'),
        models: ['exponential'],
        weighting,
        rangeSteps: 128,
        output: {
          parameters: scope.output('parameters', 1, 'float32x4'),
          diagnostics: scope.output('diagnostics', 1, 'float32x4'),
          status: scope.output('status', 1, 'uint32')
        }
      })
    );
    const compiled = scope.graph.compile();
    try {
      for (const write of scope.initialWrites) write.buffer.write(write.values);
      // Repeated encoding also checks that graph-owned parameter slots are restored each frame.
      for (let frame = 0; frame < 2; frame++) {
        submitGraph(device, compiled, undefined);
        const result = await scope.read();
        expect(result['status']).toEqual([0]);
        expect(Math.abs(result['parameters'][0] - truth.nugget)).toBeLessThan(0.02);
        expect(Math.abs(result['parameters'][1] - truth.sill)).toBeLessThan(0.02);
        expect(Math.abs(result['parameters'][2] - truth.range)).toBeLessThan(0.03);
        expect(result['diagnostics'][0]).toBe(32);
      }
    } finally {
      compiled.destroy();
      scope.destroy();
    }
  }
});

it('Cressie slots reweight noisy unequal-count bins on changed and repeated frames', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scope = new InterpolationBenchmarkScope(device, 'noisy-cressie-replay');
  const distances = Float32Array.from({length: 32}, (_, index) => (index + 1) / 8);
  const counts = Uint32Array.from({length: 32}, (_, index) => 10 + (index % 4) * 50);
  const original = Float32Array.from(
    distances,
    (distance, index) =>
      0.2 + 1.5 * (1 - Math.exp((-3 * distance) / 2.2)) + 0.2 * Math.sin(index * 1.7)
  );
  scope.graph.add(
    new GPUVariogramModelFit({
      distances: scope.input('distances', distances, 'float32'),
      semivariances: scope.input('semivariances', original, 'float32'),
      pairCounts: scope.input('pairs', counts, 'uint32'),
      models: ['exponential'],
      weighting: 'cressie',
      output: {
        parameters: scope.output('parameters', 1, 'float32x4'),
        diagnostics: scope.output('diagnostics', 1, 'float32x4'),
        status: scope.output('status', 1, 'uint32')
      }
    })
  );
  const compiled = scope.graph.compile();
  try {
    for (const write of scope.initialWrites) write.buffer.write(write.values);
    for (const scale of [1, 1.4, 1]) {
      const observations = Float32Array.from(original, value => value * scale);
      scope.initialWrites[1].buffer.write(observations);
      const input = {distances, semivariances: observations, pairCounts: counts};
      const expected = fitVariogramModel(input, {model: 'exponential', weighting: 'cressie'});
      const pairFit = fitVariogramModel(input, {model: 'exponential', weighting: 'pairs'});
      // This fixture must reject an implementation that never enables reweighting.
      expect(Math.abs(expected.range - pairFit.range)).toBeGreaterThan(0.1);
      submitGraph(device, compiled, undefined);
      const actual = await scope.read();
      expect(actual['status'][0]).toBe(0);
      for (const [index, value] of [
        expected.nugget,
        expected.sill,
        expected.range,
        expected.residual
      ].entries()) {
        expect(Math.abs(actual['parameters'][index] - value)).toBeLessThan(
          0.02 * Math.max(1, Math.abs(value))
        );
      }
      const unweighted = Array.from(
        distances,
        (distance, index) =>
          (observations[index] -
            actual['parameters'][0] -
            actual['parameters'][1] *
              getVariogramModelShape('exponential', distance / actual['parameters'][2])) **
          2
      ).reduce((total, value) => total + value, 0);
      expect(Math.abs(actual['diagnostics'][1] - unweighted)).toBeLessThan(
        2e-5 * Math.max(1, unweighted)
      );
    }
  } finally {
    compiled.destroy();
    scope.destroy();
  }
});

it('zero variogram refinement budget publishes finite candidates and budget status', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scope = new InterpolationBenchmarkScope(device, 'zero-refinement-budget');
  const distances = Float32Array.from({length: 16}, (_, index) => (index + 1) / 4);
  const observations = Float32Array.from(
    distances,
    distance => 0.2 + 1.5 * (1 - Math.exp((-3 * distance) / 2.2))
  );
  scope.graph.add(
    new GPUVariogramModelFit({
      distances: scope.input('distances', distances, 'float32'),
      semivariances: scope.input('semivariances', observations, 'float32'),
      pairCounts: scope.input('pairs', new Uint32Array(16).fill(10), 'uint32'),
      models: ['exponential'],
      maximumIterations: 0,
      weighting: 'cressie',
      output: {
        parameters: scope.output('parameters', 1, 'float32x4'),
        diagnostics: scope.output('diagnostics', 1, 'float32x4'),
        status: scope.output('status', 1, 'uint32')
      }
    })
  );
  const compiled = scope.graph.compile();
  try {
    for (const write of scope.initialWrites) write.buffer.write(write.values);
    submitGraph(device, compiled, undefined);
    const result = await scope.read();
    expect(result['status']).toEqual([3]);
    expect(result['parameters'].every(Number.isFinite)).toBe(true);
    expect(result['diagnostics'][0]).toBe(16);
    expect(result['diagnostics'][3]).toBeGreaterThan(1e-5);
    expect(Math.abs(result['parameters'][0] - 0.2)).toBeLessThan(0.03);
    expect(Math.abs(result['parameters'][1] - 1.5)).toBeLessThan(0.03);
    expect(Math.abs(result['parameters'][2] - 2.2)).toBeLessThan(0.06);
  } finally {
    compiled.destroy();
    scope.destroy();
  }
});
