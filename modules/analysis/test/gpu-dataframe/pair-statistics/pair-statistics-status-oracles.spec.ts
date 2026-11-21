// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialCorrelogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-spatial-correlogram';
import {getGPUSpatialCorrelogramParameterValues} from '../../../src/dataframe-analysis/pair-statistics/spatial-correlogram-parameters';
import {GPUVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram';
import {
  GPUSpaceTimeVariogram,
  getGPUSpaceTimeVariogramParameterValues
} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-variogram';
import {getGPUVariogramParameterValues} from '../../../src/dataframe-analysis/pair-statistics/variogram-parameters';
import {GPUMarkedPointStatistics} from '../../../src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics';
import {GPURipleyEnvelopeOperation} from '../../../src/dataframe-analysis/pair-statistics/gpu-ripley-envelope-operation';
import {getGPURipleyParameterValues} from '../../../src/dataframe-analysis/pair-statistics/ripley-parameters';
import type {GPUContributorOperationViewGetter} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {makeSpatialFixture, readIntegers, readFloats} from '../../spatial-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

it('correlogram rewrites poisoned overflow for a bounded three-pair inventory', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const coordinates = fixture.view(
      'float32',
      new Float32Array([0.25, 0.25, 0.75, 0.25, 0.25, 0.75])
    );
    const positions = fixture.graph.createDataView(coordinates.view.buffer, {
      format: 'float32x2',
      length: 3
    });
    const values = fixture.view('float32', new Float32Array([1, 2, 3]));
    const parameters = fixture.view(
      'float32',
      getGPUSpatialCorrelogramParameterValues({bounds: [0, 0, 1, 1], maximumDistance: 2})
    );
    const moransI = fixture.view('float32', new Float32Array(1));
    const pairCounts = fixture.view('uint32', new Uint32Array([99]));
    const overflow = fixture.view('uint32', new Uint32Array([0xffffffff]));
    fixture.graph.add(
      new GPUSpatialCorrelogram({
        positions,
        values: values.view,
        parameters: parameters.view,
        gridSize: [2, 2],
        bandCount: 1,
        output: {
          moransI: moransI.view,
          pairCounts: pairCounts.view,
          status: {overflow: overflow.view}
        }
      })
    );
    // Three distinct points give C(3,2)=3 unordered pairs, all within radius 2.
    // Centered values [-1,0,1] have sum of squares 2: all products are
    // bounded by 1 and three terms cannot approach the documented 2^40
    // variance-unit sum bound or uint32 pair-count saturation. Whole word 0.
    for (let encoding = 0; encoding < 2; encoding++) {
      overflow.buffer.write(new Uint32Array([0xffffffff]));
      await fixture.run();
      expect(await readIntegers(pairCounts.buffer)).toEqual([3]);
      expect(await readIntegers(overflow.buffer)).toEqual([0]);
    }
  } finally {
    fixture.destroy();
  }
});

it('variogram rewrites poisoned overflow independently of the semivariance reference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const coordinates = fixture.view(
      'float32',
      new Float32Array([0.25, 0.25, 0.75, 0.25, 0.25, 0.75])
    );
    const positions = fixture.graph.createDataView(coordinates.view.buffer, {
      format: 'float32x2',
      length: 3
    });
    const values = fixture.view('float32', new Float32Array([1, 2, 3]));
    const parameters = fixture.view(
      'float32',
      getGPUVariogramParameterValues({bounds: [0, 0, 1, 1], maximumDistance: 2})
    );
    const semivariances = fixture.view('float32', new Float32Array(1));
    const pairCounts = fixture.view('uint32', new Uint32Array([99]));
    const overflow = fixture.view('uint32', new Uint32Array([0xffffffff]));
    fixture.graph.add(
      new GPUVariogram({
        positions,
        values: values.view,
        parameters: parameters.view,
        gridSize: [2, 2],
        lagCount: 1,
        output: {
          semivariances: semivariances.view,
          pairCounts: pairCounts.view,
          status: {overflow: overflow.view}
        }
      })
    );
    // C(3,2)=3 pairs; semivariance terms are 1/2, 2 and 1/2.
    // Population variance is 2/3. Three bounded terms cannot exceed the
    // documented 2^40 variance-unit sum bound or saturate a uint32 count.
    for (let encoding = 0; encoding < 2; encoding++) {
      overflow.buffer.write(new Uint32Array([0xffffffff]));
      await fixture.run();
      expect(await readIntegers(pairCounts.buffer)).toEqual([3]);
      expect(await readIntegers(overflow.buffer)).toEqual([0]);
    }
  } finally {
    fixture.destroy();
  }
});

it('marked-point status clears poison and replays a four-pair pattern through singleton invalidity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const coordinates = fixture.view('float32', new Float32Array([1, 1, 2, 1, 3, 1, 1, 3]));
    const positions = fixture.graph.createDataView(coordinates.view.buffer, {
      format: 'float32x2',
      length: 4
    });
    const mask = fixture.view('uint32', new Uint32Array([1, 1, 1, 1]));
    const parameters = fixture.view('float32', new Float32Array([0, 0, 4, 4, 1.5, 0]));
    const statistics = fixture.view('float32', new Float32Array(4));
    const pairCounts = fixture.view('uint32', new Uint32Array([99]));
    const status = fixture.view('uint32', new Uint32Array([0xffffffff]));
    fixture.graph.add(
      new GPUMarkedPointStatistics({
        positions,
        mask: mask.view,
        parameters: parameters.view,
        radiusCount: 1,
        edgeCorrection: 'none',
        output: {statistics: statistics.view, pairCounts: pairCounts.view, status: status.view}
      })
    );
    // Only the two length-one horizontal edges qualify. Each contributes
    // both directions: four pairs; n(n-1)=12 is nonzero, all arithmetic
    // bounded, so the whole status code is 0 (valid estimate).
    await fixture.run();
    expect(await readIntegers(pairCounts.buffer)).toEqual([4]);
    expect(await readIntegers(status.buffer)).toEqual([0]);
    // One retained event cannot supply any non-self pair or same-type K:
    // denominator n(n-1)=0 forces code 1, independently of any solver.
    mask.buffer.write(new Uint32Array([1, 0, 0, 0]));
    await fixture.run();
    expect(await readIntegers(pairCounts.buffer)).toEqual([0]);
    expect(await readIntegers(status.buffer)).toEqual([1]);
    mask.buffer.write(new Uint32Array([1, 1, 1, 1]));
    await fixture.run();
    expect(await readIntegers(pairCounts.buffer)).toEqual([4]);
    expect(await readIntegers(status.buffer)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('rectangle envelope operation overwrites poisoned unplacedCount with the uniform-placement oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const coordinates = fixture.view('float32', new Float32Array([0, 0, 1, 1]));
    const positions = fixture.graph.createDataView(coordinates.view.buffer, {
      format: 'float32x2',
      length: 2
    });
    const parameters = fixture.view(
      'float32',
      getGPURipleyParameterValues({
        bounds: [0, 0, 1, 1],
        maximumDistance: 2,
        edgeCorrection: 'none'
      })
    );
    const lower = fixture.view('float32', new Float32Array(1));
    const unplacedCount = fixture.view('uint32', new Uint32Array([0xffffffff]));
    const operation = new GPURipleyEnvelopeOperation({
      positions: 'positions',
      parameters: 'parameters',
      rowCount: 2,
      gridSize: [2, 2],
      radiusCount: 1,
      simulationCount: 2,
      output: {lower: 'lower', unplacedCount: 'unplacedCount'}
    });
    const views = {
      positions,
      parameters: parameters.view,
      lower: lower.view,
      unplacedCount: unplacedCount.view
    };
    const getView: GPUContributorOperationViewGetter = port =>
      views[port as keyof typeof views] as never;
    fixture.graph.add(operation.createContributor(getView));
    // Rectangle sampling maps each finite uniform draw directly into [0,1]^2:
    // there is no rejection. Two points times two simulations therefore
    // have exactly zero unplaced draws, regardless of random seed.
    for (let encoding = 0; encoding < 2; encoding++) {
      unplacedCount.buffer.write(new Uint32Array([0xffffffff]));
      await fixture.run();
      expect(await readIntegers(unplacedCount.buffer)).toEqual([0]);
    }
  } finally {
    fixture.destroy();
  }
});

it('review 2 R5 randomization variance and z-score survive amplitude rescaling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const coordinates = fixture.view('float32', new Float32Array([0, 0, 1, 0, 2, 0, 3, 0, 4, 0]));
    const positions = fixture.graph.createDataView(coordinates.view.buffer, {
      format: 'float32x2',
      length: 5
    });
    const values = fixture.view('float32', new Float32Array(5));
    const parameters = fixture.view(
      'float32',
      getGPUSpatialCorrelogramParameterValues({
        bounds: [0, -1, 4, 1],
        maximumDistance: 1.1,
        varianceAssumption: 'randomization'
      })
    );
    const moransI = fixture.view('float32', new Float32Array(1));
    const varianceI = fixture.view('float32', new Float32Array(1));
    const zScores = fixture.view('float32', new Float32Array(1));
    const statistics = fixture.view('float32', new Float32Array(5));
    const pairCounts = fixture.view('uint32', new Uint32Array(1));
    const overflow = fixture.view('uint32', new Uint32Array([99]));
    fixture.graph.add(
      new GPUSpatialCorrelogram({
        positions,
        values: values.view,
        parameters: parameters.view,
        bandCount: 1,
        gridSize: [4, 1],
        output: {
          moransI: moransI.view,
          varianceI: varianceI.view,
          zScores: zScores.view,
          statistics: statistics.view,
          pairCounts: pairCounts.view,
          status: {overflow: overflow.view}
        }
      })
    );
    for (const amplitude of [1, 2 ** -47, 1]) {
      values.buffer.write(new Float32Array([amplitude, 0, 0, 0, 0]));
      overflow.buffer.write(new Uint32Array([99]));
      await fixture.run();
      expect(await readIntegers(pairCounts.buffer)).toEqual([4]);
      expect(await readIntegers(overflow.buffer)).toEqual([0]);
      // The independent Python enumeration of all five mark locations gives
      // I=[-1/16,-3/8,-3/8,-3/8,-1/16], mean -1/4, variance 3/128.
      // The endpoint z is sqrt(3/2); moment ratio is 13/4 at every amplitude.
      for (const [output, expected] of [
        [moransI, -0.0625],
        [varianceI, 0.0234375],
        [zScores, 1.224744871391589]
      ] as const) {
        const actual = (await readFloats(output.buffer))[0];
        expect(Number.isFinite(actual)).toBe(true);
        // Allow f32 analytic-variance cancellation and fixed-point product rounding.
        expect(Math.abs(actual / expected - 1)).toBeLessThan(4e-5);
      }
      const kurtosis = (await readFloats(statistics.buffer))[4];
      expect(Number.isFinite(kurtosis)).toBe(true);
      expect(Math.abs(kurtosis - 3.25)).toBeLessThan(2e-5);
    }
  } finally {
    fixture.destroy();
  }
});

for (const mode of ['single', 'cross', 'mixed-unit-powers'] as const) {
  it(`review 2 R4 space-time tiny-unit replay, ${mode}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const coordinates = fixture.view(
        'float32',
        new Float32Array([0.25, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75, 0.5, 0.5])
      );
      const positions = fixture.graph.createDataView(coordinates.view.buffer, {
        format: 'float32x2',
        length: 5
      });
      const values = fixture.view('float32', new Float32Array(5));
      const secondValues =
        mode === 'single' ? undefined : fixture.view('float32', new Float32Array(5));
      const times = fixture.view('float32', new Float32Array(5));
      const parameters = fixture.view(
        'float32',
        getGPUSpaceTimeVariogramParameterValues({
          bounds: [0, 0, 1, 1],
          maximumDistance: 2,
          maximumTimeLag: 1
        })
      );
      const semivariances = fixture.view('float32', new Float32Array(4));
      const pairCounts = fixture.view('uint32', new Uint32Array(4));
      const status = fixture.view('uint32', new Uint32Array(4).fill(99));
      fixture.graph.add(
        new GPUSpaceTimeVariogram({
          positions,
          values: values.view,
          secondValues: secondValues?.view,
          times: times.view,
          parameters: parameters.view,
          gridSize: [2, 2],
          lagCount: 1,
          temporalLagCount: 1,
          output: {
            semivariances: semivariances.view,
            pairCounts: pairCounts.view,
            status: status.view
          }
        })
      );
      for (const frame of [
        {values: [0, 0, 0, 0, 1], gamma: 0.2},
        {values: [0, 0, 0, 0, 2 ** -47], gamma: 1.0097419586828952e-29},
        {values: [0, 1, 2, 3, 524288].map(value => value * 2 ** -66), gamma: 1.009736180924286e-29},
        {values: [0, 0, 0, 0, 1], gamma: 0.2}
      ]) {
        // Equal times put every positive-distance pair in bin 1*2+0=2.
        // Cross column b=4a has gamma_ab=4 gamma_a. The mixed-unit case
        // b=2^47 a exercises unequal stored unit powers and gamma_ab=2^47 gamma_a.
        const multiplier = mode === 'single' ? 1 : mode === 'cross' ? 4 : 2 ** 47;
        values.buffer.write(new Float32Array(frame.values));
        secondValues?.buffer.write(new Float32Array(frame.values.map(value => value * multiplier)));
        status.buffer.write(new Uint32Array(4).fill(99));
        await fixture.run();
        expect(await readIntegers(pairCounts.buffer)).toEqual([0, 0, 10, 0]);
        expect(await readIntegers(status.buffer)).toEqual([1, 1, 0, 1]);
        const gamma = (await readFloats(semivariances.buffer))[2];
        expect(Number.isFinite(gamma)).toBe(true);
        // f32 moments plus fixed-point quantization; a relative check rejects zero.
        expect(Math.abs(gamma / (frame.gamma * multiplier) - 1)).toBeLessThan(5e-5);
      }
    } finally {
      fixture.destroy();
    }
  });
}

it('R4 and review 2 R3 rescaling a complete five-point pattern preserves bounded pair statistics without overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const coordinates = fixture.view(
      'float32',
      new Float32Array([0.25, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75, 0.5, 0.5])
    );
    const positions = fixture.graph.createDataView(coordinates.view.buffer, {
      format: 'float32x2',
      length: 5
    });
    const values = fixture.view('float32', new Float32Array([0, 0, 0, 0, 1]));
    const semivariances = fixture.view('float32', new Float32Array(1));
    const moransI = fixture.view('float32', new Float32Array(1));
    const variogramCounts = fixture.view('uint32', new Uint32Array(1));
    const correlogramCounts = fixture.view('uint32', new Uint32Array(1));
    const variogramOverflow = fixture.view('uint32', new Uint32Array([99]));
    const correlogramOverflow = fixture.view('uint32', new Uint32Array([99]));
    const variogramParameters = fixture.view(
      'float32',
      getGPUVariogramParameterValues({bounds: [0, 0, 1, 1], maximumDistance: 2})
    );
    const correlogramParameters = fixture.view(
      'float32',
      getGPUSpatialCorrelogramParameterValues({
        bounds: [0, 0, 1, 1],
        maximumDistance: 2,
        varianceAssumption: 'normality'
      })
    );
    fixture.graph.add(
      new GPUVariogram({
        positions,
        values: values.view,
        parameters: variogramParameters.view,
        lagCount: 1,
        gridSize: [2, 2],
        output: {
          semivariances: semivariances.view,
          pairCounts: variogramCounts.view,
          status: {overflow: variogramOverflow.view}
        }
      })
    );
    fixture.graph.add(
      new GPUSpatialCorrelogram({
        positions,
        values: values.view,
        parameters: correlogramParameters.view,
        bandCount: 1,
        gridSize: [2, 2],
        output: {
          moransI: moransI.view,
          pairCounts: correlogramCounts.view,
          status: {overflow: correlogramOverflow.view}
        }
      })
    );
    // Independent enumeration in pair-statistics-status-oracles.oracle.py:
    // C(5,2)=10. Four terms are a^2/2 and six are 0, so gamma=a^2/5.
    // A complete graph has Moran I=-1/(n-1)=-1/4. Zero IQR gives
    // u=(4*a^2/25)/4096: largest semivariance term=12800u, far below
    // channel capacity. Float32 rounding/quantization allow relative 3e-5;
    // testing the ratio rejects zero even for the tiny semivariance.
    for (const frame of [
      {values: [0, 0, 0, 0, 1], expected: 0.2},
      {values: [0, 0, 0, 0, 2 ** -47], expected: 1.0097419586828952e-29},
      // NumPy oracle: normal variance 8.077889447394289e-30 but u=1.8807801994016135e-39.
      {
        values: [0, 1, 2, 3, 524288].map(value => value * 2 ** -66),
        expected: 1.009736180924286e-29
      },
      {values: [0, 0, 0, 0, 1], expected: 0.2}
    ]) {
      values.buffer.write(new Float32Array(frame.values));
      variogramOverflow.buffer.write(new Uint32Array([99]));
      correlogramOverflow.buffer.write(new Uint32Array([99]));
      await fixture.run();
      expect(await readIntegers(variogramOverflow.buffer)).toEqual([0]);
      expect(await readIntegers(correlogramOverflow.buffer)).toEqual([0]);
      expect(await readIntegers(variogramCounts.buffer)).toEqual([10]);
      expect(await readIntegers(correlogramCounts.buffer)).toEqual([10]);
      const semivariance = (await readFloats(semivariances.buffer))[0];
      const moran = (await readFloats(moransI.buffer))[0];
      expect(Number.isFinite(semivariance)).toBe(true);
      expect(Number.isFinite(moran)).toBe(true);
      expect(Math.abs(semivariance / frame.expected - 1)).toBeLessThan(3e-5);
      expect(Math.abs(moran + 0.25)).toBeLessThan(3e-5);
    }
  } finally {
    fixture.destroy();
  }
});
