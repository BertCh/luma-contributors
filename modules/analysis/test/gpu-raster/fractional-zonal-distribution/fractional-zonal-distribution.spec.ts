// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPURasterFractionalCoverage,
  getGPURasterFractionalCoverageParameterValues
} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {GPURasterFractionalZonalDistribution} from '../../../src/raster-analysis/gpu-raster-fractional-zonal-distribution';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from '../../oracles/fixtures/fractional-zonal-distribution.json';
import weightedReference from '../../oracles/fixtures/raster-round-two.json';

const MOMENTS = [
  'means',
  'populationVariances',
  'sampleVariances',
  'effectiveSampleSizes',
  'weightSums',
  'squaredWeightSums',
  'coveredArea',
  'validArea'
] as const;

type Settings = {
  values: number[];
  observationWeights?: number[];
  cellAreas?: number[];
  quantileMethod?: 'exactextract' | 'inverse-cdf';
  polygon?: {positions: number[]; width: number; height: number};
  fractions: number[];
  cellIds?: number[];
  zoneIds?: number[];
  zoneCount?: number;
  probabilities?: number[];
  count?: number;
  requiredCount?: number;
  overflow?: number;
  zoneValidity?: number[];
};
function createDistribution(device: Device, settings: Settings) {
  const rig = createParityGraph(device);
  const zoneCount = settings.zoneCount ?? 1;
  const probabilities = rig.input(
    'probabilities',
    settings.probabilities ?? [0, 0.5, 1],
    'float32'
  );
  const values = rig.input('values', settings.values, 'float32');
  const observationWeights = settings.observationWeights
    ? rig.input('observation-weights', settings.observationWeights, 'float32')
    : undefined;
  const cellAreas = settings.cellAreas
    ? rig.input('cell-areas', settings.cellAreas, 'float32')
    : undefined;
  const fractions = rig.input('fractions', settings.fractions, 'float32');
  const cellIds = rig.input(
    'cell-ids',
    settings.cellIds ?? settings.fractions.map((_, index) => index),
    'uint32'
  );
  const zoneIds = rig.input(
    'zone-ids',
    settings.zoneIds ?? settings.fractions.map(() => 0),
    'uint32'
  );
  const count = rig.input('count', [settings.count ?? settings.fractions.length], 'uint32');
  const requiredCount = rig.input(
    'required',
    [settings.requiredCount ?? settings.count ?? settings.fractions.length],
    'uint32'
  );
  const overflow = rig.input('overflow', [settings.overflow ?? 0], 'uint32');
  const zoneValidity = rig.input(
    'zone-validity',
    settings.zoneValidity ?? Array(zoneCount).fill(1),
    'uint32'
  );
  const output = {
    means: rig.output('means', zoneCount, 'float32'),
    populationVariances: rig.output('population-variances', zoneCount, 'float32'),
    sampleVariances: rig.output('sample-variances', zoneCount, 'float32'),
    effectiveSampleSizes: rig.output('effective-sample-sizes', zoneCount, 'float32'),
    weightSums: rig.output('weight-sums', zoneCount, 'float32'),
    squaredWeightSums: rig.output('squared-weight-sums', zoneCount, 'float32'),
    coveredArea: rig.output('covered-area', zoneCount, 'float32'),
    validArea: rig.output('valid-area', zoneCount, 'float32'),
    quantiles: rig.output('quantiles', zoneCount * probabilities.view.length, 'float32'),
    quantileValidity: rig.output(
      'quantile-validity',
      zoneCount * probabilities.view.length,
      'uint32'
    ),
    validity: rig.output('validity', zoneCount, 'uint32'),
    status: rig.output('status', zoneCount, 'uint32')
  };
  const grid = {
    width: settings.polygon?.width ?? settings.values.length,
    height: settings.polygon?.height ?? 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  if (settings.polygon) {
    rig.graph.add(
      new GPURasterFractionalCoverage({
        grid,
        zones: {
          positions: rig.input('polygon-positions', settings.polygon.positions, 'float32x2').view,
          featureOffsets: rig.input('feature-offsets', [0, 1], 'uint32').view,
          polygonOffsets: rig.input('polygon-offsets', [0, 2], 'uint32').view,
          ringOffsets: rig.input('ring-offsets', [0, 4, 8], 'uint32').view
        },
        parameters: rig.input(
          'coverage-parameters',
          Array.from(getGPURasterFractionalCoverageParameterValues(grid)),
          'float32'
        ).view,
        output: {
          zoneIds: zoneIds.view,
          cellIds: cellIds.view,
          fractions: fractions.view,
          status: {count: count.view, requiredCount: requiredCount.view, overflow: overflow.view},
          zoneValidity: zoneValidity.view
        }
      })
    );
  }
  rig.graph.add(
    new GPURasterFractionalZonalDistribution({
      grid,
      input: {id: 'input', format: 'float32', storage: {kind: 'buffer', values: values.view}},
      zoneCount,
      coverage: {
        zoneIds: zoneIds.view,
        cellIds: cellIds.view,
        fractions: fractions.view,
        status: {count: count.view, requiredCount: requiredCount.view, overflow: overflow.view},
        zoneValidity: zoneValidity.view
      },
      probabilities: probabilities.view,
      observationWeights: observationWeights?.view,
      cellAreas: cellAreas?.view,
      quantileMethod: settings.quantileMethod,
      output: {
        means: output.means.view,
        populationVariances: output.populationVariances.view,
        sampleVariances: output.sampleVariances.view,
        effectiveSampleSizes: output.effectiveSampleSizes.view,
        weightSums: output.weightSums.view,
        squaredWeightSums: output.squaredWeightSums.view,
        coveredArea: output.coveredArea.view,
        validArea: output.validArea.view,
        quantiles: output.quantiles.view,
        quantileValidity: output.quantileValidity.view,
        validity: output.validity.view,
        status: output.status.view
      }
    })
  );
  return {
    ...rig,
    output,
    observationWeights,
    cellAreas,
    values,
    fractions,
    cellIds,
    zoneIds,
    count,
    requiredCount,
    overflow,
    zoneValidity,
    probabilities
  };
}

function expectNumbers(actual: number[], expected: (number | null)[], label: string) {
  expect(actual.length, label).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    const value = expected[index];
    if (value === null) expect(actual[index], `${label}[${index}]`).toBeNaN();
    else
      expect(Math.abs(actual[index] - value), `${label}[${index}]`).toBeLessThanOrEqual(
        Math.abs(value) >= 1048576 ? 0.06251 : 2e-6 + Math.abs(value) * 2e-6
      );
  }
}

for (const entry of fixture.cases) {
  it(`RS01 direct exactextract/NumPy parity: ${entry.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = createDistribution(device, {
      ...entry,
      values: entry.values.map(value => value ?? NaN)
    });
    const compiled = await rig.graph.compileAsync();
    try {
      submitGraph(device, compiled, undefined);
      for (const name of MOMENTS)
        expectNumbers(await readFloat32(rig.output[name].buffer, 1), [entry.expected[name]], name);
      expectNumbers(
        await readFloat32(rig.output.quantiles.buffer, entry.probabilities.length),
        entry.expected.quantiles,
        'quantiles'
      );
      expect(await readUint32(rig.output.validity.buffer, 1)).toEqual([entry.expected.validity]);
      expect(await readUint32(rig.output.status.buffer, 1)).toEqual([entry.expected.status]);
      expect(
        await readUint32(rig.output.quantileValidity.buffer, entry.probabilities.length)
      ).toEqual(entry.expected.quantiles.map(value => Number(value !== null)));
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  }, 120000);
}

it('RS01 rejects malformed tuples/incomplete coverage and clears every status on reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createDistribution(device, {values: [2, 6, 12], fractions: [1, 0.5, 0.25]});
  const compiled = await rig.graph.compileAsync();
  const reset = () => {
    rig.values.buffer.write(Float32Array.of(2, 6, 12));
    rig.fractions.buffer.write(Float32Array.of(1, 0.5, 0.25));
    rig.cellIds.buffer.write(Uint32Array.of(0, 1, 2));
    rig.zoneIds.buffer.write(Uint32Array.of(0, 0, 0));
    rig.count.buffer.write(Uint32Array.of(3));
    rig.requiredCount.buffer.write(Uint32Array.of(3));
    rig.overflow.buffer.write(Uint32Array.of(0));
    rig.zoneValidity.buffer.write(Uint32Array.of(1));
  };
  try {
    const failures = [
      {
        name: 'negative fraction',
        flag: 2,
        change: () => rig.fractions.buffer.write(Float32Array.of(1, -0.1, 0.25))
      },
      {
        name: 'nonfinite fraction',
        flag: 2,
        change: () => rig.fractions.buffer.write(Float32Array.of(1, NaN, 0.25))
      },
      {
        name: 'fraction above one',
        flag: 2,
        change: () => rig.fractions.buffer.write(Float32Array.of(1, 1.1, 0.25))
      },
      {
        name: 'invalid cell',
        flag: 2,
        change: () => rig.cellIds.buffer.write(Uint32Array.of(0, 1, 99))
      },
      {
        name: 'invalid zone',
        flag: 2,
        change: () => rig.zoneIds.buffer.write(Uint32Array.of(0, 99, 0))
      },
      {
        name: 'duplicate cell',
        flag: 2,
        change: () => rig.cellIds.buffer.write(Uint32Array.of(0, 0, 2))
      },
      {
        name: 'zero-weight duplicate cell',
        flag: 2,
        change: () => {
          rig.cellIds.buffer.write(Uint32Array.of(0, 0, 2));
          rig.fractions.buffer.write(Float32Array.of(1, 0, 0.25));
        }
      },
      {
        name: 'capacity overflow',
        flag: 1,
        change: () => {
          rig.overflow.buffer.write(Uint32Array.of(1));
          rig.requiredCount.buffer.write(Uint32Array.of(4));
        }
      },
      {
        name: 'inconsistent count',
        flag: 1,
        change: () => rig.count.buffer.write(Uint32Array.of(4))
      },
      {
        name: 'failed zone triangulation',
        flag: 1,
        change: () => rig.zoneValidity.buffer.write(Uint32Array.of(0))
      }
    ];
    for (const failure of failures) {
      reset();
      failure.change();
      submitGraph(device, compiled, undefined);
      expect((await readUint32(rig.output.status.buffer, 1))[0] & failure.flag, failure.name).toBe(
        failure.flag
      );
      expect(await readUint32(rig.output.validity.buffer, 1), failure.name).toEqual([0]);
      expectNumbers(
        await readFloat32(rig.output.quantiles.buffer, 3),
        [null, null, null],
        failure.name
      );
      expectNumbers(await readFloat32(rig.output.weightSums.buffer, 1), [null], failure.name);
      reset();
      submitGraph(device, compiled, undefined);
      expect(await readUint32(rig.output.status.buffer, 1), `reset ${failure.name}`).toEqual([0]);
      expectNumbers(await readFloat32(rig.output.means.buffer, 1), [8 / 1.75], 'restored mean');
    }
    reset();
    rig.count.buffer.write(Uint32Array.of(0));
    rig.requiredCount.buffer.write(Uint32Array.of(0));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(rig.output.status.buffer, 1)).toEqual([12]);
    expectNumbers(
      await readFloat32(rig.output.coveredArea.buffer, 1),
      [0],
      'empty covered support'
    );
  } finally {
    compiled.destroy();
    rig.destroy();
  }
}, 120000);

it('RS01 dynamic probabilities and values keep nodata and quantile validity explicit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createDistribution(device, {
    values: [1, 3, 7],
    fractions: [1, 1, 1],
    probabilities: [-0.1, NaN, 1.1]
  });
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    expectNumbers(
      await readFloat32(rig.output.quantiles.buffer, 3),
      [null, null, null],
      'invalid q'
    );
    expect(await readUint32(rig.output.quantileValidity.buffer, 3)).toEqual([0, 0, 0]);
    expect(await readUint32(rig.output.validity.buffer, 1)).toEqual([1]);
    rig.probabilities.buffer.write(Float32Array.of(0, 0.5, 1));
    rig.values.buffer.write(Float32Array.of(-0, 0, Infinity));
    submitGraph(device, compiled, undefined);
    expectNumbers(
      await readFloat32(rig.output.quantiles.buffer, 3),
      [0, 0, 0],
      'updated q and nodata'
    );
    expectNumbers(
      await readFloat32(rig.output.effectiveSampleSizes.buffer, 1),
      [2],
      'signed-zero distinct cells'
    );
    expectNumbers(await readFloat32(rig.output.coveredArea.buffer, 1), [3], 'covered');
    expectNumbers(await readFloat32(rig.output.validArea.buffer, 1), [2], 'observed');
  } finally {
    compiled.destroy();
    rig.destroy();
  }
}, 120000);

it('RS01 explicit disjoint tile tuple concatenation preserves distribution and support', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const entry = fixture.cases[0];
  // Right-core IDs are explicitly rebased to the full-domain value band. No halo cells occur.
  const permutation = [3, 4, 5, 0, 1, 2];
  const rig = createDistribution(device, {
    values: entry.values.map(value => value ?? NaN),
    fractions: permutation.map(index => entry.fractions[index]),
    cellIds: permutation,
    probabilities: entry.probabilities
  });
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    for (const name of MOMENTS)
      expectNumbers(await readFloat32(rig.output[name].buffer, 1), [entry.expected[name]], name);
    expectNumbers(
      await readFloat32(rig.output.quantiles.buffer, entry.probabilities.length),
      entry.expected.quantiles,
      'whole-domain quantiles from explicit tile tuples'
    );
  } finally {
    compiled.destroy();
    rig.destroy();
  }
}, 120000);

it('RS01 overlapping zones preserve independent source identities and partial support', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createDistribution(device, {
    values: [2, 10, NaN],
    fractions: [0.5, 0.5, 1, 0.25, 1],
    cellIds: [0, 1, 1, 2, 0],
    zoneIds: [0, 0, 1, 1, 3],
    zoneCount: 4,
    zoneValidity: [1, 1, 1, 0]
  });
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    expectNumbers(await readFloat32(rig.output.means.buffer, 4), [6, 10, null, null], 'zone means');
    expectNumbers(
      await readFloat32(rig.output.effectiveSampleSizes.buffer, 4),
      [2, 1, 0, null],
      'zone support'
    );
    expectNumbers(
      await readFloat32(rig.output.coveredArea.buffer, 4),
      [1, 1.25, 0, null],
      'covered area'
    );
    expectNumbers(await readFloat32(rig.output.validArea.buffer, 4), [1, 1, 0, null], 'valid area');
    expect(await readUint32(rig.output.validity.buffer, 4)).toEqual([1, 1, 0, 0]);
    expect(await readUint32(rig.output.status.buffer, 4)).toEqual([0, 8, 12, 9]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
}, 120000);

it('RS01 composes GPU fractional polygon coverage with holes against direct exactextract', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const entry = fixture.coverageIntegration;
  const rig = createDistribution(device, {...entry, polygon: entry, fractions: Array(9).fill(0)});
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    for (const name of MOMENTS)
      expectNumbers(await readFloat32(rig.output[name].buffer, 1), [entry.expected[name]], name);
    expectNumbers(
      await readFloat32(rig.output.quantiles.buffer, entry.probabilities.length),
      entry.expected.quantiles,
      'GPU coverage quantiles'
    );
    expect(await readUint32(rig.output.status.buffer, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
}, 120000);

it('RS01 separates observation reliability, geographic cell area, and exact inverse CDF', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createDistribution(device, {
    values: [1, 3, 9, 100],
    fractions: [1, 0.5, 0.25, 1],
    observationWeights: [2, 4, 8, 0],
    cellAreas: [10, 20, 40, 50],
    quantileMethod: 'inverse-cdf',
    probabilities: [0, 0.25, 0.5, 0.75, 1]
  });
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    expectNumbers(
      await readFloat32(rig.output.means.buffer, 1),
      [weightedReference.weighted.weighted_mean],
      'mean'
    );
    expectNumbers(
      await readFloat32(rig.output.populationVariances.buffer, 1),
      [weightedReference.weighted.weighted_variance],
      'variance'
    );
    expectNumbers(
      await readFloat32(rig.output.sampleVariances.buffer, 1),
      [weightedReference.weighted.sampleVariance],
      'sample variance'
    );
    expectNumbers(await readFloat32(rig.output.effectiveSampleSizes.buffer, 1), [3], 'support');
    expect(await readFloat32(rig.output.weightSums.buffer, 1)).toEqual([6]);
    expect(await readFloat32(rig.output.coveredArea.buffer, 1)).toEqual([80]);
    expect(await readFloat32(rig.output.validArea.buffer, 1)).toEqual([80]);
    expect(await readFloat32(rig.output.quantiles.buffer, 5)).toEqual(
      weightedReference.weighted.quantiles
    );
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

it('RS01 exact continuous distribution scales to 16384 independently weighted observations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const count = 16384;
  const rig = createDistribution(device, {
    values: Array.from({length: count}, (_, index) => index / 16),
    fractions: Array(count).fill(0.5),
    observationWeights: Array(count).fill(2),
    quantileMethod: 'inverse-cdf'
  });
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    expectNumbers(await readFloat32(rig.output.means.buffer, 1), [(count - 1) / 32], 'scale mean');
    expectNumbers(
      await readFloat32(rig.output.effectiveSampleSizes.buffer, 1),
      [count],
      'scale support'
    );
    expect(await readFloat32(rig.output.quantiles.buffer, 3)).toEqual([
      0,
      (count / 2 - 1) / 16,
      (count - 1) / 16
    ]);
    expect(await readUint32(rig.output.status.buffer, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
}, 120000);

it('RS01 weight/area changes retain explicit invalid and zero support without stale results', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createDistribution(device, {
    values: [2, 8],
    fractions: [1, 0.5],
    observationWeights: [1, 1],
    cellAreas: [2, 4],
    quantileMethod: 'inverse-cdf'
  });
  const compiled = await rig.graph.compileAsync();
  try {
    for (const weights of [
      [0, 0],
      [-1, 1],
      [NaN, 1],
      [2, 4]
    ]) {
      rig.observationWeights!.buffer.write(Float32Array.from(weights));
      submitGraph(device, compiled, undefined);
      const status = (await readUint32(rig.output.status.buffer, 1))[0];
      if (weights[0] === 0) {
        expect(status & 4).toBe(4);
        expect(await readFloat32(rig.output.validArea.buffer, 1)).toEqual([4]);
      } else if (weights[0] === 2) {
        expect(status).toBe(0);
        expect(await readFloat32(rig.output.means.buffer, 1)).toEqual([5]);
      } else expect(status & 2).toBe(2);
    }
    rig.cellAreas!.buffer.write(Float32Array.of(0, 4));
    submitGraph(device, compiled, undefined);
    expect((await readUint32(rig.output.status.buffer, 1))[0] & 2).toBe(2);
    rig.cellAreas!.buffer.write(Float32Array.of(2, 4));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(rig.output.status.buffer, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

it('RS01 nonbinary independent weights at 16384 rows match pinned NumPy moments and weighted quantiles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const reference = weightedReference.weightedScale,
    count = reference.count;
  const rig = createDistribution(device, {
    values: Array.from({length: count}, (_, index) => (index % 61) / 7),
    fractions: Array.from({length: count}, (_, index) => ((index % 7) + 1) / 8),
    observationWeights: Array.from({length: count}, (_, index) => ((index % 19) + 1) / 11),
    probabilities: reference.probabilities,
    quantileMethod: 'inverse-cdf'
  });
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    for (const name of [
      'means',
      'populationVariances',
      'sampleVariances',
      'weightSums',
      'squaredWeightSums',
      'effectiveSampleSizes'
    ] as const)
      expectNumbers(await readFloat32(rig.output[name].buffer, 1), [reference[name]], name);
    expectNumbers(
      await readFloat32(rig.output.quantiles.buffer, 3),
      reference.quantiles,
      'weighted inverse CDF'
    );
    expect(await readUint32(rig.output.status.buffer, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
}, 120000);

it('RS01 sums of 65536 tuples keep float32 accuracy on the GPU (striped partial sums, math review)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The largest accepted zone. A serial float32 loop over 65536 equal terms drifts: computed
  // with Math.fround, the serial sum of the areas (0.7 each) is off by 6.5e-4 of its value and
  // the serial sum of the squared weights by 3.5e-4. The kernel stripes each zone over 256 partial
  // sums and reduces them in a fixed tree, and centres the values on an anchor before the
  // moments. This spec checks on the GPU that those sums are right to float32 accuracy; a float
  // Kahan or Neumaier correction would not survive the shader compiler (see the math fix brief).
  const count = 65536;
  const fraction = Math.fround(0.7);
  const observationWeight = Math.fround(0.1);
  const weight = Math.fround(fraction * observationWeight);
  // A large mean with a small spread: 100000 + k / 16, k = 0..15, each exact in float32.
  const rig = createDistribution(device, {
    values: Array.from({length: count}, (_, index) => 100000 + (index % 16) / 16),
    fractions: Array(count).fill(fraction),
    observationWeights: Array(count).fill(observationWeight),
    probabilities: [0.5]
  });
  const compiled = await rig.graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(rig.output.status.buffer, 1)).toEqual([0]);
    const expectRelative = (actual: number, expected: number, tolerance: number, label: string) =>
      expect(Math.abs(actual - expected) / Math.abs(expected), label).toBeLessThan(tolerance);
    // Each of the 256 partial sums is a serial float32 sum of 256 terms, so a few parts in a
    // million remain; the serial sum over all 65536 terms is off by sixty times this bound.
    const SUM_TOLERANCE = 1e-5;
    // Float64 closed forms. Equal weights: the mean is 100000 + 15 / 32 and the population
    // variance of k / 16 for k = 0..15 is (16^2 - 1) / 12 / 256.
    expectRelative(
      (await readFloat32(rig.output.weightSums.buffer, 1))[0],
      count * weight,
      SUM_TOLERANCE,
      'weight sum'
    );
    expectRelative(
      (await readFloat32(rig.output.squaredWeightSums.buffer, 1))[0],
      count * weight * weight,
      SUM_TOLERANCE,
      'squared weight sum'
    );
    expectRelative(
      (await readFloat32(rig.output.coveredArea.buffer, 1))[0],
      count * fraction,
      SUM_TOLERANCE,
      'covered area'
    );
    expectRelative(
      (await readFloat32(rig.output.validArea.buffer, 1))[0],
      count * fraction,
      SUM_TOLERANCE,
      'valid area'
    );
    // One float32 ulp at 100000 is 0.0078.
    expect(
      Math.abs((await readFloat32(rig.output.means.buffer, 1))[0] - (100000 + 15 / 32))
    ).toBeLessThanOrEqual(0.0079);
    expectRelative(
      (await readFloat32(rig.output.populationVariances.buffer, 1))[0],
      255 / 12 / 256,
      1e-4,
      'population variance'
    );
    expectRelative(
      (await readFloat32(rig.output.effectiveSampleSizes.buffer, 1))[0],
      count,
      2 * SUM_TOLERANCE,
      'effective sample size'
    );
  } finally {
    compiled.destroy();
    rig.destroy();
  }
}, 240000);
