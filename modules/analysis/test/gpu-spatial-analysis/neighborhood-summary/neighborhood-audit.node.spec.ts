// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNeighborhoodSummaryOnCPU,
  CPUNeighborhoodSummaryOperation
} from '../../../src/gpu-spatial-analysis/neighborhood-summary/cpu-neighborhood-summary-operation';
import {computeNeighborhoodSummaryOracle, createWeights} from './neighborhood-summary-oracle';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import {GPUNeighborhoodSummaryOperation} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary-operation';
import {expectOracleRows} from '../spatial-interaction/spatial-interaction-audit-fixtures';
import {
  STATISTICS,
  INPUT,
  RAW_EXPECTED,
  STANDARDIZED_FOCAL_EXPECTED,
  INCOMPLETE_EXPECTED
} from './neighborhood-audit-fixtures';

it('keeps the lowest category on an exact standardized raw-mass tie', () => {
  // Exact binary masses: category 1=(1+9+21)/64, category 2=31/64;
  // masking category 3 leaves equal shares 1/2, entropy ln(2).
  for (const includeFocal of [false, true]) {
    const actual = computeNeighborhoodSummaryOnCPU(
      {
        offsets: Uint32Array.of(0, 5, 5, 5, 5, 5, 5),
        neighbors: Uint32Array.of(1, 2, 3, 4, 5),
        weights: Float32Array.of(1 / 64, 9 / 64, 21 / 64, 31 / 64, 2 / 64),
        categories: Uint32Array.of(2, 1, 1, 1, 2, 3),
        mask: Uint32Array.of(1, 1, 1, 1, 1, 0)
      },
      {standardized: true, includeFocal, focalWeight: 0.5}
    );
    expect(actual.modes[0]).toBe(includeFocal ? 2 : 1);
    // Focal category 2 adds 1/2 independently: shares become [1/3,2/3].
    expectOracleRows(
      actual.entropy.subarray(0, 1),
      [includeFocal ? 0.6365141682948128 : Math.LN2],
      1e-7
    );
  }
});

it('retains category accumulation residuals on an exactly equal dyadic mass partition', () => {
  // Four dyadic terms sum exactly to 1/2; category 2 has one term 1/2.
  const weights = createWeights([
    [
      [1, 1 / 8],
      [2, 1 / 8 + 2 ** -26],
      [3, 1 / 8 + 2 ** -26],
      [4, 1 / 8 - 2 ** -25],
      [5, 1 / 2]
    ],
    [],
    [],
    [],
    [],
    []
  ]);
  const categories = Uint32Array.of(0, 1, 1, 1, 1, 2);
  const cpu = computeNeighborhoodSummaryOnCPU({...weights, categories});
  expect(cpu.modes[0]).toBe(1);
  const reference = computeNeighborhoodSummaryOracle({
    weights,
    categories,
    values: new Float32Array(6)
  });
  expect(reference[0].mode).toBe(1);
  expect(reference[0].entropy).toBeCloseTo(Math.LN2, 7);
});

it('publishes the mean from the public weighted sum without anchor reconstruction loss', () => {
  const weights = createWeights([
    [
      [1, 1],
      [2, 1],
      [3, 1],
      [4, 1]
    ],
    [],
    [],
    [],
    []
  ]);
  const values = Float32Array.of(0, 1, 1, -1, -1 + 2 ** -24);
  const categories = new Uint32Array(5);
  // sum=2^-24 and W=4 exactly, so mean=2^-26.
  const cpu = computeNeighborhoodSummaryOnCPU(
    {...weights, values},
    {statistics: ['sum', 'weightSum', 'mean']}
  );
  expectOracleRows(cpu.table.subarray(0, 3), [2 ** -24, 4, 2 ** -26], 0);
  expect(computeNeighborhoodSummaryOracle({weights, values, categories})[0].mean).toBe(2 ** -26);
});

it('reserves the no-mode sentinel as missing category in the CPU function and operation', async () => {
  const input = {
    offsets: Uint32Array.of(0, 1, 1),
    neighbors: Uint32Array.of(1),
    weights: Float32Array.of(1),
    categories: Uint32Array.of(7, 0xffffffff)
  };
  for (const includeFocal of [false, true]) {
    const modes = [includeFocal ? 7 : 0xffffffff, 0xffffffff];
    const entropy = [includeFocal ? 0 : NaN, NaN];
    const actual = computeNeighborhoodSummaryOnCPU(input, {includeFocal});
    expectOracleRows(actual.modes, modes, 0);
    // A missing category gives no probability distribution, hence NaN, not entropy zero.
    expectOracleRows(actual.entropy, entropy, 0);
    const operation = new CPUNeighborhoodSummaryOperation({
      rowCount: 2,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      categories: 'categories',
      includeFocal,
      output: {modes: 'modes', entropy: 'entropy'}
    });
    const inputs: Record<string, GPUCPUVector> = Object.fromEntries(
      Object.entries(input).map(([name, values]) => [
        name,
        {
          format: values instanceof Float32Array ? 'float32' : 'uint32',
          length: values.length,
          chunks: [{values, length: values.length}],
          pack: () => values
        }
      ])
    );
    const outputs = (await operation.execute({inputs})) as Record<
      string,
      Float32Array | Uint32Array
    >;
    expectOracleRows(outputs['modes'], modes, 0);
    expectOracleRows(outputs['entropy'], entropy, 0);
  }
});

for (const includeFocal of [false, true]) {
  it(`matches all CPU numeric/categorical/status outputs to the independent oracle with focal=${includeFocal}`, async () => {
    const expected = includeFocal ? STANDARDIZED_FOCAL_EXPECTED : RAW_EXPECTED;
    const actual = computeNeighborhoodSummaryOnCPU(INPUT, {
      statistics: STATISTICS,
      includeFocal,
      focalWeight: 2,
      standardized: includeFocal
    });
    for (const port of ['table', 'modes', 'entropy', 'overflow'] as const)
      expectOracleRows(
        actual[port],
        expected[port],
        port === 'table' || port === 'entropy' ? 1e-6 : 0
      );
    const props = {
      rowCount: 8,
      weights: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        weights: 'weights',
        information: {
          rowCount: 8,
          transform: includeFocal ? ('row' as const) : ('raw' as const),
          symmetric: 'unknown' as const
        }
      },
      values: 'values',
      categories: 'categories',
      mask: 'mask',
      includeFocal,
      focalWeight: 2,
      statistics: [...STATISTICS],
      output: {table: 'table', modes: 'modes', entropy: 'entropy', status: {overflow: 'overflow'}}
    };
    const operation = new CPUNeighborhoodSummaryOperation(props);
    props.includeFocal = !includeFocal;
    const inputs: Record<string, GPUCPUVector> = Object.fromEntries(
      Object.entries(INPUT).map(([port, values]) => [
        port,
        {
          format: values instanceof Float32Array ? 'float32' : 'uint32',
          length: values.length,
          chunks: [{values, length: values.length}],
          pack: () => values
        }
      ])
    );
    const outputs = (await operation.execute({inputs})) as Record<
      string,
      Float32Array | Uint32Array
    >;
    for (const port of ['table', 'modes', 'entropy', 'overflow'] as const)
      expectOracleRows(
        outputs[port],
        expected[port],
        port === 'table' || port === 'entropy' ? 1e-6 : 0
      );
  });
}

for (const [field, statusInput] of [
  ['weights.status.overflow', {status: {overflow: Uint32Array.of(1)}}],
  ['weights.status.candidateOverflow', {status: {candidateOverflow: Uint32Array.of(1)}}],
  ['weightsIterationLimitReached', {weightsIterationLimitReached: Uint32Array.of(1)}],
  ['weightsConverged', {weightsConverged: Uint32Array.of(0)}]
] as const) {
  it(`suppresses dependent CPU results on incomplete ${field}`, () => {
    const actual = computeNeighborhoodSummaryOnCPU(
      {...INPUT, ...statusInput},
      {statistics: STATISTICS}
    );
    for (const port of ['table', 'modes', 'entropy', 'overflow'] as const)
      expectOracleRows(actual[port], INCOMPLETE_EXPECTED[port], 0);
  });
}

it('keeps a finite even median near the float32 extrema in the old f32 test reference too', () => {
  const value = Math.fround(3.2e38);
  const actual = computeNeighborhoodSummaryOracle({
    weights: createWeights([
      [
        [1, 1],
        [2, 1]
      ],
      [],
      []
    ]),
    values: Float32Array.of(0, value, value),
    categories: Uint32Array.of(1, 2, 2)
  });
  expect(actual[0].median).toBe(value);
});

it('uses the float32 focal weight that was validated for the GPU model', () => {
  const actual = computeNeighborhoodSummaryOnCPU(
    {
      offsets: new Uint32Array(9),
      neighbors: Uint32Array.of(0),
      weights: Float32Array.of(0),
      values: Float32Array.of(2, 3, 4, 5, 6, 7, 8, 9)
    },
    {statistics: ['count', 'weightSum', 'mean'], includeFocal: true, focalWeight: 1e-50}
  );
  // The validated shader prop rounds to zero. Focal membership still counts,
  // but a zero total weight has no weighted mean.
  expectOracleRows(actual.table, Array.from({length: 8}, () => [1, 0, NaN]).flat(), 0);
  expect(() =>
    computeNeighborhoodSummaryOnCPU(
      {
        offsets: new Uint32Array(9),
        neighbors: Uint32Array.of(0),
        weights: Float32Array.of(0),
        values: new Float32Array(8)
      },
      {statistics: ['mean'], includeFocal: true, focalWeight: -1e-50}
    )
  ).toThrow();
});

it('parses every completeness, median and categorical shader emitted by the GPU operation', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      offsets: {format: 'uint32', length: 9},
      neighbors: {format: 'uint32', length: 16},
      weights: {format: 'float32', length: 16},
      values: {format: 'float32', length: 8},
      categories: {format: 'uint32', length: 8},
      mask: {format: 'uint32', length: 8},
      weightsOverflow: {format: 'uint32', length: 1},
      weightsCandidateOverflow: {format: 'uint32', length: 1},
      weightsIterationLimitReached: {format: 'uint32', length: 1},
      weightsConverged: {format: 'uint32', length: 1},
      table: {format: 'float32', length: 64},
      modes: {format: 'uint32', length: 8},
      entropy: {format: 'float32', length: 8},
      overflow: {format: 'uint32', length: 1}
    });
    const operation = new GPUNeighborhoodSummaryOperation({
      rowCount: 8,
      statistics: STATISTICS,
      values: 'values',
      categories: 'categories',
      mask: 'mask',
      weights: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        weights: 'weights',
        status: {
          overflow: 'weightsOverflow',
          candidateOverflow: 'weightsCandidateOverflow',
          iterationLimitReached: 'weightsIterationLimitReached',
          converged: 'weightsConverged'
        }
      },
      output: {table: 'table', modes: 'modes', entropy: 'entropy', status: {overflow: 'overflow'}}
    });
    const contributor = operation.createContributor(getView);
    for (const kernel of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)))
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
  } finally {
    device.destroy();
  }
});
