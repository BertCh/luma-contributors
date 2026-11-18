// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUPermutationParameterValues,
  GPULocalPermutationTest,
  GPU_LOCAL_PERMUTATION_NOT_TESTED,
  GPU_PERMUTATION_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {
  GPUSpaceTimeLocalMoran,
  type GPUSpaceTimeLocalMoranVariant
} from '../../../src/gpu-spatial-analysis/space-time-statistics/index';
import {
  GPUSpatialWeightsAlgebra,
  type GPUSpaceTimeWeightsLayout
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  GPULocalMoran,
  getGPUSpatialAutocorrelationParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  computeLocalPermutationOracle,
  createWeights
} from '../permutation-inference/permutation-oracle';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import type {OracleCSR} from '../spatial-weights/spatial-weights-oracle';
import {SPACE_TIME_STATISTICS_REFERENCE as REFERENCE} from './space-time-statistics-reference';

// Oracles: test/oracles/space-time-statistics.py (esda 2.10.0) for the statistics and p-values; an
// independent double-precision CPU twin below for the statistics; permutation-oracle.ts (the TS
// twin of the Philox draws) for the exceedance counts.

const CELLS = REFERENCE.n;
const PERIODS = REFERENCE.periods;
const PERMUTATIONS = REFERENCE.permutations;
const SEED = 20240607;

it('analytic and permutation Moran suppress the same incomplete CSR and recover on reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const overflow = new GPUParameterBuffer(device, {id: 'moran-loss', format: 'uint32', length: 1});
  const candidateOverflow = new GPUParameterBuffer(device, {
    id: 'moran-candidate-loss',
    format: 'uint32',
    length: 1
  });
  let compiled: ReturnType<typeof rig.graph.compile> | undefined;
  try {
    const weights = {
      ...rig.uploadWeights({
        offsets: [0, 1, 3, 5, 6],
        neighbors: [1, 0, 2, 1, 3, 2],
        weights: [1, 1, 1, 1, 1, 1],
        distances: []
      }),
      status: {
        overflow: overflow.importToGraph(rig.graph),
        candidateOverflow: candidateOverflow.importToGraph(rig.graph)
      }
    };
    const analytic = rig.output('float32', 4),
      observed = rig.output('float32', 4),
      pseudo = rig.output('float32', 4);
    const completeness = rig.output('uint32', 1);
    rig.graph.add(
      new GPULocalMoran({
        weights,
        values: rig.input(Float32Array.of(1, 2, 4, 8), 'float32', 4),
        parameters: rig.input(
          getGPUSpatialAutocorrelationParameterValues({}),
          'float32',
          getGPUSpatialAutocorrelationParameterValues({}).length
        ),
        output: {zScores: rig.output('float32', 4).view, localI: analytic.view}
      })
    );
    rig.graph.add(
      new GPUSpaceTimeLocalMoran({
        weights,
        cube: rig.input(Float32Array.of(0, 0, 0, 0, 1, 2, 4, 8), 'float32', 8),
        layout: 'period-major',
        periodCount: 2,
        period: 1,
        variant: 'differential',
        maximumPermutations: 7,
        maximumNeighbors: 4,
        parameters: rig.input(
          getGPUPermutationParameterValues({seed: 23, permutationCount: 7}),
          'uint32',
          GPU_PERMUTATION_PARAMETER_LENGTH
        ),
        output: {
          exceedances: rig.output('uint32', 4).view,
          observed: observed.view,
          pseudoPValues: pseudo.view,
          status: {overflow: completeness.view}
        }
      })
    );
    compiled = rig.graph.compile();
    for (const [finalLoss, candidateLoss] of [
      [0, 1],
      [0, 0],
      [1, 0],
      [0, 0]
    ]) {
      overflow.write(Uint32Array.of(finalLoss));
      candidateOverflow.write(Uint32Array.of(candidateLoss));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(completeness.buffer, 1)).toEqual([
        finalLoss || candidateLoss ? 1 : 0
      ]);
      const results = await Promise.all([
        readFloat32(analytic.buffer, 4),
        readFloat32(observed.buffer, 4),
        readFloat32(pseudo.buffer, 4)
      ]);
      for (const result of results)
        expect(result.every(finalLoss || candidateLoss ? Number.isNaN : Number.isFinite)).toBe(
          true
        );
    }
  } finally {
    compiled?.destroy();
    rig.destroy();
    overflow.destroy();
    candidateOverflow.destroy();
  }
});
const LEVEL = 0.05;
/** Five-sigma difference of two independent p-value estimates, plus esda's two-sided definition gap. */
const P_VALUE_BOUND = 5 * Math.sqrt((2 * 0.25) / PERMUTATIONS) + 0.015;

const spatialCSR: OracleCSR = (() => {
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (const row of REFERENCE.neighbors) {
    for (const neighbor of row) {
      neighbors.push(neighbor);
      weights.push(1 / REFERENCE.k);
    }
    offsets.push(neighbors.length);
  }
  return {offsets, neighbors, weights, distances: []};
})();

/** f32 cube in the requested layout from the period-major fixture lists. */
function layoutCube(periodMajor: number[][], layout: GPUSpaceTimeWeightsLayout): Float32Array {
  const cube = new Float32Array(CELLS * PERIODS);
  for (let period = 0; period < PERIODS; period++) {
    for (let cell = 0; cell < CELLS; cell++) {
      cube[layout === 'period-major' ? period * CELLS + cell : cell * PERIODS + period] =
        periodMajor[period][cell];
    }
  }
  return cube;
}

function sliceOf(period: number): Float32Array {
  return Float32Array.from(REFERENCE.cube[period]);
}

/** Double-precision local Moran (esda scaling) of one variable pair, row-standardized neighbors. */
function computeMoranReference(
  focal: ArrayLike<number>,
  other: ArrayLike<number>,
  bivariate: boolean
): number[] {
  const count = focal.length;
  const standardize = (values: ArrayLike<number>) => {
    const mean = Array.from(values).reduce((total, value) => total + value, 0) / count;
    const deviation = Math.sqrt(
      Array.from(values).reduce((total, value) => total + (value - mean) ** 2, 0) / count
    );
    return Array.from(values, value => (value - mean) / deviation);
  };
  const zFocal = standardize(focal);
  const zOther = bivariate ? standardize(other) : zFocal;
  const denominator = zFocal.reduce((total, value) => total + value * value, 0);
  return zFocal.map((value, row) => {
    const neighbors = REFERENCE.neighbors[row];
    const lag =
      neighbors.reduce((total, neighbor) => total + zOther[neighbor], 0) / neighbors.length;
    // esda (both forms): I = (n - 1) z_focal lag(z_other) / sum z_focal^2.
    return ((count - 1) * value * lag) / denominator;
  });
}

type Variant = {
  label: string;
  variant: GPUSpaceTimeLocalMoranVariant;
  direction?: 'inward' | 'outward';
  lag: number;
  key: 'laggedInward' | 'laggedOutward' | 'differential';
};

const VARIANTS: Variant[] = [
  {label: 'lagged inward lag 1', variant: 'lagged', lag: 1, key: 'laggedInward'},
  {
    label: 'lagged outward lag 2',
    variant: 'lagged',
    direction: 'outward',
    lag: 2,
    key: 'laggedOutward'
  },
  {label: 'differential lag 1', variant: 'differential', lag: 1, key: 'differential'},
  {label: 'differential lag 2', variant: 'differential', lag: 2, key: 'differential'}
];

/** Expected focal and permuted columns of a variant, in f32 like the slice kernel. */
function getColumns(item: Variant): {values: Float32Array; secondValues?: Float32Array} {
  const now = sliceOf(REFERENCE.period);
  const before = sliceOf(REFERENCE.period - item.lag);
  if (item.variant === 'differential') {
    return {values: now.map((value, cell) => Math.fround(value - before[cell]))};
  }
  return item.direction === 'outward'
    ? {values: before, secondValues: now}
    : {values: now, secondValues: before};
}

async function runRecipe(rig: WeightsRig, item: Variant, layout: GPUSpaceTimeWeightsLayout) {
  const device = rig.device;
  const weights = {
    ...rig.uploadWeights(spatialCSR),
    information: {rowCount: CELLS, transform: 'row' as const, symmetric: 'unknown' as const}
  };
  const cube = rig.input(layoutCube(REFERENCE.cube, layout), 'float32', CELLS * PERIODS);
  const parameters = new GPUParameterBuffer(device, {
    id: 'space-time-parameters',
    format: 'uint32',
    length: GPU_PERMUTATION_PARAMETER_LENGTH,
    values: getGPUPermutationParameterValues({
      seed: SEED,
      permutationCount: PERMUTATIONS,
      significanceLevel: LEVEL
    })
  });
  const exceedances = rig.output('uint32', CELLS);
  const pseudoPValues = rig.output('float32', CELLS);
  const adjustedPValues = rig.output('float32', CELLS);
  const observed = rig.output('float32', CELLS);
  const overflow = rig.output('uint32', 1);
  rig.run(
    new GPUSpaceTimeLocalMoran({
      weights,
      cube,
      layout,
      periodCount: PERIODS,
      period: REFERENCE.period,
      lag: item.lag,
      variant: item.variant,
      direction: item.direction,
      parameters: parameters.importToGraph(rig.graph),
      maximumPermutations: PERMUTATIONS,
      maximumNeighbors: 8,
      islands: 'keep',
      multipleTesting: 'benjamini-yekutieli',
      output: {
        exceedances: exceedances.view,
        pseudoPValues: pseudoPValues.view,
        adjustedPValues: adjustedPValues.view,
        observed: observed.view,
        status: {overflow: overflow.view}
      }
    })
  );
  return {
    exceedances: await readUint32(exceedances.buffer, CELLS),
    pseudoPValues: await readFloat32(pseudoPValues.buffer, CELLS),
    adjustedPValues: await readFloat32(adjustedPValues.buffer, CELLS),
    observed: await readFloat32(observed.buffer, CELLS),
    overflow: (await readUint32(overflow.buffer, 1))[0],
    parameters
  };
}

for (const layout of ['period-major', 'cell-major'] as const) {
  it(`GPUSpaceTimeLocalMoran ${layout} matches esda, the double CPU twin and the Philox twin`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    let totalSignificantRows = 0;
    for (const item of VARIANTS) {
      const rig = new WeightsRig(device);
      const result = await runRecipe(rig, item, layout);
      const reference = REFERENCE.moran[String(item.lag)][item.key];
      const columns = getColumns(item);
      const cpu = computeMoranReference(
        columns.values,
        columns.secondValues ?? columns.values,
        item.variant === 'lagged'
      );
      const twin = computeLocalPermutationOracle({
        weights: createWeights(
          REFERENCE.neighbors.map(row =>
            row.map((neighbor): [number, number] => [neighbor, 1 / REFERENCE.k])
          )
        ),
        rowStandardized: true,
        ...columns,
        statistic: item.variant === 'lagged' ? 'localBivariateMoran' : 'localMoran',
        alternative: 'two-sided',
        islands: 'keep',
        maximumNeighbors: 8,
        multipleTesting: 'benjamini-yekutieli',
        seed: SEED,
        permutationCount: PERMUTATIONS,
        significanceLevel: LEVEL
      });
      expect(result.overflow, `${item.label} overflow`).toBe(0);
      let mismatched = 0;
      let significantRows = 0;
      for (let cell = 0; cell < CELLS; cell++) {
        const label = `${layout} ${item.label} cell ${cell}`;
        // The independent double reference agrees with esda, so it is a valid second oracle.
        expect(Math.abs(cpu[cell] - reference.I[cell]), `${label} cpu vs esda`).toBeLessThan(1e-6);
        expect(
          Math.abs(result.observed[cell] - reference.I[cell]),
          `${label} observed`
        ).toBeLessThanOrEqual(1e-4 + 1e-4 * Math.abs(reference.I[cell]));
        if (result.exceedances[cell] !== twin.exceedances[cell]) {
          mismatched++;
          expect(
            Math.abs(result.exceedances[cell] - twin.exceedances[cell]),
            label
          ).toBeLessThanOrEqual(1);
        }
        expect(result.exceedances[cell]).not.toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
        expect(
          Math.abs(result.pseudoPValues[cell] - reference.pSim[cell]),
          `${label} p`
        ).toBeLessThanOrEqual(P_VALUE_BOUND);
        // Benjamini-Yekutieli never lowers a p-value.
        expect(result.adjustedPValues[cell]).toBeGreaterThanOrEqual(
          result.pseudoPValues[cell] - 1e-6
        );
        if (reference.pSim[cell] <= LEVEL) significantRows++;
      }
      expect(mismatched, `${item.label} rows off the Philox twin`).toBeLessThanOrEqual(2);
      totalSignificantRows += significantRows;
      result.parameters.destroy();
      rig.destroy();
    }
    // The comparison must cover rows on both sides of the significance level.
    expect(totalSignificantRows).toBeGreaterThan(0);
  });
}

it('GPUSpaceTimeLocalMoran refuses a period before its lag and a bad variant', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const weights = rig.uploadWeights(spatialCSR);
  const cube = rig.input(layoutCube(REFERENCE.cube, 'period-major'), 'float32', CELLS * PERIODS);
  const parameters = rig.input(
    new Uint32Array(GPU_PERMUTATION_PARAMETER_LENGTH),
    'uint32',
    GPU_PERMUTATION_PARAMETER_LENGTH
  );
  const build = (overrides: Record<string, unknown>) => () =>
    new GPUSpaceTimeLocalMoran({
      weights,
      cube,
      layout: 'period-major',
      periodCount: PERIODS,
      period: 3,
      variant: 'lagged',
      parameters,
      maximumPermutations: 99,
      output: {
        exceedances: rig.output('uint32', CELLS).view,
        pseudoPValues: rig.output('float32', CELLS).view,
        status: {overflow: rig.output('uint32', 1).view}
      },
      ...overrides
    });
  expect(build({})).not.toThrow();
  expect(build({period: 0})).toThrow(/period/);
  expect(build({period: 3, lag: 4})).toThrow(/period/);
  expect(build({period: PERIODS})).toThrow(/period/);
  expect(build({variant: 'other'})).toThrow(/variant/);
  expect(build({periodCount: PERIODS + 1})).toThrow(/cube/);
  rig.destroy();
});

it('space-time weights compose with the permutation test: pooled lagged Moran on the whole cube', async () => {
  // The algebra's lag-1 matrix makes `localBivariateMoran(values = cube, secondValues = cube)` the
  // lagged Moran of every period at once (permutation drawn over the pooled cube).
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const layout of ['period-major', 'cell-major'] as const) {
    const rig = new WeightsRig(device);
    const rows = CELLS * PERIODS;
    const spatial = {
      ...rig.uploadWeights(spatialCSR),
      information: {rowCount: CELLS, transform: 'row' as const, symmetric: 'unknown' as const}
    };
    const capacity = spatialCSR.neighbors.length * (PERIODS - 1);
    const spaceTimeOutput = rig.weightsOutput(rows, capacity);
    const algebra = new GPUSpatialWeightsAlgebra({
      operation: 'spaceTime',
      weights: spatial,
      periodCount: PERIODS,
      layout,
      lags: [{lag: 1}],
      output: {...spaceTimeOutput.spatialWeights, status: {overflow: rig.output('uint32', 1).view}}
    });
    const cubeValues = layoutCube(REFERENCE.cube, layout);
    const cube = rig.input(cubeValues, 'float32', rows);
    const secondCube = rig.input(cubeValues, 'float32', rows);
    const parameters = new GPUParameterBuffer(device, {
      id: 'pooled-parameters',
      format: 'uint32',
      length: GPU_PERMUTATION_PARAMETER_LENGTH,
      values: getGPUPermutationParameterValues({
        seed: SEED,
        permutationCount: 99,
        significanceLevel: LEVEL
      })
    });
    const observed = rig.output('float32', rows);
    const exceedances = rig.output('uint32', rows);
    const pseudoPValues = rig.output('float32', rows);
    rig.run(
      algebra,
      new GPULocalPermutationTest({
        weights: algebra.outputWeights,
        values: cube,
        secondValues: secondCube,
        statistic: 'localBivariateMoran',
        islands: 'keep',
        parameters: parameters.importToGraph(rig.graph),
        maximumPermutations: 99,
        maximumNeighbors: 8,
        output: {
          exceedances: exceedances.view,
          pseudoPValues: pseudoPValues.view,
          observed: observed.view,
          status: {overflow: rig.output('uint32', 1).view}
        }
      })
    );
    const actual = await readFloat32(observed.buffer, rows);
    const tested = await readUint32(exceedances.buffer, rows);
    // Double reference: pooled z-scores over every row; row (cell, t) lags the neighbors at t - 1.
    const flat: number[] = [];
    for (let cell = 0; cell < CELLS; cell++) {
      for (let period = 0; period < PERIODS; period++) flat.push(REFERENCE.cube[period][cell]);
    }
    const mean = flat.reduce((total, value) => total + value, 0) / rows;
    const deviation = Math.sqrt(
      flat.reduce((total, value) => total + (value - mean) ** 2, 0) / rows
    );
    const z = (cell: number, period: number) => (REFERENCE.cube[period][cell] - mean) / deviation;
    const denominator = flat.reduce((total, value) => total + ((value - mean) / deviation) ** 2, 0);
    for (let cell = 0; cell < CELLS; cell++) {
      for (let period = 0; period < PERIODS; period++) {
        const row = layout === 'period-major' ? period * CELLS + cell : cell * PERIODS + period;
        if (period === 0) {
          // No past period: an island, kept in the moments but never tested.
          expect(tested[row]).toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
          continue;
        }
        const lag =
          REFERENCE.neighbors[cell].reduce(
            (total, neighbor) => total + z(neighbor, period - 1),
            0
          ) / REFERENCE.k;
        const expected = ((rows - 1) * z(cell, period) * lag) / denominator;
        expect(Math.abs(actual[row] - expected), `${layout} row ${row}`).toBeLessThanOrEqual(
          1e-4 + 1e-4 * Math.abs(expected)
        );
        expect(tested[row]).not.toBe(GPU_LOCAL_PERMUTATION_NOT_TESTED);
      }
    }
    parameters.destroy();
    rig.destroy();
  }
});
