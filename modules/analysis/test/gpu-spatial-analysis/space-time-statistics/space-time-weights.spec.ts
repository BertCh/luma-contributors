// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUSpatialWeightsAlgebra,
  type GPUSpaceTimeWeightsLayout,
  type GPUSpatialWeightsSpaceTimeLag
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRandomWeights} from '../spatial-weights/spatial-weights-algebra-oracle';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {assertValidCSR, type OracleCSR} from '../spatial-weights/spatial-weights-oracle';

const CELLS = 7;
const PERIODS = 5;

/** Dense row-by-row spatial matrix of a CSR. */
function toDense(csr: OracleCSR, size: number): number[][] {
  const dense = Array.from({length: size}, () => new Array<number>(size).fill(0));
  for (let row = 0; row < size; row++) {
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
      dense[row][csr.neighbors[slot]] = csr.weights[slot];
    }
  }
  return dense;
}

/**
 * Independent oracle: the dense Kronecker sum `sum_lag T_lag (x) S_lag` with `T_lag[t, s] = weight`
 * where `s = t - lag` and `S_lag = W` (`W` with its diagonal replaced by 1 for `includeSelf`),
 * indexed by the cube layout, then read off as CSR in ascending column order.
 */
function computeSpaceTimeOracle(
  spatial: number[][],
  lags: GPUSpatialWeightsSpaceTimeLag[],
  layout: GPUSpaceTimeWeightsLayout
): OracleCSR {
  const rows = CELLS * PERIODS;
  const indexOf = (cell: number, period: number) =>
    layout === 'period-major' ? period * CELLS + cell : cell * PERIODS + period;
  const dense = Array.from({length: rows}, () => new Array<number>(rows).fill(0));
  const present = Array.from({length: rows}, () => new Array<boolean>(rows).fill(false));
  for (const {lag, weight = 1, includeSelf} of lags) {
    for (let period = 0; period < PERIODS; period++) {
      const source = period - lag;
      if (source < 0 || source >= PERIODS) continue;
      for (let cell = 0; cell < CELLS; cell++) {
        for (let other = 0; other < CELLS; other++) {
          const isSelf = includeSelf && other === cell;
          const spatialValue = isSelf ? 1 : spatial[cell][other];
          if (spatialValue === 0) continue;
          const row = indexOf(cell, period);
          const column = indexOf(other, source);
          dense[row][column] += weight * spatialValue;
          present[row][column] = true;
        }
      }
    }
  }
  const csr: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < rows; column++) {
      if (present[row][column]) {
        csr.neighbors.push(column);
        csr.weights.push(dense[row][column]);
      }
    }
    csr.offsets.push(csr.neighbors.length);
  }
  return csr;
}

async function runSpaceTime(
  spatial: OracleCSR,
  lags: GPUSpatialWeightsSpaceTimeLag[],
  layout: GPUSpaceTimeWeightsLayout,
  capacity: number
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const weights = rig.uploadWeights(spatial, 3);
  const output = rig.weightsOutput(CELLS * PERIODS, capacity);
  const overflow = rig.output('uint32', 1);
  const total = rig.output('uint32', 1);
  const algebra = new GPUSpatialWeightsAlgebra({
    operation: 'spaceTime',
    weights,
    periodCount: PERIODS,
    layout,
    lags,
    output: {...output.spatialWeights, status: {overflow: overflow.view, requiredCount: total.view}}
  });
  rig.run(algebra);
  const result = {
    csr: await output.read(),
    overflow: (await readUint32(overflow.buffer, 1))[0],
    total: (await readUint32(total.buffer, 1))[0],
    information: algebra.outputWeights.information
  };
  rig.destroy();
  return result;
}

function expectSameCSR(actual: OracleCSR, expected: OracleCSR, label: string): void {
  expect(actual.offsets, `${label} offsets`).toEqual(expected.offsets);
  expect(actual.neighbors, `${label} neighbors`).toEqual(expected.neighbors);
  expect(actual.weights.length).toBe(expected.weights.length);
  actual.weights.forEach((weight, slot) => {
    expect(Math.abs(weight - expected.weights[slot]), `${label} weight ${slot}`).toBeLessThan(
      1e-5 * Math.max(1, Math.abs(expected.weights[slot]))
    );
  });
}

const LAG_SETS: Record<string, GPUSpatialWeightsSpaceTimeLag[]> = {
  'contemporaneous plus two lags with temporal weights': [
    {lag: 0},
    {lag: 1, weight: 0.5},
    {lag: 2, weight: 0.25}
  ],
  'pure lag 1': [{lag: 1}],
  'lagged self and a lead': [
    {lag: 1, weight: 2, includeSelf: true},
    {lag: -1, weight: 0.5},
    {lag: 3, includeSelf: true}
  ]
};

for (const layout of ['period-major', 'cell-major'] as const) {
  for (const [name, lags] of Object.entries(LAG_SETS)) {
    it(`GPUSpatialWeightsAlgebra spaceTime ${layout}: ${name} matches the dense Kronecker oracle`, async () => {
      const spatial = createRandomWeights(CELLS, 0.5, 11);
      const expected = computeSpaceTimeOracle(toDense(spatial, CELLS), lags, layout);
      expect(expected.neighbors.length).toBeGreaterThan(CELLS * PERIODS);
      const result = await runSpaceTime(spatial, lags, layout, expected.neighbors.length + 4);
      assertValidCSR(result.csr, CELLS * PERIODS, `${layout} ${name}`);
      expectSameCSR(result.csr, expected, `${layout} ${name}`);
      expect(result.overflow).toBe(0);
      expect(result.total).toBe(expected.neighbors.length);
      expect(result.information?.rowCount).toBe(CELLS * PERIODS);
    });
  }
}

it('GPUSpatialWeightsAlgebra spaceTime replaces a diagonal slot by the unit self entry', async () => {
  // Cell 2 lists itself with weight 5; includeSelf must give 1, a plain lag keeps 5.
  const spatial = createRandomWeights(CELLS, 0.4, 5);
  const dense = toDense(spatial, CELLS);
  dense[2][2] = 5;
  const csr: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
  for (let row = 0; row < CELLS; row++) {
    for (let column = 0; column < CELLS; column++) {
      if (dense[row][column] > 0) {
        csr.neighbors.push(column);
        csr.weights.push(dense[row][column]);
      }
    }
    csr.offsets.push(csr.neighbors.length);
  }
  for (const layout of ['period-major', 'cell-major'] as const) {
    for (const includeSelf of [true, false]) {
      const lags = [{lag: 1, includeSelf}];
      const expected = computeSpaceTimeOracle(dense, lags, layout);
      const result = await runSpaceTime(csr, lags, layout, expected.neighbors.length + 2);
      expectSameCSR(result.csr, expected, `${layout} self ${includeSelf}`);
    }
  }
});

it('GPUSpatialWeightsAlgebra spaceTime flags overflow and keeps a valid prefix', async () => {
  const spatial = createRandomWeights(CELLS, 0.5, 11);
  const lags = [{lag: 0}, {lag: 1}];
  const expected = computeSpaceTimeOracle(toDense(spatial, CELLS), lags, 'period-major');
  const capacity = Math.floor(expected.neighbors.length / 2);
  const result = await runSpaceTime(spatial, lags, 'period-major', capacity);
  expect(result.overflow).toBe(1);
  expect(result.total).toBe(expected.neighbors.length);
  expect(result.csr.offsets[CELLS * PERIODS]).toBeLessThanOrEqual(capacity);
});

it('GPUSpatialWeightsAlgebra spaceTime refuses invalid lags and row counts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const weights = rig.uploadWeights(createRandomWeights(CELLS, 0.5, 11));
  const build = (
    lags: GPUSpatialWeightsSpaceTimeLag[],
    periodCount = PERIODS,
    rows = CELLS * PERIODS
  ) => {
    const output = rig.weightsOutput(rows, 8);
    return () =>
      new GPUSpatialWeightsAlgebra({
        operation: 'spaceTime',
        weights,
        periodCount,
        layout: 'period-major',
        lags,
        output: {...output.spatialWeights, status: {overflow: rig.output('uint32', 1).view}}
      });
  };
  expect(build([{lag: 1}])).not.toThrow();
  expect(build([{lag: 0, includeSelf: true}])).toThrow(/includeSelf/);
  expect(build([{lag: 1}, {lag: 1}])).toThrow(/distinct/);
  expect(build([{lag: 1, weight: 0}])).toThrow(/positive/);
  expect(build([])).toThrow(/lags length/);
  expect(build([{lag: 1}], PERIODS + 1)).toThrow(/cellCount \* periodCount/);
  rig.destroy();
});
