// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  getGPUSpaceTimeParameterValues,
  GPU_JACQUEZ_MAXIMUM_K,
  GPU_LOCAL_KNOX_SUMMARY,
  GPU_LOCAL_KNOX_SUMMARY_LENGTH,
  GPU_NEAR_REPEAT_CELL_SUMMARY,
  GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH,
  GPU_SPACE_TIME_SUMMARY,
  GPU_SPACE_TIME_SUMMARY_LENGTH,
  GPUJacquezTest,
  GPUKnoxTest,
  GPULocalKnox,
  GPUNearRepeatKnox
} from '../../../src/gpu-spatial-analysis/space-time-tests/index';
import fixture from '../../oracles/fixtures/space-time-near-repeat.json';
import {AnalysisRig, createSeededRandom} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';
import {buildBand, createEvents, uploadPairs} from './space-time-scenes';
import {computeSequentialStop} from './space-time-oracle';
import {
  computeJacquezOracle,
  computeLocalKnoxOracle,
  computeNearRepeatOracle,
  getMonteCarloPValueTolerance,
  getNearestNeighborCSR
} from './space-time-statistics-oracle';

/** Lattice events: integer positions and integer times, so pairs sit exactly on band edges. */
function createLatticeEvents(count: number, seed: number) {
  const random = createSeededRandom(seed);
  const positions = new Float32Array(count * 2);
  const times = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    const clustered = index % 3 === 0;
    const center = Math.floor(random() * 3);
    positions[index * 2] = Math.floor(clustered ? center * 8 + random() * 4 : random() * 24);
    positions[index * 2 + 1] = Math.floor(clustered ? center * 6 + random() * 4 : random() * 24);
    times[index] = Math.floor(clustered ? center * 10 + random() * 5 : random() * 30);
  }
  return {positions, times};
}

const fixtureLattice = {
  positions: Float32Array.from(fixture.lattice.positions),
  times: Float32Array.from(fixture.lattice.times)
};

function pValuesAgree(
  actual: number,
  expected: number,
  actualPermutations: number,
  expectedPermutations: number
) {
  return (
    Math.abs(actual - expected) <=
    getMonteCarloPValueTolerance(actual, expected, actualPermutations, expectedPermutations)
  );
}

describe('GPUNearRepeatKnox', () => {
  const run = async (options: {
    positions: Float32Array;
    times: Float32Array;
    distanceEdges: number[];
    timeEdges: number[];
    seed: number;
    permutations: number;
    maximumPermutations?: number;
  }) => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {positions, times, distanceEdges, timeEdges, seed, permutations} = options;
    const maximumPermutations = options.maximumPermutations ?? permutations;
    const csr = buildBand(positions, distanceEdges[distanceEdges.length - 1]);
    const rig = new AnalysisRig(device);
    const cells = (distanceEdges.length - 1) * (timeEdges.length - 1);
    const counts = rig.output('uint32', (maximumPermutations + 1) * cells);
    const summary = rig.output('float32', cells * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH);
    rig.run(
      new GPUNearRepeatKnox({
        pairs: uploadPairs(rig, csr),
        times: rig.input(times, 'float32'),
        parameters: rig.input(
          getGPUSpaceTimeParameterValues({seed, permutationCount: permutations}),
          'uint32'
        ),
        distanceEdges,
        timeEdges,
        maximumPermutations,
        output: {counts, summary}
      })
    );
    const result = {
      csr,
      cells,
      counts: await rig.readUint(counts),
      summary: await rig.readFloat(summary)
    };
    rig.destroy();
    return result;
  };

  it('matches a brute-force double-precision table for every permutation, edges inclusive', async () => {
    for (const {count, seed, permutations, maximumPermutations} of [
      {count: 150, seed: 7, permutations: 60, maximumPermutations: 128},
      {count: 41, seed: 123456789012, permutations: 25, maximumPermutations: 25}
    ]) {
      const {positions, times} = createLatticeEvents(count, 3);
      // Integer coordinates and times put whole bands of pairs exactly on 1, 2, 5 and 10.
      const distanceEdges = [0, 1, 2, 5, 10];
      const timeEdges = [0, 1, 3, 10];
      const actual = await run({
        positions,
        times,
        distanceEdges,
        timeEdges,
        seed,
        permutations,
        maximumPermutations
      });
      const expected = computeNearRepeatOracle(
        actual.csr,
        times,
        seed,
        permutations,
        distanceEdges,
        timeEdges
      );
      expect(expected.tables[0].reduce((sum, value) => sum + value, 0)).toBeGreaterThan(20);
      expect(new Set(expected.tables.map(table => table.join())).size).toBeGreaterThan(5);
      for (let slot = 0; slot <= permutations; slot++) {
        expect(actual.counts.slice(slot * actual.cells, (slot + 1) * actual.cells)).toEqual(
          expected.tables[slot]
        );
      }
      expect(
        actual.counts.slice((permutations + 1) * actual.cells).every(value => value === 0)
      ).toBe(true);
      const index = GPU_NEAR_REPEAT_CELL_SUMMARY;
      for (let cell = 0; cell < actual.cells; cell++) {
        const cellSummary = actual.summary.slice(
          cell * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH,
          (cell + 1) * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH
        );
        expect(cellSummary[index.observed]).toBe(expected.tables[0][cell]);
        expect(cellSummary[index.greaterCount]).toBe(expected.greater[cell]);
        expect(cellSummary[index.lesserCount]).toBe(expected.lesser[cell]);
        expect(cellSummary[index.pseudoPGreater]).toBeCloseTo(
          (expected.greater[cell] + 1) / (permutations + 1),
          6
        );
        expect(cellSummary[index.pseudoPLesser]).toBeCloseTo(
          (expected.lesser[cell] + 1) / (permutations + 1),
          6
        );
        expect(Math.abs(cellSummary[index.permutationMean] - expected.mean[cell])).toBeLessThan(
          1e-4 * (1 + expected.mean[cell])
        );
        if (expected.mean[cell] > 0) {
          expect(cellSummary[index.knoxRatio]).toBeCloseTo(
            expected.tables[0][cell] / expected.mean[cell],
            4
          );
        } else {
          expect(cellSummary[index.knoxRatio]).toBeNaN();
        }
      }
    }
  });

  it('reduces to GPUKnoxTest when the table has one cell, on the same permutations', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {positions, times} = createEvents(240, 5);
    const radius = 9;
    const threshold = 3;
    const seed = 41;
    const permutations = 50;
    const table = await run({
      positions,
      times,
      distanceEdges: [0, radius],
      timeEdges: [0, threshold],
      seed,
      permutations,
      maximumPermutations: 64
    });
    const rig = new AnalysisRig(device);
    const statistics = rig.output('uint32', 65);
    rig.run(
      new GPUKnoxTest({
        pairs: rig.weights(table.csr, 4),
        times: rig.input(times, 'float32'),
        parameters: rig.input(
          getGPUSpaceTimeParameterValues({
            seed,
            permutationCount: permutations,
            timeThreshold: threshold
          }),
          'uint32'
        ),
        maximumPermutations: 64,
        output: {statistics, summary: rig.output('float32', GPU_SPACE_TIME_SUMMARY_LENGTH)}
      })
    );
    const knox = await rig.readUint(statistics);
    rig.destroy();
    expect(knox[0]).toBeGreaterThan(0);
    expect(table.counts.slice(0, permutations + 1)).toEqual(knox.slice(0, permutations + 1));
  });

  it('agrees with the pointpats Knox runs of the lattice fixture (cell counts exact, null statistical)', async () => {
    const {table} = fixture.lattice;
    const permutations = 3999;
    const actual = await run({
      ...fixtureLattice,
      distanceEdges: table.spatialEdges,
      timeEdges: table.temporalEdges,
      seed: 20260,
      permutations,
      maximumPermutations: permutations
    });
    const index = GPU_NEAR_REPEAT_CELL_SUMMARY;
    expect(actual.cells).toBe(table.observed.length);
    for (let cell = 0; cell < actual.cells; cell++) {
      const cellSummary = actual.summary.slice(
        cell * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH,
        (cell + 1) * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH
      );
      expect(cellSummary[index.observed], `cell ${cell} observed`).toBe(table.observed[cell]);
      // The permutation mean has a standard error of sd / sqrt(P) on each side.
      const standardError = Math.sqrt(
        table.permutationVariance[cell] / permutations +
          table.permutationVariance[cell] / table.permutations
      );
      expect(
        Math.abs(cellSummary[index.permutationMean] - table.permutationMean[cell]),
        `cell ${cell} mean`
      ).toBeLessThanOrEqual(4.5 * standardError + 1e-6);
      expect(
        pValuesAgree(
          cellSummary[index.pseudoPGreater],
          table.pseudoPGreater[cell],
          permutations,
          table.permutations
        ),
        `cell ${cell} p ${cellSummary[index.pseudoPGreater]} vs ${table.pseudoPGreater[cell]}`
      ).toBe(true);
    }
    // At least one cell must carry real evidence, or the p-value check says nothing.
    expect(Math.min(...table.pseudoPGreater)).toBeLessThan(0.05);
    expect(Math.max(...table.pseudoPGreater)).toBeGreaterThan(0.3);
  });
});

describe('GPULocalKnox', () => {
  const run = async (options: {
    positions: Float32Array;
    times: Float32Array;
    radius: number;
    threshold: number;
    seed: number;
    permutations: number;
    maximumPermutations: number;
  }) => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {positions, times, radius, threshold, seed, permutations, maximumPermutations} = options;
    const csr = buildBand(positions, radius);
    const rig = new AnalysisRig(device);
    const summary = rig.output('float32', times.length * GPU_LOCAL_KNOX_SUMMARY_LENGTH);
    rig.run(
      new GPULocalKnox({
        pairs: rig.weights(csr, 4),
        times: rig.input(times, 'float32'),
        parameters: rig.input(
          getGPUSpaceTimeParameterValues({
            seed,
            permutationCount: permutations,
            timeThreshold: threshold
          }),
          'uint32'
        ),
        maximumPermutations,
        output: {summary}
      })
    );
    const values = await rig.readFloat(summary);
    rig.destroy();
    const field = (name: keyof typeof GPU_LOCAL_KNOX_SUMMARY) =>
      Array.from(
        {length: times.length},
        (_, row) => values[row * GPU_LOCAL_KNOX_SUMMARY_LENGTH + GPU_LOCAL_KNOX_SUMMARY[name]]
      );
    return {csr, field};
  };

  it('matches the conditional-permutation oracle exactly', async () => {
    for (const {count, radius, threshold, seed, permutations} of [
      {count: 130, radius: 6, threshold: 4, seed: 5, permutations: 30},
      {count: 45, radius: 40, threshold: 25.5, seed: 987654321, permutations: 17}
    ]) {
      const {positions, times} = createEvents(count, 8);
      const actual = await run({
        positions,
        times,
        radius,
        threshold,
        seed,
        permutations,
        maximumPermutations: 64
      });
      const expected = computeLocalKnoxOracle(actual.csr, times, seed, permutations, threshold);
      expect(new Set(expected.observed).size).toBeGreaterThan(2);
      expect(new Set(expected.greater).size).toBeGreaterThan(2);
      expect(actual.field('observed')).toEqual(expected.observed);
      expect(actual.field('greaterCount')).toEqual(expected.greater);
      expect(actual.field('lesserCount')).toEqual(expected.lesser);
      actual.field('permutationMean').forEach((mean, row) => {
        expect(Math.abs(mean - expected.mean[row])).toBeLessThan(1e-4 * (1 + expected.mean[row]));
      });
      actual.field('pseudoPGreater').forEach((p, row) => {
        expect(p).toBeCloseTo((expected.greater[row] + 1) / (permutations + 1), 6);
      });
    }
  });

  it('agrees with pointpats KnoxLocal on the lattice fixture (counts exact, null statistical)', async () => {
    const {local} = fixture.lattice;
    const permutations = 3999;
    const actual = await run({
      ...fixtureLattice,
      radius: local.delta,
      threshold: local.tau,
      seed: 777,
      permutations,
      maximumPermutations: permutations
    });
    expect(actual.field('observed')).toEqual(local.observed);
    expect(Math.max(...local.observed)).toBeGreaterThan(1);
    const mean = actual.field('permutationMean');
    const pGreater = actual.field('pseudoPGreater');
    local.observed.forEach((_, row) => {
      expect(Math.abs(mean[row] - local.permutationMean[row]), `event ${row} mean`).toBeLessThan(
        0.15 + 0.1 * local.permutationMean[row]
      );
      expect(
        pValuesAgree(pGreater[row], local.pseudoPGreater[row], permutations, local.permutations),
        `event ${row} p ${pGreater[row]} vs ${local.pseudoPGreater[row]}`
      ).toBe(true);
    });
    // Both must see the same events as unusual, not just similar numbers.
    expect(pGreater.filter(p => p < 0.05).length).toBeGreaterThan(0);
    expect(local.pseudoPGreater.filter(p => p < 0.05).length).toBeGreaterThan(0);
  });
});

describe('GPUJacquezTest', () => {
  const run = async (options: {
    positions: Float32Array;
    times: Float32Array;
    k: number;
    seed: number;
    permutations: number;
    maximumPermutations: number;
    stopAfterExceedances?: number;
  }) => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {positions, times, k, seed, permutations, maximumPermutations} = options;
    const csr = getNearestNeighborCSR(positions, k);
    const rig = new AnalysisRig(device);
    const statistics = rig.output('uint32', maximumPermutations + 1);
    const summary = rig.output('float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
    const permutationCount =
      options.stopAfterExceedances === undefined ? undefined : rig.output('uint32', 1);
    rig.run(
      new GPUJacquezTest({
        neighbors: rig.weights(csr, 4),
        times: rig.input(times, 'float32'),
        parameters: rig.input(
          getGPUSpaceTimeParameterValues({seed, permutationCount: permutations}),
          'uint32'
        ),
        k,
        maximumPermutations,
        ...(options.stopAfterExceedances === undefined
          ? {}
          : {stopAfterExceedances: options.stopAfterExceedances}),
        output: {statistics, summary, permutationCount}
      })
    );
    const result = {
      csr,
      statistics: await rig.readUint(statistics),
      summary: await rig.readFloat(summary),
      permutationsUsed: permutationCount ? (await rig.readUint(permutationCount))[0] : undefined
    };
    rig.destroy();
    return result;
  };

  it('matches the brute-force k-nearest-neighbor oracle for every permutation', async () => {
    for (const {count, k, seed, permutations} of [
      {count: 180, k: 3, seed: 11, permutations: 40},
      {count: 60, k: 1, seed: 5, permutations: 20},
      {count: 33, k: 7, seed: 123456789012, permutations: 15}
    ]) {
      const {positions, times} = createEvents(count, 6);
      const actual = await run({positions, times, k, seed, permutations, maximumPermutations: 64});
      const expected = computeJacquezOracle(actual.csr, times, k, seed, permutations);
      expect(new Set(expected).size).toBeGreaterThan(3);
      expect(actual.statistics.slice(0, permutations + 1)).toEqual(expected);
      expect(actual.statistics.slice(permutations + 1).every(value => value === 0)).toBe(true);
      const index = GPU_SPACE_TIME_SUMMARY;
      const permuted = expected.slice(1);
      const greater = permuted.filter(value => value >= expected[0]).length;
      expect(actual.summary[index.observed]).toBe(expected[0]);
      expect(actual.summary[index.pairCount]).toBe(count * k);
      expect(actual.summary[index.timeClosePairs]).toBe(count * k);
      expect(actual.summary[index.expected]).toBeCloseTo((count * k * k) / (count - 1), 3);
      expect(actual.summary[index.greaterCount]).toBe(greater);
      expect(actual.summary[index.pseudoPGreater]).toBeCloseTo(
        (greater + 1) / (permutations + 1),
        6
      );
    }
  });

  it('stops by Besag-Clifford exactly as the sequential rule on the oracle statistics', async () => {
    const {positions, times} = createEvents(120, 14);
    const k = 3;
    const seed = 9;
    const permutations = 200;
    const full = computeJacquezOracle(
      getNearestNeighborCSR(positions, k),
      times,
      k,
      seed,
      permutations
    );
    for (const stopAfterExceedances of [3, 15]) {
      const rule = computeSequentialStop(full, permutations, stopAfterExceedances);
      const actual = await run({
        positions,
        times,
        k,
        seed,
        permutations,
        maximumPermutations: 256,
        stopAfterExceedances
      });
      expect(actual.permutationsUsed).toBe(rule.permutationsUsed);
      expect(actual.summary[GPU_SPACE_TIME_SUMMARY.pseudoPGreater]).toBeCloseTo(rule.pValue, 6);
      expect(actual.statistics.slice(0, rule.permutationsUsed + 1)).toEqual(
        full.slice(0, rule.permutationsUsed + 1)
      );
      expect(actual.statistics.slice(rule.permutationsUsed + 1).every(value => value === 0)).toBe(
        true
      );
    }
  });

  it('agrees with pointpats jacquez on the fixture (statistic exact, p-value statistical)', async () => {
    const {jacquez} = fixture;
    const permutations = 3999;
    const actual = await run({
      positions: Float32Array.from(jacquez.positions),
      times: Float32Array.from(jacquez.times),
      k: jacquez.k,
      seed: 4242,
      permutations,
      maximumPermutations: permutations
    });
    const index = GPU_SPACE_TIME_SUMMARY;
    expect(actual.statistics[0]).toBe(jacquez.observed);
    expect(actual.summary[index.observed]).toBe(jacquez.observed);
    // The analytic null mean n k^2 / (n - 1) is independent of both implementations.
    const permuted = actual.statistics.slice(1, permutations + 1);
    const mean = permuted.reduce((sum, value) => sum + value, 0) / permutations;
    const variance =
      permuted.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (permutations - 1);
    expect(Math.abs(mean - jacquez.expectedUnderNull)).toBeLessThan(
      4.5 * Math.sqrt(variance / permutations)
    );
    expect(actual.summary[index.expected]).toBeCloseTo(jacquez.expectedUnderNull, 3);
    expect(
      pValuesAgree(
        actual.summary[index.pseudoPGreater],
        jacquez.pseudoPGreater,
        permutations,
        jacquez.permutations
      ),
      `p ${actual.summary[index.pseudoPGreater]} vs ${jacquez.pseudoPGreater}`
    ).toBe(true);
    expect(jacquez.observed).toBeGreaterThan(jacquez.expectedUnderNull);
    expect(GPU_JACQUEZ_MAXIMUM_K).toBeGreaterThanOrEqual(jacquez.k);
  });
});
