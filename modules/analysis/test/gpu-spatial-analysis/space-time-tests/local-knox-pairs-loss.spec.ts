// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPULocalKnox,
  GPUNearRepeatKnox,
  getGPUSpaceTimeParameterValues,
  GPU_LOCAL_KNOX_SUMMARY_LENGTH,
  GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH
} from '../../../src/gpu-spatial-analysis/space-time-tests/index';
import {
  getGPUEmergingHotSpotsParameterValues,
  GPUEmergingHotSpots,
  GPU_EMERGING_HOT_SPOTS_CATEGORIES,
  GPU_EMERGING_HOT_SPOTS_NO_SLICE
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {computeSpaceTimeGiStarWeighted} from '../emerging-hot-spots/emerging-hot-spots-oracle';

// PTS-104 on a device: one compiled graph per contributor, submitted with candidateOverflow
// set, cleared, overflow set and cleared. Fixture and hand-worked values as in
// local-knox-pairs-loss.node.spec.ts: times [0, 1, 10], the one close pair (0, 1) listed both
// ways at distance 0.5; local Knox observed [1, 1, 0]; near-repeat observed table [1, 0, 0, 0].
const FLAG_STATES = [
  {overflow: 0, candidateOverflow: 1},
  {overflow: 0, candidateOverflow: 0},
  {overflow: 1, candidateOverflow: 0},
  {overflow: 0, candidateOverflow: 0}
];

function createLossFlags(rig: AnalysisRig) {
  const create = (id: string) =>
    new GPUParameterBuffer(rig.device, {
      id,
      format: 'uint32',
      length: 1,
      values: Uint32Array.of(0)
    });
  const overflow = create('overflow');
  const candidateOverflow = create('candidate-overflow');
  return {
    overflow,
    candidateOverflow,
    status: {
      overflow: overflow.importToGraph(rig.graph),
      candidateOverflow: candidateOverflow.importToGraph(rig.graph)
    },
    write(state: {overflow: number; candidateOverflow: number}) {
      overflow.write(Uint32Array.of(state.overflow));
      candidateOverflow.write(Uint32Array.of(state.candidateOverflow));
    },
    destroy() {
      overflow.destroy();
      candidateOverflow.destroy();
    }
  };
}

function createPairs(rig: AnalysisRig, status: object) {
  return {
    offsets: rig.input(Uint32Array.of(0, 1, 2, 2), 'uint32'),
    neighbors: rig.input(Uint32Array.of(1, 0), 'uint32'),
    weights: rig.input(Float32Array.of(1, 1), 'float32'),
    distances: rig.input(Float32Array.of(0.5, 0.5), 'float32'),
    status
  };
}

it('PTS-104 local and near-repeat Knox suppress on pair loss and recover on one graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const flags = createLossFlags(rig);
  let compiled: ReturnType<typeof rig.graph.compile> | undefined;
  try {
    const pairs = createPairs(rig, flags.status);
    const times = rig.input(Float32Array.of(0, 1, 10), 'float32');
    const parameters = rig.input(
      getGPUSpaceTimeParameterValues({seed: 11, permutationCount: 9, timeThreshold: 2}),
      'uint32'
    );
    const localSummary = rig.output('float32', 3 * GPU_LOCAL_KNOX_SUMMARY_LENGTH);
    const counts = rig.output('uint32', 10 * 4);
    const cellSummary = rig.output('float32', 4 * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH);
    rig.graph.add(
      new GPULocalKnox({
        pairs,
        times,
        parameters,
        maximumPermutations: 9,
        output: {summary: localSummary}
      })
    );
    rig.graph.add(
      new GPUNearRepeatKnox({
        id: 'near-repeat',
        pairs,
        times,
        parameters,
        distanceEdges: [0, 1, 2],
        timeEdges: [0, 2, 20],
        maximumPermutations: 9,
        output: {counts, summary: cellSummary}
      })
    );
    compiled = rig.graph.compile();
    for (const state of FLAG_STATES) {
      flags.write(state);
      submitGraph(device, compiled, undefined);
      const local = await rig.readFloat(localSummary);
      const table = await rig.readUint(counts);
      const cells = await rig.readFloat(cellSummary);
      const label = JSON.stringify(state);
      if (state.overflow || state.candidateOverflow) {
        expect(local.every(Number.isNaN), label).toBe(true);
        expect(table, label).toEqual(new Array(40).fill(0));
        expect(cells.every(Number.isNaN), label).toBe(true);
      } else {
        expect(local.every(Number.isFinite), label).toBe(true);
        expect([0, 1, 2].map(event => local[event * GPU_LOCAL_KNOX_SUMMARY_LENGTH])).toEqual([
          1, 1, 0
        ]);
        expect(table.slice(0, 4)).toEqual([1, 0, 0, 0]);
        for (let slot = 1; slot <= 9; slot++)
          expect(table.slice(slot * 4, slot * 4 + 4).reduce((sum, count) => sum + count, 0)).toBe(
            1
          );
        expect([0, 1, 2, 3].map(cell => cells[cell * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH])).toEqual(
          [1, 0, 0, 0]
        );
      }
    }
  } finally {
    compiled?.destroy();
    flags.destroy();
    rig.destroy();
  }
});

it('PTS-104 emerging hot spots weights mode suppresses on weights loss and recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const flags = createLossFlags(rig);
  let compiled: ReturnType<typeof rig.graph.compile> | undefined;
  try {
    // Three cells on a path 0 - 1 - 2, three slices; values indexed cell * 3 + slice.
    const csr = {offsets: [0, 1, 3, 4], neighbors: [1, 0, 2, 1], weights: [1, 1, 1, 1]};
    const values = Float32Array.of(1, 2, 3, 4, 5, 6, 2, 8, 1);
    // minimumTrendSliceCount 0 keeps the three-slice cells out of the ordinary UNDETERMINED rule.
    const settings = {
      temporalWindow: 1,
      multipleTesting: 'none' as const,
      criticalZ: 1.96,
      minimumTrendSliceCount: 0
    };
    const packed = getGPUEmergingHotSpotsParameterValues(settings);
    const output = {
      giZScores: rig.output('float32', 9),
      lastValidSlice: rig.output('uint32', 3),
      trendZ: rig.output('float32', 3),
      trendP: rig.output('float32', 3),
      trendS: rig.output('sint32', 3),
      category: rig.output('uint32', 3),
      hotSliceCount: rig.output('uint32', 3),
      coldSliceCount: rig.output('uint32', 3),
      globalStatistics: rig.output('float32', 4)
    };
    rig.graph.add(
      new GPUEmergingHotSpots({
        values: rig.input(values, 'float32'),
        weights: {
          offsets: rig.input(Uint32Array.from(csr.offsets), 'uint32'),
          neighbors: rig.input(Uint32Array.from(csr.neighbors), 'uint32'),
          weights: rig.input(Float32Array.from(csr.weights), 'float32'),
          status: flags.status
        },
        sliceCount: 3,
        multipleTesting: 'none',
        parameters: rig.input(packed, 'float32'),
        output
      })
    );
    compiled = rig.graph.compile();
    // Independent float64 weighted Gi* oracle (focal weight 1, window of one previous slice).
    const expectedZ = computeSpaceTimeGiStarWeighted(
      {gridWidth: 3, gridHeight: 1, sliceCount: 3, values},
      csr,
      1,
      packed
    );
    expect(Array.from(expectedZ).every(Number.isFinite)).toBe(true);
    for (const state of FLAG_STATES) {
      flags.write(state);
      submitGraph(device, compiled, undefined);
      const label = JSON.stringify(state);
      const zScores = await rig.readFloat(output.giZScores);
      const globalStatistics = await rig.readFloat(output.globalStatistics);
      expect(globalStatistics[0], label).toBe(9);
      if (state.overflow || state.candidateOverflow) {
        expect(zScores.every(Number.isNaN), label).toBe(true);
        expect((await rig.readFloat(output.trendZ)).every(Number.isNaN), label).toBe(true);
        expect((await rig.readFloat(output.trendP)).every(Number.isNaN), label).toBe(true);
        expect(await rig.readUint(output.hotSliceCount), label).toEqual([0, 0, 0]);
        expect(await rig.readUint(output.coldSliceCount), label).toEqual([0, 0, 0]);
        expect(await rig.readUint(output.lastValidSlice), label).toEqual(
          new Array(3).fill(GPU_EMERGING_HOT_SPOTS_NO_SLICE)
        );
        expect(await rig.readUint(output.category), label).toEqual(
          new Array(3).fill(GPU_EMERGING_HOT_SPOTS_CATEGORIES.UNDETERMINED)
        );
      } else {
        zScores.forEach((z, bin) => {
          expect(Number.isFinite(z), `${label} bin ${bin}`).toBe(true);
          expect(Math.abs(z - expectedZ[bin]), `${label} bin ${bin}`).toBeLessThan(1e-4);
        });
        expect(await rig.readUint(output.lastValidSlice), label).toEqual([2, 2, 2]);
        expect(
          (await rig.readUint(output.category)).every(
            category => category !== GPU_EMERGING_HOT_SPOTS_CATEGORIES.UNDETERMINED
          ),
          label
        ).toBe(true);
      }
    }
  } finally {
    compiled?.destroy();
    flags.destroy();
    rig.destroy();
  }
}, 60000);
