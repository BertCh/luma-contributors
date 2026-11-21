// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {recordWGSLKernelNodes} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPU_LOCAL_KNOX_SUMMARY_LENGTH,
  GPULocalKnox
} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-local-knox';
import {
  GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH,
  GPUNearRepeatKnox
} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-near-repeat-knox';
import {getGPUSpaceTimeParameterValues} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

// PTS-104. Three events, times [0, 1, 10]. The pair list holds both directions of the one close
// pair (0, 1) at distance 0.5; event 2 has no spatial neighbor. Hand-worked observed values:
// - local Knox with timeThreshold 2: event 0 counts neighbor 1 (|0 - 1| <= 2), event 1 counts
//   neighbor 0, event 2 counts nothing, so the observed words are [1, 1, 0];
// - near-repeat with distance edges [0, 1, 2] and time edges [0, 2, 20]: the one pair falls in
//   distance band 0 (0.5 <= 1) and time band 0 (1 <= 2), so the observed table (slot 0 of
//   `counts`, cell = distanceBand * 2 + timeBand) is [1, 0, 0, 0].
// Either pair loss flag must suppress every summary word (quiet NaN) and, for near-repeat, zero
// the count tables, as GPUKnoxTest does; clearing the flag restores the ordinary result on the
// same compiled nodes.
// The CPU interpreter cannot run the cooperative (workgroup barrier) permutation kernels, so they
// are left out: local Knox then has zero permutation accumulators (mean 0, greater and lesser
// counts 0, p-values (0 + 1) / (2 + 1)), and near-repeat gets a stand-in histogram written by the
// test. The headless spec local-knox-pairs-loss.spec.ts runs the whole graph on a device.
const OFFSETS = Uint32Array.of(0, 1, 2, 2);
const NEIGHBORS = Uint32Array.of(1, 0);
const WEIGHTS = Float32Array.of(1, 1);
const DISTANCES = Float32Array.of(0.5, 0.5);
const TIMES = Float32Array.of(0, 1, 10);
const MAXIMUM_PERMUTATIONS = 2;
const FLAG_STATES = [
  {overflow: 0, candidateOverflow: 1},
  {overflow: 0, candidateOverflow: 0},
  {overflow: 1, candidateOverflow: 0},
  {overflow: 0, candidateOverflow: 0}
];

function createScene(graph: GPUCommandGraph) {
  const view = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const overflow = view('pairs-overflow', 'uint32', 1);
  const candidateOverflow = view('pairs-candidate-overflow', 'uint32', 1);
  const pairs = {
    offsets: view('offsets', 'uint32', 4),
    neighbors: view('neighbors', 'uint32', 2),
    weights: view('weights', 'float32', 2),
    distances: view('distances', 'float32', 2),
    status: {overflow, candidateOverflow}
  };
  const times = view('times', 'float32', 3);
  const parameters = view('parameters', 'uint32', 4);
  const memory = createCPUGraphMemory();
  memory.write(pairs.offsets, OFFSETS);
  memory.write(pairs.neighbors, NEIGHBORS);
  memory.write(pairs.weights, WEIGHTS);
  memory.write(pairs.distances, DISTANCES);
  memory.write(times, TIMES);
  memory.write(
    parameters,
    getGPUSpaceTimeParameterValues({seed: 7, permutationCount: 2, timeThreshold: 2})
  );
  return {view, overflow, candidateOverflow, pairs, times, parameters, memory};
}

it('PTS-104 local Knox suppresses on candidateOverflow alone and restores when it clears', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  try {
    const scene = createScene(graph);
    const summary = scene.view('summary', 'float32', 3 * GPU_LOCAL_KNOX_SUMMARY_LENGTH);
    const nodes = new GPULocalKnox({
      pairs: scene.pairs,
      times: scene.times,
      parameters: scene.parameters,
      maximumPermutations: MAXIMUM_PERMUTATIONS,
      output: {summary}
    })
      .getCommandNodes(graph)
      .filter(node => !node.id.endsWith('-permutations'));
    expect(nodes.filter(node => node.id.includes('-incomplete-'))).toHaveLength(2);
    for (const state of FLAG_STATES) {
      scene.memory.write(scene.overflow, Uint32Array.of(state.overflow));
      scene.memory.write(scene.candidateOverflow, Uint32Array.of(state.candidateOverflow));
      scene.memory.write(summary, new Float32Array(summary.length).fill(-7));
      executeCommandNodesOnCPU(nodes, scene.memory);
      const words = Array.from(scene.memory.readFloat32(summary));
      if (state.overflow || state.candidateOverflow) {
        expect(words.every(Number.isNaN), JSON.stringify(state)).toBe(true);
      } else {
        expect(words.every(Number.isFinite)).toBe(true);
        // [observed, permutationMean, greater, lesser, pGreater, pLesser] per event.
        expect(words).toEqual(
          [1, 1, 0].flatMap(observed => [observed, 0, 0, 0, Math.fround(1 / 3), Math.fround(1 / 3)])
        );
      }
    }
  } finally {
    device.destroy();
  }
});

it('PTS-104 near-repeat Knox suppresses on candidateOverflow alone and restores when it clears', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  try {
    const scene = createScene(graph);
    const cells = 4;
    const counts = scene.view('counts', 'uint32', (MAXIMUM_PERMUTATIONS + 1) * cells);
    const summary = scene.view('summary', 'float32', cells * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH);
    const nodes = new GPUNearRepeatKnox({
      pairs: scene.pairs,
      times: scene.times,
      parameters: scene.parameters,
      distanceEdges: [0, 1, 2],
      timeEdges: [0, 2, 20],
      maximumPermutations: MAXIMUM_PERMUTATIONS,
      output: {counts, summary}
    })
      .getCommandNodes(graph)
      .filter(node => !node.id.endsWith('-histogram'));
    expect(nodes.filter(node => node.id.includes('-incomplete-'))).toHaveLength(2);
    // Stand-in histogram: the observed table from the hand-worked pair, then two permuted tables
    // that each hold the one pair (cell 0, then cell 1).
    const standIn = Uint32Array.of(1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0);
    for (const state of FLAG_STATES) {
      scene.memory.write(scene.overflow, Uint32Array.of(state.overflow));
      scene.memory.write(scene.candidateOverflow, Uint32Array.of(state.candidateOverflow));
      scene.memory.write(counts, standIn);
      scene.memory.write(summary, new Float32Array(summary.length).fill(-7));
      executeCommandNodesOnCPU(nodes, scene.memory);
      const table = Array.from(scene.memory.readUint32(counts));
      const words = Array.from(scene.memory.readFloat32(summary));
      if (state.overflow || state.candidateOverflow) {
        expect(table, JSON.stringify(state)).toEqual(new Array(counts.length).fill(0));
        expect(words.every(Number.isNaN), JSON.stringify(state)).toBe(true);
      } else {
        expect(table).toEqual(Array.from(standIn));
        expect([0, 1, 2, 3].map(cell => words[cell * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH])).toEqual(
          [1, 0, 0, 0]
        );
        // Cell 0: permuted counts [1, 0], mean 0.5, Knox ratio 1 / 0.5 = 2.
        expect(words[1]).toBe(0.5);
        expect(words[7]).toBe(2);
      }
    }
  } finally {
    device.destroy();
  }
});
