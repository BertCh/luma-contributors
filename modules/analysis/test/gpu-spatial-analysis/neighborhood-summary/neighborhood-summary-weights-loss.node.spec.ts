// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {recordWGSLKernelNodes} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPU_NEIGHBORHOOD_SUMMARY_NO_MODE,
  GPUNeighborhoodSummary
} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import {GPUNeighborhoodSummaryOperation} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary-operation';
import {
  computeNeighborhoodSummaryOnCPU,
  CPUNeighborhoodSummaryOperation
} from '../../../src/gpu-spatial-analysis/neighborhood-summary/cpu-neighborhood-summary-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

// PTS-101. Three rows: 0 -> 1 and 1 -> 0 with weight 1, row 2 has no neighbors.
// Values [10, 20, 30], categories [5, 7, 9]. Hand-worked ordinary result:
// row 0: count 1, mean 20 (its one neighbor), mode 7, entropy 0 (one category);
// row 1: count 1, mean 10, mode 5, entropy 0;
// row 2: count 0, mean NaN (zero weight sum), mode NO_MODE, entropy NaN.
// With either loss flag set, every output is NaN (count too) and every mode is NO_MODE.
const OFFSETS = Uint32Array.of(0, 1, 2, 2);
const NEIGHBORS = Uint32Array.of(1, 0);
const WEIGHTS = Float32Array.of(1, 1);
const VALUES = Float32Array.of(10, 20, 30);
const CATEGORIES = Uint32Array.of(5, 7, 9);
const ORDINARY_TABLE = [1, 20, 1, 10, 0, NaN];
const ORDINARY_MODES = [7, 5, GPU_NEIGHBORHOOD_SUMMARY_NO_MODE];
const ORDINARY_ENTROPY = [0, 0, NaN];
const NO_MODES = [0, 0, 0].map(() => GPU_NEIGHBORHOOD_SUMMARY_NO_MODE);
const FLAG_STATES = [
  {overflow: 0, candidateOverflow: 0, complete: true},
  {overflow: 0, candidateOverflow: 1, complete: false},
  {overflow: 0, candidateOverflow: 0, complete: true},
  {overflow: 1, candidateOverflow: 0, complete: false},
  {overflow: 0, candidateOverflow: 0, complete: true}
];

function expectSameNumbers(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, index) => {
    if (Number.isNaN(value)) expect(Number.isNaN(actual[index]), `index ${index}`).toBe(true);
    else expect(actual[index], `index ${index}`).toBe(value);
  });
}

it('PTS-101 GPU summary suppresses on candidateOverflow alone and restores when it clears', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  try {
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const overflow = view('weights-overflow', 'uint32', 1);
    const candidateOverflow = view('weights-candidate-overflow', 'uint32', 1);
    const input = {
      offsets: view('offsets', 'uint32', 4),
      neighbors: view('neighbors', 'uint32', 2),
      weights: view('weights', 'float32', 2),
      values: view('values', 'float32', 3),
      categories: view('categories', 'uint32', 3)
    };
    const output = {
      table: view('table', 'float32', 6),
      modes: view('modes', 'uint32', 3),
      entropy: view('entropy', 'float32', 3)
    };
    const nodes = new GPUNeighborhoodSummary({
      weights: {
        offsets: input.offsets,
        neighbors: input.neighbors,
        weights: input.weights,
        status: {overflow, candidateOverflow}
      },
      values: input.values,
      categories: input.categories,
      statistics: ['count', 'mean'],
      output
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(input.offsets, OFFSETS);
    memory.write(input.neighbors, NEIGHBORS);
    memory.write(input.weights, WEIGHTS);
    memory.write(input.values, VALUES);
    memory.write(input.categories, CATEGORIES);
    // One compiled node list, flags rewritten between submissions (set, clear, set, clear).
    for (const state of FLAG_STATES) {
      memory.write(overflow, Uint32Array.of(state.overflow));
      memory.write(candidateOverflow, Uint32Array.of(state.candidateOverflow));
      memory.write(output.table, new Float32Array(6).fill(-7));
      memory.write(output.modes, new Uint32Array(3).fill(99));
      memory.write(output.entropy, new Float32Array(3).fill(-7));
      executeCommandNodesOnCPU(nodes, memory);
      expectSameNumbers(
        memory.readFloat32(output.table),
        state.complete ? ORDINARY_TABLE : new Array(6).fill(NaN)
      );
      expect(Array.from(memory.readUint32(output.modes))).toEqual(
        state.complete ? ORDINARY_MODES : NO_MODES
      );
      expectSameNumbers(
        memory.readFloat32(output.entropy),
        state.complete ? ORDINARY_ENTROPY : [NaN, NaN, NaN]
      );
    }
  } finally {
    device.destroy();
  }
});

for (const mutation of ['replace-status', 'remove-chunk'] as const) {
  it(`W5-18 captures child loss views before ${mutation} and recovers from live flag contents`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const offsets = createTransientView(graph, 'offsets', 'uint32', 3);
      const neighbors = createTransientView(graph, 'neighbors', 'uint32', 1);
      const weights = createTransientView(graph, 'weights', 'float32', 1);
      const values = createTransientView(graph, 'values', 'float32', 2);
      const candidateOverflow = createTransientView(graph, 'child-loss', 'uint32', 1);
      const replacement = createTransientView(graph, 'replacement', 'uint32', 1);
      // Whole weights that also carry child loss metadata (as chunked weights do): the public
      // GPUSpatialWeights type has no chunks, so the extension is declared here and the
      // object is passed as a variable, not a fresh literal.
      type WeightsWithChildLoss = GPUSpatialWeights & {
        chunks: {weights: {status: {candidateOverflow: GraphDataView<'uint32'>}}}[];
      };
      const chunks: WeightsWithChildLoss['chunks'] = [{weights: {status: {candidateOverflow}}}];
      const weightsWithChildLoss: WeightsWithChildLoss = {offsets, neighbors, weights, chunks};
      const table = createTransientView(graph, 'table', 'float32', 4);
      const contributor = new GPUNeighborhoodSummary({
        weights: weightsWithChildLoss,
        values,
        statistics: ['count', 'mean'],
        output: {table}
      });
      if (mutation === 'replace-status')
        chunks[0].weights.status = {candidateOverflow: replacement};
      else chunks.splice(0);
      const nodes = contributor.getCommandNodes(graph);
      const memory = createCPUGraphMemory();
      memory.write(offsets, Uint32Array.of(0, 1, 1));
      memory.write(neighbors, Uint32Array.of(1));
      memory.write(weights, Float32Array.of(1));
      memory.write(values, Float32Array.of(10, 20));
      for (const flag of [1, 0, 1, 0]) {
        memory.write(candidateOverflow, Uint32Array.of(flag));
        executeCommandNodesOnCPU(nodes, memory);
        expectSameNumbers(memory.readFloat32(table), flag ? [NaN, NaN, NaN, NaN] : [1, 20, 0, NaN]);
      }
    } finally {
      device.destroy();
    }
  });
}

it('PTS-101 operation form forwards candidateOverflow into the GPU contributor', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  try {
    const getView = createOperationTestViewGetter(graph, {
      offsets: {format: 'uint32', length: 4},
      neighbors: {format: 'uint32', length: 2},
      weights: {format: 'float32', length: 2},
      weightsCandidateOverflow: {format: 'uint32', length: 1},
      values: {format: 'float32', length: 3},
      table: {format: 'float32', length: 6}
    });
    const operation = new GPUNeighborhoodSummaryOperation({
      rowCount: 3,
      weights: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        weights: 'weights',
        status: {candidateOverflow: 'candidate-loss'}
      },
      values: 'values',
      statistics: ['count', 'mean'],
      output: {table: 'table'}
    });
    expect(operation.inputs).toHaveProperty('weightsCandidateOverflow', 'candidate-loss');
    const nodes = operation.createContributor(getView).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(getView('offsets'), OFFSETS);
    memory.write(getView('neighbors'), NEIGHBORS);
    memory.write(getView('weights'), WEIGHTS);
    memory.write(getView('values'), VALUES);
    for (const candidateOverflow of [1, 0]) {
      memory.write(getView('weightsCandidateOverflow'), Uint32Array.of(candidateOverflow));
      memory.write(getView('table'), new Float32Array(6).fill(-7));
      executeCommandNodesOnCPU(nodes, memory);
      expectSameNumbers(
        memory.readFloat32(getView('table')),
        candidateOverflow ? new Array(6).fill(NaN) : ORDINARY_TABLE
      );
    }
  } finally {
    device.destroy();
  }
});

it('PTS-101 CPU twin and CPU operation suppress on candidateOverflow alone', () => {
  const input = {offsets: OFFSETS, neighbors: NEIGHBORS, weights: WEIGHTS, values: VALUES};
  const options = {statistics: ['count', 'mean'] as const};
  for (const candidateOverflow of [1, 0]) {
    const result = computeNeighborhoodSummaryOnCPU(
      {
        ...input,
        categories: CATEGORIES,
        status: {
          overflow: Uint32Array.of(0),
          candidateOverflow: Uint32Array.of(candidateOverflow)
        }
      },
      options
    );
    expectSameNumbers(result.table, candidateOverflow ? new Array(6).fill(NaN) : ORDINARY_TABLE);
    expect(Array.from(result.modes)).toEqual(candidateOverflow ? NO_MODES : ORDINARY_MODES);
  }
  const rows = (values: Float32Array | Uint32Array) => ({
    format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const operation = new CPUNeighborhoodSummaryOperation({
    rowCount: 3,
    weights: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: {candidateOverflow: 'candidate-loss'}
    },
    values: 'values',
    statistics: ['count', 'mean'],
    output: {table: 'table'}
  });
  for (const candidateOverflow of [1, 0]) {
    const result = operation.execute({
      inputs: {
        offsets: rows(OFFSETS),
        neighbors: rows(NEIGHBORS),
        weights: rows(WEIGHTS),
        weightsCandidateOverflow: rows(Uint32Array.of(candidateOverflow)),
        values: rows(VALUES)
      }
    }) as Record<string, Float32Array>;
    expectSameNumbers(result['table'], candidateOverflow ? new Array(6).fill(NaN) : ORDINARY_TABLE);
  }
});
