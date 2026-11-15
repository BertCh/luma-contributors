// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUChunkedSpatialWeightsTranspose,
  type GPUChunkedSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from './spatial-weights-harness';
import type {OracleCSR} from './spatial-weights-oracle';

type Device = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

/** Column-first CPU oracle: directly inspects each original CSR row, without sort or scan. */
function transposeOnCPU(input: OracleCSR, columns: number): OracleCSR {
  const result: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
  for (let column = 0; column < columns; column++) {
    for (let row = 0; row + 1 < input.offsets.length; row++) {
      for (let slot = input.offsets[row]; slot < input.offsets[row + 1]; slot++) {
        if (input.neighbors[slot] === column) {
          result.neighbors.push(row);
          result.weights.push(input.weights[slot]);
          result.distances.push(input.distances[slot]);
        }
      }
    }
    result.offsets.push(result.neighbors.length);
  }
  return result;
}

/** Independent dense union-of-patterns symmetry count, including a present zero weight. */
function countAsymmetryOnCPU(input: OracleCSR, tolerance: number): number {
  const rows = input.offsets.length - 1;
  const dense: (number | undefined)[][] = Array.from({length: rows}, () => Array(rows));
  for (let row = 0; row < rows; row++) {
    for (let slot = input.offsets[row]; slot < input.offsets[row + 1]; slot++) {
      if (input.neighbors[slot] < rows) dense[row][input.neighbors[slot]] = input.weights[slot];
    }
  }
  let mismatches = 0;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < rows; column++) {
      const first = dense[row][column];
      const second = dense[column][row];
      if (first === undefined && second === undefined) continue;
      if (first === undefined || second === undefined || Math.abs(first - second) > tolerance)
        mismatches++;
    }
  }
  return mismatches;
}

function sliceRows(input: OracleCSR, rowStart: number, rowCount: number): OracleCSR {
  const begin = input.offsets[rowStart];
  const end = input.offsets[rowStart + rowCount];
  return {
    offsets: input.offsets.slice(rowStart, rowStart + rowCount + 1).map(value => value - begin),
    neighbors: input.neighbors.slice(begin, end),
    weights: input.weights.slice(begin, end),
    distances: input.distances.slice(begin, end)
  };
}

async function checkTranspose(
  device: Device,
  input: OracleCSR,
  inputRows: number[],
  outputRows: number[],
  capacities: number[],
  options: {symmetry?: boolean; tolerance?: number; distances?: boolean; repeat?: boolean} = {}
): Promise<void> {
  const rig = new WeightsRig(device);
  const columns = outputRows.reduce((sum, count) => sum + count, 0);
  const expected = transposeOnCPU(input, columns);
  let rowStart = 0;
  const weights: GPUChunkedSpatialWeights = {
    rowCount: input.offsets.length - 1,
    chunks: inputRows.map(rowCount => {
      // Real padding contains valid column IDs. It must never become a transposed slot.
      const chunkWeights = rig.uploadWeights(sliceRows(input, rowStart, rowCount), 3);
      // Empty input chunks still carry a distance view when the caller requests distances.
      if (!chunkWeights.distances && options.distances !== false) {
        const capacity = chunkWeights.neighbors.length;
        chunkWeights.distances = rig.input(new Float32Array(capacity), 'float32', capacity);
      }
      const chunk = {rowStart, rowCount, weights: chunkWeights};
      rowStart += rowCount;
      return chunk;
    })
  };
  expect(rowStart).toBe(weights.rowCount);
  let outputStart = 0;
  const outputs = outputRows.map((rowCount, index) => {
    const capacity = capacities[index];
    const offsets = rig.output('uint32', rowCount + 1);
    const neighbors = rig.output('uint32', capacity);
    const values = rig.output('float32', capacity);
    const distances = options.distances === false ? undefined : rig.output('float32', capacity);
    const count = rig.output('uint32', 1);
    const requiredCount = rig.output('uint32', 1);
    const overflow = rig.output('uint32', 1);
    const chunk = {
      rowStart: outputStart,
      rowCount,
      weights: {
        offsets: offsets.view,
        neighbors: neighbors.view,
        weights: values.view,
        distances: distances?.view,
        status: {count: count.view, requiredCount: requiredCount.view, overflow: overflow.view}
      }
    };
    outputStart += rowCount;
    return {chunk, capacity, offsets, neighbors, values, distances, count, requiredCount, overflow};
  });
  const asymmetricSlots = options.symmetry ? rig.output('uint32', 1) : undefined;
  const transpose = new GPUChunkedSpatialWeightsTranspose({
    weights,
    columnCount: columns,
    symmetryTolerance: options.tolerance,
    output: {chunks: outputs.map(output => output.chunk), asymmetricSlots: asymmetricSlots?.view}
  });
  const check = async (oracle: OracleCSR) => {
    for (const output of outputs) {
      const {chunk, capacity} = output;
      const begin = oracle.offsets[chunk.rowStart];
      const required = oracle.offsets[chunk.rowStart + chunk.rowCount] - begin;
      const retained = Math.min(required, capacity);
      expect(await readUint32(output.offsets.buffer, chunk.rowCount + 1)).toEqual(
        oracle.offsets
          .slice(chunk.rowStart, chunk.rowStart + chunk.rowCount + 1)
          .map(value => Math.min(value - begin, capacity))
      );
      const pad = (values: number[]) => [
        ...values.slice(begin, begin + retained),
        ...Array(capacity - retained).fill(0)
      ];
      expect(await readUint32(output.neighbors.buffer, capacity)).toEqual(pad(oracle.neighbors));
      expect(await readFloat32(output.values.buffer, capacity)).toEqual(pad(oracle.weights));
      if (output.distances)
        expect(await readFloat32(output.distances.buffer, capacity)).toEqual(pad(oracle.distances));
      expect(await readUint32(output.count.buffer, 1)).toEqual([retained]);
      expect(await readUint32(output.requiredCount.buffer, 1)).toEqual([required]);
      expect(await readUint32(output.overflow.buffer, 1)).toEqual([Number(required > capacity)]);
    }
  };
  rig.graph.add(transpose);
  const compiled = rig.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    await check(expected);
    if (asymmetricSlots) {
      expect(await readUint32(asymmetricSlots.buffer, 1)).toEqual([
        countAsymmetryOnCPU(input, options.tolerance ?? 0)
      ]);
    }
    if (options.repeat) {
      // Re-encode the same nodes after every input becomes empty. Counts, flags and tails reset.
      for (const chunk of weights.chunks) {
        rig.bufferOf(chunk.weights.offsets).write(new Uint32Array(chunk.rowCount + 1));
      }
      submitGraph(device, compiled, undefined);
      await check({offsets: Array(columns + 1).fill(0), neighbors: [], weights: [], distances: []});
      if (asymmetricSlots) expect(await readUint32(asymmetricSlots.buffer, 1)).toEqual([0]);
    }
  } finally {
    compiled.destroy();
    rig.destroy();
  }
}

const HUB: OracleCSR = {
  offsets: [0, 2, 2, 4, 5, 7, 9, 9],
  neighbors: [2, 5, 0, 5, 2, 2, 5, 0, 2],
  weights: [1, 4, 2, 2, 3, 4, 1, 4, 2],
  distances: [2, 5, 2, 3, 1, 2, 1, 5, 3]
};

it('matches a CPU transpose across unrelated input/output boundaries, empty rows, tied columns and padding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const inputRows of [[7], [1, 1, 2, 3], [2, 3, 1, 1]]) {
    await checkTranspose(device, HUB, inputRows, [2, 1, 4], [4, 5, 6], {symmetry: true});
  }
  await checkTranspose(device, HUB, [3, 4], [1, 1, 1, 1, 1, 1, 1], [3, 1, 5, 1, 1, 4, 1], {
    distances: false
  });
});

it('clamps each chunk independently, including a partially retained hub row, and keeps symmetry exact on overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  await checkTranspose(device, HUB, [1, 3, 3], [2, 1, 4], [1, 2, 1], {symmetry: true});
  // The first chunk overflows although total output capacity is larger than the required total.
  await checkTranspose(device, HUB, [2, 5], [3, 4], [1, 30], {
    symmetry: true,
    tolerance: 1,
    repeat: true
  });
});

it('handles rectangular CSR, drops out-of-range columns and leaves unreachable chunks empty', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rectangular: OracleCSR = {
    offsets: [0, 3, 3, 5, 6, 6],
    neighbors: [0, 4, 99, 1, 4, 4],
    weights: [0, 2, 3, 4, 5, 6],
    distances: [1, 2, 3, 4, 5, 6]
  };
  await checkTranspose(device, rectangular, [1, 1, 3], [2, 3, 4], [4, 5, 1]);
  await checkTranspose(
    device,
    {offsets: [0, 0, 0, 0], neighbors: [], weights: [], distances: []},
    [1, 2],
    [1, 1, 3],
    [1, 2, 3],
    {distances: false}
  );
});

it('matches the CPU oracle through radix sorts and multi-level scans', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 137;
  const columns = 521;
  const input: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      if ((row * 17 + column * 13) % 29 === 0) {
        input.neighbors.push(column);
        input.weights.push((row + column) / 4);
        input.distances.push((row + 2 * column) / 8);
      }
    }
    input.offsets.push(input.neighbors.length);
  }
  await checkTranspose(device, input, [65, 72], [257, 1, 263], [1300, 10, 1300]);
});

it('counts symmetry on the union of patterns, including missing reverse zero weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const input: OracleCSR = {
    offsets: [0, 1, 2, 3],
    neighbors: [1, 0, 0],
    weights: [2, 2.5, 0],
    distances: [1, 1, 2]
  };
  for (const tolerance of [0, 0.5]) {
    await checkTranspose(device, input, [1, 2], [2, 1], [5, 2], {symmetry: true, tolerance});
  }
});

it('reports unknown normalization after truncating double-standardized weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const input = rig.uploadWeights({
    offsets: [0, 1, 2],
    neighbors: [1, 0],
    weights: [0.5, 0.5],
    distances: []
  });
  const weights: GPUChunkedSpatialWeights = {
    rowCount: 2,
    information: {rowCount: 2, transform: 'double', symmetric: true},
    chunks: [{rowStart: 0, rowCount: 2, weights: input}]
  };
  const output = rig.weightsOutput(2, 1);
  const count = rig.output('uint32', 1);
  const requiredCount = rig.output('uint32', 1);
  const overflow = rig.output('uint32', 1);
  const transpose = new GPUChunkedSpatialWeightsTranspose({
    weights,
    output: {
      chunks: [
        {
          rowStart: 0,
          rowCount: 2,
          weights: {
            ...output.spatialWeights,
            status: {count: count.view, requiredCount: requiredCount.view, overflow: overflow.view}
          }
        }
      ]
    }
  });
  try {
    rig.run(transpose);
    const actual = await output.read();
    expect(actual.offsets).toEqual([0, 1, 1]);
    expect(actual.neighbors).toEqual([1]);
    expect(actual.weights.reduce((sum, weight) => sum + weight, 0)).toBe(0.5);
    expect(await readUint32(count.buffer, 1)).toEqual([1]);
    expect(await readUint32(requiredCount.buffer, 1)).toEqual([2]);
    expect(await readUint32(overflow.buffer, 1)).toEqual([1]);
    expect(transpose.outputWeights.information?.transform).toBe('unknown');
    expect(transpose.outputWeights.chunks[0].weights.information?.transform).toBe('unknown');
  } finally {
    rig.destroy();
  }
});

for (const layout of ['packed', 'chunked'] as const) {
  it(`feeds corrected ${layout} output views through a saved description to a downstream kernel`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    const weights: GPUChunkedSpatialWeights = {
      rowCount: 2,
      chunks: [
        {
          rowStart: 0,
          rowCount: 2,
          weights: rig.uploadWeights({
            offsets: [0, 1, 2],
            neighbors: [1, 0],
            weights: [2, 3],
            distances: []
          })
        }
      ]
    };
    const destination = rig.weightsOutput(2, 2);
    const count = rig.output('uint32', 1);
    const requiredCount = rig.output('uint32', 1);
    const overflow = rig.output('uint32', 1);
    const output =
      layout === 'packed'
        ? destination.spatialWeights
        : {
            chunks: [
              {
                rowStart: 0,
                rowCount: 2,
                weights: {
                  ...destination.spatialWeights,
                  status: {
                    count: count.view,
                    requiredCount: requiredCount.view,
                    overflow: overflow.view
                  }
                }
              }
            ]
          };
    const transpose = new GPUChunkedSpatialWeightsTranspose({weights, output});
    const description = transpose.outputWeights;
    const describedWeights = 'chunks' in description ? description.chunks[0].weights : description;
    const corrected = rig.output('uint32', 2);
    const target = 'chunks' in output ? output.chunks[0].weights : output;
    target.neighbors = corrected.view;
    const copied = rig.output('uint32', 2);
    try {
      rig.run(transpose, {
        getCommandNodes: graph => [
          createWGSLKernelNode(graph, {
            id: 'read-transpose-description',
            operation: 'TransposeDescriptionTest',
            bindings: [
              {name: 'neighbors', view: describedWeights.neighbors, type: 'u32', access: 'read'},
              {name: 'copied', view: copied.view, type: 'u32', access: 'read_write'}
            ],
            invocationCount: 2,
            body: 'copied[copiedOffset + index] = neighbors[neighborsOffset + index];'
          })
        ]
      });
      // Hand-derived transpose: row 0 receives source 1; row 1 receives source 0.
      expect(await readUint32(copied.buffer, 2)).toEqual([1, 0]);
      expect(describedWeights.neighbors).toBe(corrected.view);
      expect(transpose.outputWeights).toBe(description);
    } finally {
      rig.destroy();
    }
  });
}
