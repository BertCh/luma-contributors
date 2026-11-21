// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {executeWeightsAuditKernel} from './weights-audit-executor';
import {GPUChunkedSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose';
import {
  TRANSPOSE_AUDIT_INPUT as INPUT,
  TRANSPOSE_AUDIT_EXPECTED as EXPECTED
} from './weights-transpose-audit-fixture';

it('packed transpose refuses destinations it cannot write and inherits input loss', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `destination-${serial++}`, format, length);
    const weights = {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 2),
      weights: view('float32', 2)
    };
    const output = {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 2),
      weights: view('float32', 2)
    };
    expect(
      () =>
        new GPUSpatialWeightsTranspose({
          weights,
          output: {...output, distances: view('float32', 2)}
        })
    ).toThrow(/output.distances/);
    for (const field of [
      'overflow',
      'candidateOverflow',
      'count',
      'requiredCount',
      'invalidCount'
    ] as const) {
      expect(
        () =>
          new GPUSpatialWeightsTranspose({
            weights,
            output: {...output, status: {[field]: view('uint32', 1)}}
          })
      ).toThrow(/output.status/);
    }
    const overflow = view('uint32', 1),
      candidateOverflow = view('uint32', 1);
    const transpose = new GPUSpatialWeightsTranspose({
      weights: {...weights, status: {overflow, candidateOverflow}},
      output
    });
    expect(transpose.outputWeights.status).toMatchObject({overflow, candidateOverflow});
  } finally {
    device.destroy();
  }
});

it('rectangular packed transpose omits input cardinalities after dropping columns in both source layouts', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const chunked of [false, true]) {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
        createTransientView(graph, `count-${serial++}`, format, length);
      const status = {
        count: view('uint32', 1),
        requiredCount: view('uint32', 1),
        overflow: view('uint32', 1),
        candidateOverflow: view('uint32', 1)
      };
      const weights = {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 2),
        weights: view('float32', 2),
        status
      };
      const output = {
        offsets: view('uint32', 2),
        neighbors: view('uint32', 2),
        weights: view('float32', 2)
      };
      const transpose = chunked
        ? new GPUChunkedSpatialWeightsTranspose({
            weights: {
              rowCount: 2,
              chunks: [{rowStart: 0, rowCount: 2, weights: {...weights, status: undefined}}],
              status
            },
            columnCount: 1,
            output
          })
        : new GPUSpatialWeightsTranspose({weights, columnCount: 1, output});
      expect(transpose.outputWeights.status?.count).toBeUndefined();
      expect(transpose.outputWeights.status?.requiredCount).toBeUndefined();
      expect(transpose.outputWeights.status?.overflow).toBe(status.overflow);
      expect(transpose.outputWeights.status?.candidateOverflow).toBe(status.candidateOverflow);
    }
  } finally {
    device.destroy();
  }
});

it('transpose first-party kernels match SciPy sparse literals, including zero slots and tolerance ties', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const [columns, scale] of [
      [8, 1],
      [8, 2],
      [5, 1],
      [5, 2]
    ]) {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
        createTransientView(graph, `transpose-audit-${serial++}`, format, length);
      const overflow = view('uint32', 1);
      const transpose = new GPUSpatialWeightsTranspose({
        weights: {
          offsets: view('uint32', 9),
          neighbors: view('uint32', 15),
          weights: view('float32', 15),
          distances: view('float32', 15),
          status: {overflow}
        },
        columnCount: columns,
        output: {
          offsets: view('uint32', columns + 1),
          neighbors: view('uint32', 15),
          weights: view('float32', 15),
          distances: view('float32', 15),
          asymmetricSlots: columns === 8 ? view('uint32', 1) : undefined
        }
      });
      expect(transpose.outputWeights.status?.overflow).toBe(overflow);
      const records = getGPUCommandNodeKernelSources(transpose.getCommandNodes(graph));
      const buffers = {
        offsets: Uint32Array.from(INPUT.offsets),
        neighbors: Uint32Array.from([...INPUT.neighbors, 0, 0]),
        weights: Float32Array.from([...INPUT.weights.map(value => value * scale), -99, -99]),
        distances: Float32Array.from([...INPUT.distances, -99, -99]),
        slotKeys: new Uint32Array(15),
        slotIndices: new Uint32Array(15),
        sortedKeys: new Uint32Array(15),
        sortedSlots: new Uint32Array(15),
        columnCounts: new Uint32Array(columns),
        columnFill: new Uint32Array(columns),
        columnStarts: new Uint32Array(columns),
        destinations: new Uint32Array(15),
        outputOffsets: new Uint32Array(columns + 1).fill(99),
        outputNeighbors: new Uint32Array(15).fill(99),
        outputWeights: new Float32Array(15).fill(-99),
        outputDistances: new Float32Array(15).fill(-99),
        rowMismatches: new Uint32Array(8),
        tolerance: Float32Array.of(0)
      };
      const execute = (suffix: string) =>
        executeWeightsAuditKernel(
          records.find(record => record.id.endsWith(`-${suffix}`))!,
          buffers
        );
      execute('clear-columns');
      execute('clear-output');
      execute('slot-keys');
      // Replace only upstream GPUSort and GPUScan with their public definitions.
      const order = Array.from({length: 15}, (_, index) => index).sort(
        (left, right) => buffers.slotKeys[left] - buffers.slotKeys[right]
      );
      buffers.sortedKeys.set(order.map(index => buffers.slotKeys[index]));
      buffers.sortedSlots.set(order.map(index => buffers.slotIndices[index]));
      Object.assign(buffers, {columnTarget: buffers.columnCounts});
      execute('count-columns');
      let count = 0;
      buffers.columnCounts.forEach((value, index) => {
        buffers.columnStarts[index] = count;
        count += value;
      });
      execute('offsets');
      execute('destinations');
      execute('scatter');
      const used = columns === 8 ? 12 : 10;
      expect(Array.from(buffers.outputOffsets)).toEqual(EXPECTED.offsets.slice(0, columns + 1));
      for (const [actual, expected] of [
        [buffers.outputNeighbors, EXPECTED.neighbors],
        [buffers.outputWeights, EXPECTED.weights.map(value => value * scale)],
        [buffers.outputDistances, EXPECTED.distances]
      ] as const) {
        expect(Array.from(actual)).toEqual([
          ...expected.slice(0, used),
          ...Array(15 - used).fill(0)
        ]);
      }
      if (columns === 8) {
        Object.assign(buffers, {
          transposeOffsets: buffers.outputOffsets,
          transposeNeighbors: buffers.outputNeighbors,
          transposeWeights: buffers.outputWeights
        });
        for (const [tolerance, expected] of [
          [0, 6],
          [1, 4]
        ]) {
          buffers.tolerance[0] = tolerance * scale;
          execute('symmetry-rows');
          expect(buffers.rowMismatches.reduce((total, value) => total + value, 0)).toBe(expected);
        }
      }
    }
  } finally {
    device.destroy();
  }
});
