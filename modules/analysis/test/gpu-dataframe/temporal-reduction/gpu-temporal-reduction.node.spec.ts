// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUTemporalReductionParameterValues,
  getGPUTemporalReductionWordParameterValues,
  GPUTemporalReduction,
  type GPUTemporalReductionProps
} from '../../../src/dataframe-analysis/temporal-reduction/index';
import {getOrderedFloatKey, reduceTemporalBucketsOnCPU} from '../../oracles/temporal-reduction-cpu';
import {ADVERSARIAL_WIDTHS, createAdversarialEdgeScene} from './adversarial-edges';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUTemporalReductionProps> = {}
): GPUTemporalReductionProps {
  const view = <Format extends 'uint32' | 'float32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    cellIds: view('uint32', 10),
    timestamps: view('float32', 10),
    values: view('float32', 10),
    parameters: view('float32', 2),
    cellCount: 3,
    bucketCount: 4,
    output: {
      counts: view('uint32', 12),
      minimums: view('float32', 12),
      maximums: view('float32', 12),
      first: view('float32', 12),
      last: view('float32', 12),
      occupiedSlots: {
        ids: view('uint32', 5),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      }
    },
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUTemporalReductionProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUTemporalReduction(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('final1b F1 rejects candidate publication into temporal cell IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const base = createProps(graph);
    const cells = createTransientView(graph, 'alias-cells', 'uint32', 10);
    expect(() =>
      new GPUTemporalReduction({
        ...base,
        cellIds: cells,
        output: {
          ...base.output,
          occupiedSlots: {
            ...base.output.occupiedSlots,
            status: {
              ...base.output.occupiedSlots.status,
              candidateOverflow: graph.createDataView(cells.buffer, {format: 'uint32', length: 1})
            }
          }
        }
      }).getCommandNodes(graph)
    ).toThrow(/share buffers|overlap/);
  } finally {
    device.destroy();
  }
});

it('GPUTemporalReduction parameter helpers pack layouts', () => {
  expect(Array.from(getGPUTemporalReductionParameterValues(2.5, 4))).toEqual([2.5, 4]);
  expect(Array.from(getGPUTemporalReductionWordParameterValues(-1n, 3600000))).toEqual([
    0xffffffff, 0xffffffff, 3600000, 0
  ]);
  expect(() => getGPUTemporalReductionWordParameterValues(0, 0)).toThrow(/width/);
  // Widths of 2^32 and more fill the second width word (nanosecond hours, 2^64 - 1).
  // 3.6e12 = 838 * 2^32 + 817405952
  expect(Array.from(getGPUTemporalReductionWordParameterValues(0, 3_600_000_000_000))).toEqual([
    0, 0, 817_405_952, 838
  ]);
  expect(Array.from(getGPUTemporalReductionWordParameterValues(0, 2 ** 32))).toEqual([0, 0, 0, 1]);
  expect(Array.from(getGPUTemporalReductionWordParameterValues(0, 0xffffffffffffffffn))).toEqual([
    0, 0, 0xffffffff, 0xffffffff
  ]);
  expect(() => getGPUTemporalReductionWordParameterValues(0, 2n ** 64n)).toThrow(/width/);
  expect(() => getGPUTemporalReductionWordParameterValues(0, 0n)).toThrow(/width/);
  expect(() => getGPUTemporalReductionWordParameterValues(0, 1.5)).toThrow(/width/);
  expect(() => getGPUTemporalReductionWordParameterValues(0.5, 1)).toThrow(/origin/);
  expect(() => getGPUTemporalReductionParameterValues(0, 1, new Float32Array(1))).toThrow(/hold/);
});

it('getOrderedFloatKey preserves float order', () => {
  const values = [-Infinity, -1e30, -2, -0.5, -0, 0, 1e-30, 0.5, 2, 1e30, Infinity];
  const keys = values.map(getOrderedFloatKey);
  for (let index = 1; index < keys.length; index++) {
    expect(keys[index]).toBeGreaterThan(keys[index - 1]);
  }
});

it('reduceTemporalBucketsOnCPU reduces, breaks ties by row, and drops invalid rows', () => {
  const result = reduceTemporalBucketsOnCPU({
    // rows: 0..5 in cell 0 bucket 0 (times 0..), row 6 skipped cell, row 7 masked, row 8 NaN value,
    // row 9 out of range
    cellIds: [0, 0, 0, 0, 1, 1, 0xffffffff, 0, 0, 0],
    timestamps: Float32Array.from([1, 0, 0, 3, 5, 5, 0, 0, 0, 99]),
    values: Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, NaN, 90]),
    mask: [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    cellCount: 2,
    bucketCount: 2,
    origin: 0,
    width: 4
  });
  // cell 0 bucket 0: rows 0..3 -> first is time 0 with lowest row (1), last is time 3 (row 3).
  expect(result.count[0]).toBe(4);
  expect([result.minimums[0], result.maximums[0], result.first[0], result.last[0]]).toEqual([
    10, 40, 20, 40
  ]);
  // cell 1 bucket 1: rows 4 and 5 tie on time 5, both first and last are the lower row (50).
  expect(result.count[3]).toBe(2);
  expect([result.minimums[3], result.maximums[3], result.first[3], result.last[3]]).toEqual([
    50, 60, 50, 50
  ]);
  expect(result.occupiedSlots).toEqual([0, 3]);
  expect(result.minimums[1]).toBeNaN();
});

it('reduceTemporalBucketsOnCPU word mode is exact beyond f32', () => {
  const origin = 1_700_000_000_000n;
  const result = reduceTemporalBucketsOnCPU({
    cellIds: [0, 0, 0, 0],
    timestamps: BigInt64Array.from([
      origin + 1n,
      origin + 3_599_999n,
      origin + 3_600_000n,
      origin - 1n
    ]),
    values: Float32Array.from([1, 2, 3, 4]),
    cellCount: 1,
    bucketCount: 2,
    origin,
    width: 3_600_000
  });
  expect(Array.from(result.count)).toEqual([2, 1]);
  expect([result.first[0], result.last[0]]).toEqual([1, 2]);
});

it('GPUTemporalReduction rejects invalid properties', () => {
  expectThrows(() => ({cellCount: 0}), /cellCount/);
  expectThrows(() => ({bucketCount: 1.5}), /bucketCount/);
  expectThrows(() => ({cellCount: 70000, bucketCount: 70000}), /exceed/);
  expectThrows(
    graph => ({
      cellIds: createTransientView(graph, 'short-cells', 'uint32', 9)
    }),
    /cellIds length/
  );
  expectThrows(
    graph => ({
      parameters: createTransientView(graph, 'short-params', 'float32', 1)
    }),
    /parameters must hold/
  );
  expectThrows(
    graph => ({
      timestamps: createTransientView(graph, 'words', 'uint32x2', 10)
    }),
    /parameters/
  );
  expectThrows(
    graph => ({
      cellCount: 100,
      bucketCount: 100,
      timestamps: createTransientView(graph, 'words2', 'uint32x2', 10),
      parameters: createTransientView(graph, 'word-params', 'uint32', 3)
    }),
    /parameters must hold 4/
  );
  expectThrows(graph => {
    const base = createProps(graph);
    return {
      output: {
        ...base.output,
        minimums: createTransientView(graph, 'short-min', 'float32', 11)
      }
    };
  }, /output.minimums/);
  expectThrows(graph => {
    const base = createProps(graph);
    return {output: {...base.output, counts: base.cellIds as never}};
  }, /output.counts/);
});

it('GPUTemporalReduction creates deterministic node IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUTemporalReduction(createProps(graph));
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  for (const step of [
    'init',
    'classify',
    'extremes',
    'tie-rows',
    'finish-values',
    'finish-extrema',
    'publish'
  ]) {
    expect(ids).toContain(`temporal-reduction-${step}`);
  }
  expect(contributor.slotCount).toBe(12);
  device.destroy();
});

it('reduceTemporalBucketsOnCPU puts fround(k * width) exactly in bucket k', () => {
  for (const width of ADVERSARIAL_WIDTHS) {
    const scene = createAdversarialEdgeScene(width);
    const result = reduceTemporalBucketsOnCPU({...scene, origin: 0, width});
    for (const [row, edge] of scene.edgeBuckets.entries()) {
      if (edge !== undefined) {
        expect(result.count[row * scene.bucketCount + edge], `width ${width} edge ${edge}`).toBe(1);
        // The f32 neighbor just below belongs to the previous bucket.
        expect(result.count[(row - 1) * scene.bucketCount + edge - 1]).toBe(1);
      }
    }
  }
});

it('ROAD-103 word slot declarations isolate the eight-byte storage binding boundary', () => {
  const slotCount = 512;
  for (const bindingBytes of [8 * slotCount, 8 * slotCount - 1]) {
    const device = createNullWebGPUDevice();
    // Same logical layout on both devices. Only the eight-byte time-key slot binding binds:
    // row times=8 bytes, other slot columns=2048 bytes, maxBufferSize=1 GiB.
    Object.assign(device.limits, {
      maxStorageBufferBindingSize: bindingBytes,
      maxBufferSize: 2 ** 30
    });
    try {
      const graph = new GPUCommandGraph(device);
      const view = <Format extends 'uint32' | 'uint32x2' | 'float32'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `boundary-${serial++}`, format, length);
      const contributor = new GPUTemporalReduction({
        cellIds: view('uint32', 1),
        timestamps: view('uint32x2', 1),
        values: view('float32', 1),
        parameters: view('uint32', 4),
        cellCount: 8,
        bucketCount: 64,
        output: {
          counts: view('uint32', slotCount),
          minimums: view('float32', slotCount),
          maximums: view('float32', slotCount),
          first: view('float32', slotCount),
          last: view('float32', slotCount),
          occupiedSlots: {
            ids: view('uint32', slotCount),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          }
        }
      });
      expect(contributor.slotCount).toBe(slotCount);
      if (bindingBytes === 8 * slotCount)
        expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
      else
        expect(() => contributor.getCommandNodes(graph)).toThrow(
          /^temporal-reduction cellCount \* bucketCount must be at most 511 with uint32x2 times$/
        );
    } finally {
      device.destroy();
    }
  }
});

it('ROAD-103 rejects logical slot products above 0xfffffffe at construction', () => {
  expectThrows(
    () => ({cellCount: 0xffffffff, bucketCount: 1}),
    /^temporal-reduction cellCount \* bucketCount must not exceed 4294967294$/
  );
});
